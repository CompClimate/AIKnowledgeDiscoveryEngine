import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn

class PointwiseCBM(nn.Module):
    def __init__(self, n_features, n_concepts, hidden_dim, output_dim):
        super().__init__()
        self.n_features = n_features
        self.n_concepts = n_concepts
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.concept_net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_concepts*output_dim),
            #nn.Sigmoid()  # concepts typically in [0,1]
        )
        
        self.output_net = nn.Linear(n_concepts*output_dim, output_dim)

    def forward(self, x):
        B, V, T, Y, X = x.shape
        x = x.permute(0, 3, 4, 1, 2)      # (B, Y, X, T, V)
        x = x.reshape(B * Y * X, T * V)  # flatten spatial points
        
        concepts = self.concept_net(x)    # (B*Y*X, n_concepts)
        concepts = concepts.view(B * Y * X, self.output_dim * self.n_concepts)
        
        output = self.output_net(concepts) # (B*Y*X, output_dim)
        
        # reshape back to grid
        concepts = concepts.view(B, Y, X, self.output_dim, self.n_concepts)
        output = output.view(B, Y, X, self.output_dim, -1)

        concepts = concepts.permute(0, 4, 3, 1, 2)  # (B, n_concepts, lead, Y, X)
        output = output.permute(0, 4, 3, 1, 2)      # (B, output_dim, lead, Y, X)
        
        return output, concepts
    
class PointwiseNoCBM(nn.Module):
    def __init__(self, n_features, n_concepts, hidden_dim, output_dim):
        super().__init__()
        self.n_features = n_features
        self.n_concepts = n_concepts
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.output_net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            #nn.Sigmoid()  # concepts typically in [0,1]
        )

    def forward(self, x):
        B, V, T, Y, X = x.shape
        x = x.permute(0, 3, 4, 1, 2)      # (B, Y, X, T, V)
        x = x.reshape(B * Y * X, T * V)  # flatten spatial points
        
        output = self.output_net(x)    # (B*Y*X, output_dim)
        output = output.view(B * Y * X, self.output_dim)
        
        # reshape back to grid
        output = output.view(B, Y, X, self.output_dim, -1)

        output = output.permute(0, 4, 3, 1, 2)      # (B, output_dim, lead, Y, X)
        concepts = torch.zeros(B, Y, X, self.output_dim, self.n_concepts)
        
        return output, concepts

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels_list, dropout=0.2):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        channels = [in_channels] + out_channels_list

        for i in range(len(out_channels_list)):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                )
            )
            self.pool_layers.append(nn.MaxPool2d(2, 2))

    def forward(self, x):
        skip_connections = []
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            skip_connections.append(x)
            x = pool(x)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels_list, encoder_channels_list, dropout=0.2):
        super().__init__()
        self.upconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        channels = [in_channels] + out_channels_list

        for i in range(len(out_channels_list)):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            skip_ch = encoder_channels_list[-(i + 1)]

            # upsampling
            self.upconv_layers.append(nn.ConvTranspose2d(in_ch, out_ch, 2, 2))

            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                )
            )

    def forward(self, x, skip_connections):
        # reversing encoder skip connections 
        skip_connections = skip_connections[::-1]
        for i, (upconv, conv) in enumerate(zip(self.upconv_layers, self.conv_layers)):
            x = upconv(x)
            skip = skip_connections[i]
            skip = skip[:, :, :x.shape[2], :x.shape[3]]  # reshape to match
            # concataning skip connections 
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
        return x


class UNetCBM(nn.Module):
    def __init__(self, n_features, n_concepts, output_dim, n_free_concepts=0, channels_list=[32, 64, 128, 256]):
        super().__init__()
        self.n_features = n_features  # This is actually T*V in PointwiseCBM
        self.n_concepts = n_concepts
        self.n_free_concepts = n_free_concepts
        self.output_dim = output_dim

        # Encoder takes n_features as input (which is T*V concatenated)
        self.encoder = Encoder(n_features, channels_list)

        # UNet bottleneck
        bottleneck_ch = channels_list[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder (same number of stages as encoder)
        decoder_channels = [channels_list[-1]] + channels_list[::-1]
        self.decoder = Decoder(decoder_channels[0], decoder_channels[1:], channels_list)

        # Supervised concept prediction head
        self.concept_head = nn.Conv2d(channels_list[0], n_concepts * output_dim, kernel_size=3, padding=1)

        # Free (unsupervised) concept head
        if n_free_concepts > 0:
            self.free_concept_head = nn.Conv2d(channels_list[0], n_free_concepts * output_dim, kernel_size=3, padding=1)

        # Output head takes both supervised and free concepts
        total_concept_channels = (n_concepts + n_free_concepts) * output_dim
        self.output_head = nn.Sequential(
            nn.Conv2d(total_concept_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Input shape: (B, V, T, Y, X)
        B, V, T, Y, X = x.shape
        
        # Reshape to (B, V*T, Y, X) to match PointwiseCBM's expected input format
        # PointwiseCBM expects n_features = V*T (concatenated temporal features)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, V, Y, X)
        x = x.reshape(B, V * T, Y, X)  # (B, V*T, Y, X)

        # padding so skip connections and pooling work
        pad_lat = (16 - Y % 16) % 16
        pad_lon = (16 - X % 16) % 16
        pad_left = pad_lat // 2
        pad_right = pad_lat - pad_left
        pad_top = pad_lon // 2
        pad_bottom = pad_lon - pad_top
        # using reflect to maintain smoothness at boundary
        x = nn.functional.pad(x, (pad_top, pad_bottom, pad_left, pad_right), mode='reflect')

        # UNet forward pass
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)

        # Supervised concept predictions
        concepts = self.concept_head(x)  # (B, n_concepts*output_dim, Y, X)

        # Free concept predictions
        if self.n_free_concepts > 0:
            free = self.free_concept_head(x)  # (B, n_free*output_dim, Y, X)
            all_concepts = torch.cat([concepts, free], dim=1)
        else:
            all_concepts = concepts
            free = None

        # Final output using all concept maps
        output = self.output_head(all_concepts)  # (B, output_dim, Y, X)

        # Crop to original spatial dimensions
        concepts = concepts[:, :, :Y, :X]
        output = output[:, :, :Y, :X]

        # Reshape supervised concepts
        concepts = concepts.view(B, self.n_concepts, self.output_dim, Y, X)

        output = output.unsqueeze(1)  # (B, 1, output_dim, Y, X)

        # Reshape free concepts if present
        if free is not None:
            free = free[:, :, :Y, :X]
            free = free.view(B, self.n_free_concepts, self.output_dim, Y, X)

        return output, concepts, free

def test_models_match():
    """Test if PointwiseCBM and UNetCBM produce similar outputs"""
    
    # Hyperparameters
    batch_size = 1
    n_features = 5  # V
    lead_times = 6  # T
    n_concepts = 6
    output_dim = 3
    height = 302  # Y
    width = 400   # X
    hidden_dim = 64
    
    # Create dummy input: (B, V, T, Y, X)
    x = torch.randn(batch_size, n_features, lead_times, height, width)
    
    # Initialize both models
    # Create models with n_features = V*T
    unet = UNetCBM(n_features=n_features*lead_times, n_concepts=n_concepts, output_dim=output_dim)
    pointwise = PointwiseCBM(n_features=n_features*lead_times, n_concepts=n_concepts, hidden_dim=hidden_dim, output_dim=output_dim)
    
    # Set both to eval mode
    pointwise.eval()
    unet.eval()
    
    # Forward pass
    with torch.no_grad():
        output_pw, concepts_pw = pointwise(x)
        output_unet, concepts_unet = unet(x)
    
    # Check shapes
    print("=" * 60)
    print("OUTPUT SHAPES")
    print("=" * 60)
    print(f"Input shape: {x.shape}")
    print(f"\nPointwiseCBM:")
    print(f"  Output shape: {output_pw.shape}")
    print(f"  Concepts shape: {concepts_pw.shape}")
    print(f"\nUNetCBM:")
    print(f"  Output shape: {output_unet.shape}")
    print(f"  Concepts shape: {concepts_unet.shape}")
    
    # Check if shapes match
    print("\n" + "=" * 60)
    print("SHAPE COMPATIBILITY")
    print("=" * 60)
    shapes_match = (output_pw.shape == output_unet.shape and 
                    concepts_pw.shape == concepts_unet.shape)
    print(f"Shapes match: {shapes_match}")
    if not shapes_match:
        print(f"  Output shape mismatch: {output_pw.shape} vs {output_unet.shape}")
        print(f"  Concepts shape mismatch: {concepts_pw.shape} vs {concepts_unet.shape}")
    
    # Check value ranges
    print("\n" + "=" * 60)
    print("VALUE RANGES")
    print("=" * 60)
    print(f"PointwiseCBM output: min={output_pw.min():.4f}, max={output_pw.max():.4f}, mean={output_pw.mean():.4f}")
    print(f"UNetCBM output:      min={output_unet.min():.4f}, max={output_unet.max():.4f}, mean={output_unet.mean():.4f}")
    print(f"\nPointwiseCBM concepts: min={concepts_pw.min():.4f}, max={concepts_pw.max():.4f}, mean={concepts_pw.mean():.4f}")
    print(f"UNetCBM concepts:      min={concepts_unet.min():.4f}, max={concepts_unet.max():.4f}, mean={concepts_unet.mean():.4f}")
    
    # Compute MSE between outputs (they won't be identical, but should be in same ballpark)
    print("\n" + "=" * 60)
    print("OUTPUT COMPARISON")
    print("=" * 60)
    output_mse = torch.mean((output_pw - output_unet) ** 2).item()
    concepts_mse = torch.mean((concepts_pw - concepts_unet) ** 2).item()
    print(f"MSE between outputs:  {output_mse:.6f}")
    print(f"MSE between concepts: {concepts_mse:.6f}")
    print("\nNote: MSE will be high since models have different random weights.")
    print("The important thing is that shapes and formats match!")


def test_with_same_weights():
    """Test with identical input/output to verify interface matches"""
    
    batch_size = 1
    n_features = 2
    lead_times = 3
    n_concepts = 4
    output_dim = 1
    height = 16
    width = 16
    
    x = torch.randn(batch_size, n_features, lead_times, height, width)
    
    unet = UNetCBM(n_features * lead_times, n_concepts, output_dim)
    unet.eval()
    
    with torch.no_grad():
        output, concepts = unet(x)
    
    print("=" * 60)
    print("UNET CBM VERIFICATION")
    print("=" * 60)
    print(f"Input: (B={batch_size}, V={n_features}, T={lead_times}, Y={height}, X={width})")
    print(f"Output: {output.shape} - should be (B, output_dim={output_dim}, T?, Y={height}, X={width})")
    print(f"Concepts: {concepts.shape} - should be (B, n_concepts={n_concepts}, output_dim={output_dim}, Y={height}, X={width})")
    
    # Verify dimensions
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[1] == output_dim, "Output dim mismatch"
    assert concepts.shape[0] == batch_size, "Batch size mismatch"
    assert concepts.shape[1] == n_concepts, "n_concepts mismatch"
    print("\n✓ All assertions passed!")


if __name__ == "__main__":
    print("Testing PointwiseCBM vs UNetCBM\n")
    test_models_match()
    print("\n\n")
    test_with_same_weights()