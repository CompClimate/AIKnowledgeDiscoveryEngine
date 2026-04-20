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
        
        self.output_net = nn.Sequential(
            nn.Linear(n_concepts*output_dim, output_dim),
            nn.Sigmoid()
        )

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
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
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
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
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
    def __init__(self, n_features, n_concepts, output_dim, n_free_concepts=0, channels_list=[32, 64, 128, 256]):  # wider: [64, 128, 256, 512]
        super().__init__()
        self.n_features = n_features  # this is actually T*V in PointwiseCBM
        self.n_concepts = n_concepts
        self.n_free_concepts = n_free_concepts
        self.output_dim = output_dim

        # encoder takes n_features as input (which is T*V concatenated)
        self.encoder = Encoder(n_features, channels_list)

        bottleneck_ch = channels_list[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # decoder (same number of stages as encoder)
        decoder_channels = [channels_list[-1]] + channels_list[::-1]
        self.decoder = Decoder(decoder_channels[0], decoder_channels[1:], channels_list)

        # supervised concept prediction head
        self.concept_head = nn.Conv2d(channels_list[0], n_concepts * output_dim, kernel_size=3, padding=1)

        # free (unsupervised) concept head
        if n_free_concepts > 0:
            self.free_concept_head = nn.Conv2d(channels_list[0], n_free_concepts * output_dim, kernel_size=3, padding=1)

        # output head: linear combination of supervised + free concepts
        total_concept_channels = (n_concepts + n_free_concepts) * output_dim
        self.output_head = nn.Conv2d(total_concept_channels, output_dim, kernel_size=1)

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

        # Supervised concept predictions (UNet path only)
        concepts = self.concept_head(x)  # (B, n_concepts*output_dim, Y_padded, X_padded)

        # Free concept predictions
        if self.n_free_concepts > 0:
            free = self.free_concept_head(x)  # (B, n_free*output_dim, Y, X)
        else:
            free = None

        # Additive: free concept is extra linear channel in output head
        all_concepts = torch.cat([concepts, free], dim=1) if free is not None else concepts
        output = self.output_head(all_concepts)
        pred_sup = output
        pred_free = None

        # Crop to original spatial dimensions
        concepts = concepts[:, :, :Y, :X]
        pred_sup = pred_sup[:, :, :Y, :X]
        if pred_free is not None:
            pred_free = pred_free[:, :, :Y, :X]
        output = output[:, :, :Y, :X]

        # Reshape supervised concepts
        concepts = concepts.view(B, self.n_concepts, self.output_dim, Y, X)

        output = output.unsqueeze(1)  # (B, 1, output_dim, Y, X)

        # Reshape free concepts if present
        if free is not None:
            free = free[:, :, :Y, :X]
            free = free.view(B, self.n_free_concepts, self.output_dim, Y, X)

        # Unsqueeze pred_sup/pred_free to match output's (B, 1, output_dim, Y, X) shape
        pred_sup = pred_sup.unsqueeze(1)
        if pred_free is not None:
            pred_free = pred_free.unsqueeze(1)

        return output, concepts, free 
