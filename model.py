import torch
import torch.nn as nn
from torchinfo import summary

# referred to https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/ for skeleton

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels_list):
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
    def __init__(self, in_channels, out_channels_list, encoder_channels_list):
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
    def __init__(self, in_channels, concept_channels, out_channels, channels_list):
        super().__init__()
        self.encoder = Encoder(in_channels, channels_list)

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

        # Concept prediction head
        self.concept_head = nn.Conv2d(channels_list[0], concept_channels, kernel_size=3, padding=1)

        # fcn head 
        self.output_head = nn.Sequential(
            nn.Conv2d(concept_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):

        # padding so skip connections and pooling work
        lat, lon = x.shape[2], x.shape[3]
        pad_lat = (16 - lat % 16) % 16
        pad_lon = (16 - lon % 16) % 16
        pad_left = pad_lat // 2
        pad_right = pad_lon - pad_left
        pad_top = pad_lat // 2
        pad_bottom = pad_lon - pad_top
        # using reflect to maintain smoothness at boundary
        x = nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

        # unet 
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)

        # concept predictions
        concepts = self.concept_head(x)  

        # final output using only concept maps
        out = self.output_head(concepts)

        # crop to original dims
        print('before crop concepts: ', concepts.shape)
        print('before crop output: ', out.shape)

        concepts = concepts[:, :, :lat, :lon]
        out = out[:, :, :lat, :lon]
        return out, concepts


if __name__ == "__main__":
    # these could be modified in the config
    in_channels = 50
    concept_channels = 9
    out_channels = 1
    channels_list = [64, 128, 256, 512]
    batch_size = 2
    model = UNetCBM(in_channels=in_channels, 
                    concept_channels=concept_channels, 
                    out_channels=out_channels, 
                    channels_list=channels_list)
    # 2D ocean field 
    x = torch.randn(batch_size, 50, 1021, 1442)
    y, c = model(x)
    print('input: ,', x.shape)
    print('concepts: ', c.shape)
    print('output: ', y.shape)
    summary(model, input_size=(batch_size, 50, 1021, 1442))
