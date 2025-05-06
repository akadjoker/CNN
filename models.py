import torch
import torch.nn as nn

#"small"
# class LaneNet(nn.Module):
#     def __init__(self):
#         super(LaneNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),  # supondo imagens RGB
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
#             nn.ReLU(),

#             nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
#             nn.ReLU(),

#             nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
#             nn.Sigmoid()  # valores entre 0 e 1 para segmentação
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

#medio
# class LaneNet(nn.Module):
#     def __init__(self):
#         super(LaneNet, self).__init__()

#         # --- Encoder ---
#         self.enc1 = self.conv_block(3, 32)
#         self.pool1 = nn.MaxPool2d(2)  # 80x160 -> 40x80

#         self.enc2 = self.conv_block(32, 64)
#         self.pool2 = nn.MaxPool2d(2)  # 40x80 -> 20x40

#         self.enc3 = self.conv_block(64, 128)
#         self.pool3 = nn.MaxPool2d(2)  # 20x40 -> 10x20

#         self.enc4 = self.conv_block(128, 256)
#         # pool3 não vai mais fundo, mantemos 10x20

#         # --- Decoder ---
#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 10x20 -> 20x40
#         self.dec3 = self.conv_block(256, 128)  # concat com enc3

#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 20x40 -> 40x80
#         self.dec2 = self.conv_block(128, 64)   # concat com enc2

#         self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # 40x80 -> 80x160
#         self.dec1 = self.conv_block(64, 32)    # concat com enc1

#         self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
#         self.activation = nn.Sigmoid()

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # Encoder
#         enc1 = self.enc1(x)
#         x = self.pool1(enc1)

#         enc2 = self.enc2(x)
#         x = self.pool2(enc2)

#         enc3 = self.enc3(x)
#         x = self.pool3(enc3)

#         x = self.enc4(x)

#         # Decoder
#         x = self.up3(x)
#         x = torch.cat([x, enc3], dim=1)
#         x = self.dec3(x)

#         x = self.up2(x)
#         x = torch.cat([x, enc2], dim=1)
#         x = self.dec2(x)

#         x = self.up1(x)
#         x = torch.cat([x, enc1], dim=1)
#         x = self.dec1(x)

#         x = self.out_conv(x)
#         return self.activation(x)

 

class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()

        # --- Encoder ---
        self.enc1 = self.conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)  # 80x160 -> 40x80

        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)  # 40x80 -> 20x40

        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)  # 20x40 -> 10x20

        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)  # 10x20 -> 5x10

        self.enc5 = self.conv_block(512, 1024)  # fundo

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 5x10 -> 10x20
        self.dec4 = self.conv_block(1024, 512)  # concat com enc4

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # 10x20 -> 20x40
        self.dec3 = self.conv_block(512, 256)  # concat com enc3

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # 20x40 -> 40x80
        self.dec2 = self.conv_block(256, 128)  # concat com enc2

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # 40x80 -> 80x160
        self.dec1 = self.conv_block(128, 64)   # concat com enc1

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.activation = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),   # Melhorar estabilidade
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        enc4 = self.enc4(x)
        x = self.pool4(enc4)

        x = self.enc5(x)

        # Decoder
        x = self.up4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        x = self.out_conv(x)
        return self.activation(x)
