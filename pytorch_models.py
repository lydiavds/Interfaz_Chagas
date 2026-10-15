"""
Arquitectura de modelos PyTorch para FiCRoN

IMPORTANTE: Estas arquitecturas deben coincidir EXACTAMENTE con las que se usaron
para entrenar los modelos .pth. Si la arquitectura no coincide, el modelo no cargará.
"""

import torch
import torch.nn as nn
from torchvision import models

# Arquitectura U-Net con ResNet18 como encoder
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder (ResNet18 preentrenado)
        resnet = models.resnet18(pretrained=False)  # Cambiado a False para evitar descarga
        self.encoder1 = nn.Sequential(*list(resnet.children())[:4])  # Hasta MaxPool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # Decoder
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)    # 64 canales, 256x256
        e2 = self.encoder2(e1)   # 64 canales, 256x256
        e3 = self.encoder3(e2)   # 128 canales, 128x128
        e4 = self.encoder4(e3)   # 256 canales, 64x64
        e5 = self.encoder5(e4)   # 512 canales, 32x32

        # Decoder con skip connections
        d5 = self.upconv5(e5)    # 256 canales, 64x64
        d5 = torch.cat([d5, e4], dim=1)  # 512 canales, 64x64
        d5 = self.decoder5(d5)   # 256 canales, 64x64

        d4 = self.upconv4(d5)    # 128 canales, 128x128
        d4 = torch.cat([d4, e3], dim=1)  # 256 canales, 128x128
        d4 = self.decoder4(d4)   # 128 canales, 128x128

        d3 = self.upconv3(d4)    # 64 canales, 256x256
        d3 = torch.cat([d3, e2], dim=1)  # 128 canales, 256x256
        d3 = self.decoder3(d3)   # 64 canales, 256x256

        d2 = self.upconv2(d3)    # 64 canales, 512x512
        d2 = self.decoder2(d2)   # 64 canales, 512x512 (sin concatenación con e1)

        d1 = self.upconv1(d2)    # 32 canales, 1024x1024
        out = self.decoder1(d1)  # 1 canal, 1024x1024

        return out


#Se usa la misma arquitectura para los modelos de las fases 1 y 2.

#También se usa misma arquitectura para el modelo nucleos2, ya que al final ha sido un fine-tune entrenado sobre el de Miguel


#Modelo para parásitos (UNet modificada)
import torch
import torch.nn as nn
import torchvision.models as models

class UNet_Parasitos(nn.Module):
    def __init__(self, num_groups=8):
        super(UNet_Parasitos, self).__init__()

        # Encoder (ResNet18 preentrenado)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder1 = nn.Sequential(*list(resnet.children())[:4])  # Hasta MaxPool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # Decoder — cada bloque: ConvTranspose -> concat -> Conv -> GroupNorm -> ReLU
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 256),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # Salida con ReLU para asegurar densidades >= 0, pero sin normalizacion porque si no se ponen mal las escalas
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            #nn.ReLU() #Parece que va mal con esto
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)    # 64 canales, 256x256
        e2 = self.encoder2(e1)   # 64 canales, 256x256
        e3 = self.encoder3(e2)   # 128 canales, 128x128
        e4 = self.encoder4(e3)   # 256 canales, 64x64
        e5 = self.encoder5(e4)   # 512 canales, 32x32

        # Decoder con skip connections
        d5 = self.upconv5(e5)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.decoder5(d5)

        d4 = self.upconv4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = self.decoder2(d2)  # sin concatenación con e1, como en tu original

        d1 = self.upconv1(d2)
        out = self.decoder1(d1)  # sin GN/ReLU: salida cruda del mapa de densidad

        return out

