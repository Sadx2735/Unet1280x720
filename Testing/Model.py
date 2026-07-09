import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. Model Architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet1280x720(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet1280x720, self).__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bn = self.bottleneck(self.pool4(d4))
        u1 = self.up1(bn)
        u1 = self.up_conv1(torch.cat([u1, d4], dim=1))
        u2 = self.up2(u1)
        u2 = self.up_conv2(torch.cat([u2, d3], dim=1))
        u3 = self.up3(u2)
        u3 = self.up_conv3(torch.cat([u3, d2], dim=1))
        u4 = self.up4(u3)
        u4 = self.up_conv4(torch.cat([u4, d1], dim=1))
        return self.out_conv(u4)

# 2. Combined Loss
class DiceBCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return BCE + dice

# 3. Dataset
class UNetDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir, self.masks_dir, self.transform = images_dir, masks_dir, transform
        self.images = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))]
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dir, self.images[idx])).convert("RGB").resize((1280, 720))
        mask = Image.open(os.path.join(self.masks_dir, self.images[idx])).convert("L").resize((1280, 720))
        img = self.transform(img)
        mask = self.transform(mask)
        return img, (mask > 0).float()

# 4. Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet1280x720().to(device)