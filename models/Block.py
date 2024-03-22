from torch import nn
import torch
from torch import torchvision
class Block(nn.Module):
    def __init__(self, out_channels, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        self.relu = nn.ReLU(),
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, channels = [64, 128, 256, 512, 1024]):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(channels[i], channels[i-1]) for i in range(1, len(channels))])
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.max_pool(x)
        return features
    
class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
