from Block import Block, Encoder, Decoder
from torch import nn

class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64)):
        super(UNet, self).__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head    = nn.Conv2d(dec_chs[-1], 1, kernel_size=1)

    def forward(self, x):
        enc_features = self.encoder(x)
        out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        out = self.head(out)
        return out