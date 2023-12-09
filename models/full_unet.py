"""

Setting up the UNet

Expansion and modification of tiny UNet from Dataflowr: https://dataflowr.github.io/website/modules/18a-diffusion/

"""

import torch
from torch import nn

class SimpleConv(nn.Module):
    """
    This class represents a convolutional block with a 2D convolutional layer, a batch normalization function, and an activation function.

    :param in_c number of input channels
    :param out_c number of output channels
    :param kernel_size, stride and padding are parameters of the convolution layer
    :param activation activation function is ReLU if not specified
    :param normalize boolean value set to True as default

    """
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.activation = nn.ReLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out) if self.normalize else out
        out = self.activation(out)
        return out
    
class FullUNet(nn.Module):
    """
    This class represents the full UNet with time embeddings
    
    :param in_c the number of input channels
    :param out_c the number of output channels
    :param n_steps the number of time embeddings
    :param time_emb_dim the size of each embedding vector
    
    """
    def __init__(self, in_c=1, out_c=1, n_steps=1000, time_emb_dim=100):
        super().__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = self._sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False) # setting this means the the temporal embeddings are not trainable

        # Downsampling
        self.te1 = self._make_te(time_emb_dim, in_c)
        self.b1 = self.ConvBlock(in_c, 64)
        self.pool = nn.MaxPool2d(2)
        
        self.te2 = self._make_te(time_emb_dim, 64)
        self.b2 = self.ConvBlock(64, 128)
        
        self.te3 = self._make_te(time_emb_dim, 128)
        self.b3 = self.ConvBlock(128, 256)
        
        self.te4 = self._make_te(time_emb_dim, 256)
        self.b4 = self.ConvBlock(256, 512)

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 512)
        self.b_mid = self.ConvBlock(512, 1024)

        # Second half
        self.up1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 1024)
        self.b5 = self.ConvBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.te6 = self._make_te(time_emb_dim, 512)
        self.b6 = self.ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.te7 = self._make_te(time_emb_dim, 256)
        self.b7 = self.ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.te8 = self._make_te(time_emb_dim, 128)
        self.b8 = self.ConvBlock(128, 64)
        self.conv_out = nn.Conv2d(64, out_c, 3, 1, 1)

    def forward(self, x, t): 
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1)) 
        out2 = self.b2(self.pool(out1) + self.te2(t).reshape(n, -1, 1, 1)) 
        out3 = self.b3(self.pool(out2) + self.te3(t).reshape(n, -1, 1, 1)) 
        out4 = self.b4(self.pool(out3) + self.te4(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.pool(out4) + self.te_mid(t).reshape(n, -1, 1, 1))

        out5 = torch.cat([out4, self.up1(out_mid)], axis=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))
        out6 = torch.cat((out3, self.up2(out5)), dim=1)
        out6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1, 1))  
        out7 = torch.cat((out2, self.up3(out6)), dim=1)  
        out7 = self.b7(out7 + self.te7(t).reshape(n, -1, 1, 1))
        out8 = torch.cat((out1, self.up4(out7)), dim=1)
        out8 = self.b8(out8 + self.te8(t).reshape(n, -1, 1, 1))
        out = self.conv_out(out8)
        return out

    # temporal embedding
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))
    
    def _sinusoidal_embedding(self, n, d):
        # Returns the standard positional embedding
        embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
        sin_mask = torch.arange(0, n, 2)

        embedding[sin_mask] = torch.sin(embedding[sin_mask])
        embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

        return embedding
      
    def ConvBlock(self, in_c, out_c):
        """
        This function creates a sequential block consisting of two SimpleConv blocks.
        """  
        return nn.Sequential(SimpleConv(in_c, out_c), 
                             SimpleConv(out_c, out_c))
