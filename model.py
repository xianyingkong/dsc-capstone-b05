import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm.notebook import tqdm
from torch.distributions.laplace import Laplace
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPM(nn.Module):
    def __init__(self, network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device) -> None:
        super(DDPM, self).__init__()
        self.num_timesteps = num_timesteps
        # creating a 1-d tensor of size num_timesteps
        # setting variances betas
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        # let alpha = 1 - beta at each timestep
        self.alphas = 1.0 - self.betas
        # this is the cumulative product of all alphas - this will be used on image at t when getting image at t+1
        # the vector alphas_cumprod will have size num_timesteps, still in 1-d
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        # sqrt_alphas_cumprod will have size num_timesteps
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5 # used in add_noise
        # sqrt_one_minus_alphas_cumprod will have size num_timesteps
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5 # used in add_noise and step

    def add_noise(self, x_start, x_noise, timesteps):
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        # according to the solution, this is for broadcasting? why?
        # the new shape of s1 and s2 will be (timesteps, 1, 1, 1)
        s1 = s1.reshape(-1, 1, 1, 1)
        s2 = s2.reshape(-1, 1, 1, 1)
        return x_start*s1 + x_noise*s2

    def reverse(self, x, t):
        # The network return the estimation of the noise we added
        return self.network(x, t)
    
    def step(self, model_output, timestep, sample):
        """
        This function is one step of sampling (algorithm 2 in DDPM paper)
        As we loop through t = T, ..., 1, we will call this method
        
        @param model_output is the predicted epsilon, noise added to the current sample, which is x_{t}
        @param timestep is the current timestep we are at
        @param sample is the current sample, x_{t}
        @return a prediction of the previous sample, x_{t-1}
        """
        # one step of sampling
        # this is algorithm 2 - sampling, for each t, do..
        # timestep (1)
        t = timestep
        # coef_epsilon is a 1-d tensor with size (timestep)
        coef_epsilon = (1-self.alphas[t])/self.sqrt_one_minus_alphas_cumprod[t]
        coef_epsilon = coef_epsilon.reshape(-1, 1, 1, 1)
        coef_first = 1/(self.alphas[t]**0.5)
        coef_first = coef_first.reshape(-1, 1, 1, 1)
        # x_{t-1} without the variance
        pred_prev_sample = coef_first*(sample - coef_epsilon*model_output)
        
        variance = 0
        if t > 0:
            # Returns a tensor with the same size as input that is 
            # filled with random numbers from a normal distribution 
            # with mean 0 and variance 1
            z = torch.randn_like(sample).to(self.device)
            variance = (self.betas[t] ** 0.5)*z
            
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample


class MyConv(nn.Module):
    """
    This class represents a convolutional block with optional layer normalization, 
    a 2D convolutional layer, an activation function, and optional normalization.

    @param in_c number of input channels
    @param out_c number of output channels
    @param kernel_size, stride and padding are parameters of the convolution layer
    @param activation activation function is SiLU if not specified
    @param normalize boolean value set to True as default

    """
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyConv, self).__init__()
        self.ln = nn.LayerNorm(shape) # normalizes the layer
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        return out


class MyTinyUNet(nn.Module):
    # Here is a network with 3 down and 3 up with the tiny block
    def __init__(self, in_c=1, out_c=1, size=32, n_steps=1000, time_emb_dim=100):
        super(MyTinyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = self._sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False) # setting this means the the temporal embeddings are not trainable

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = self.MyTinyBlock(size, in_c, 10)
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)
        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = self.MyTinyBlock(size//2, 10, 20)
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)
        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = self.MyTinyBlock(size//4, 20, 40)
        self.down3 = nn.Conv2d(40, 40, 4, 2, 1)

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyConv((40, size//8, size//8), 40, 20),
            MyConv((20, size//8, size//8), 20, 20),
            MyConv((20, size//8, size//8), 20, 40)
        )

        # Second half
        self.up1 = nn.ConvTranspose2d(40, 40, 4, 2, 1)
        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = self.MyTinyUp(size//4, 80)
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = self.MyTinyUp(size//2, 40)
        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = self.MyTinyBlock(size, 20, 10)
        self.conv_out = nn.Conv2d(10, out_c, 3, 1, 1)

    def forward(self, x, t): # x is (bs, in_c, size, size) t is (bs)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (bs, 10, size/2, size/2)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (bs, 20, size/4, size/4)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (bs, 40, size/8, size/8)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (bs, 40, size/8, size/8)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (bs, 80, size/8, size/8)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (bs, 20, size/8, size/8)
        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (bs, 40, size/4, size/4)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (bs, 10, size/2, size/2)
        out = torch.cat((out1, self.up3(out5)), dim=1)  # (bs, 20, size, size)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (bs, 10, size, size)
        out = self.conv_out(out) # (bs, out_c, size, size)
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
      
    def MyTinyBlock(self, size, in_c, out_c):
        """
        This function creates a sequential block consisting of three MyConv blocks. 
        It is essentially a mini neural network block comprising three convolutional layers with normalization and activation.
        """  
        return nn.Sequential(MyConv((in_c, size, size), in_c, out_c), 
                             MyConv((out_c, size, size), out_c, out_c), 
                             MyConv((out_c, size, size), out_c, out_c))


    def MyTinyUp(self, size, in_c):
        return nn.Sequential(MyConv((in_c, size, size), in_c, in_c//2), 
                             MyConv((in_c//2, size, size), in_c//2, in_c//4), 
                             MyConv((in_c//4, size, size), in_c//4, in_c//4))