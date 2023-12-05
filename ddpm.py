"""
    Diffusion model and image generation
    
    Ported directly from Dataflowr with slight modifications to allow other noise distributions to be experimented with
    https://dataflowr.github.io/website/modules/18a-diffusion/
"""

import torch
from torch import nn
from noise_utils import generate_noise
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPM(nn.Module):
    """
    DDPM model consisting of the forward and reverse process

    :param network is the neural network being used, i.e. UNet
    :param num_timesteps is the number of timesteps in the diffusion process, default is 1000 timesteps
    :param beta_start the starting value of the linear noise scheduler
    :param beta_end the ending value of the linear noise scheduler
    :param device the computation device
    """
    def __init__(self, network, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=device) -> None:
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
        """
        This is the forward process of adding noise
        
        @param x_start the input
        @param x_noise the noise to be added to the input
        @param timesteps timestep at which the noise is added
        @return input after adding noise
        """
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        # the new shape of s1 and s2 will be (timesteps, 1, 1, 1)
        s1 = s1.reshape(-1, 1, 1, 1)
        s2 = s2.reshape(-1, 1, 1, 1)
        return x_start*s1 + x_noise*s2

    def reverse(self, x, t):
        # The network return the estimation of the noise we added
        return self.network(x, t)
    
    def step(self, model_output, timestep, sample, noise_type, params):
        """
        This is one step of sampling as we loop through t = T, ..., 1
        
        @param model_output is the predicted epsilon, noise added to the current sample, which is x_{t}
        @param timestep is the current timestep we are at
        @param sample is the current sample, x_{t}
        @return a prediction of the previous sample, x_{t-1}
        """
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
            # filled with random numbers from a specific distribution
            z = generate_noise(noise_type, sample.shape, params).to(self.device)
            variance = (self.betas[t] ** 0.5)*z
            
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

def generate_image(ddpm, sample_size, channel, size, device, noise_type="Gaussian", params=None):
    """
    Generate image from random noise given noise type. 
    
    :param ddpm the trained ddpm model
    :param sample_size number of samples to be generated
    :param channel the number of channels of the input
    :param size the size of the input
    :param device the computation device
    :param noise_type options available are Gaussian, Laplace, S&P (Salt and Pepper). Default is Gaussian noise
    :param params the parameters of the noise distribution
    :return images generated from random noise
    """
    frames = []
    frames_mid = []
    ddpm.eval()
    batch_shape = torch.Size([sample_size, channel, size, size])
    with torch.no_grad():
        timesteps = list(range(ddpm.num_timesteps))[::-1]
        sample = generate_noise(noise_type, batch_shape, params).to(device)
    
        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
            residual = ddpm.reverse(sample, time_tensor)
            sample = ddpm.step(residual, time_tensor[0], sample, noise_type, params)

            if t==len(timesteps)/2:
                for i in range(sample_size):
                    frames_mid.append(sample[i].detach().cpu())

        for i in range(sample_size):
            frames.append(sample[i].detach().cpu())
    return frames, frames_mid