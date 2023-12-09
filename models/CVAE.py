import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, kernel_size,  init_channels, image_channels, latent_dim):
        super(ConvVAE, self).__init__()
        
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc5 = nn.Conv2d(
            in_channels=init_channels*8, out_channels=64, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        # fully connected layers for learning representations
        self.b1 = nn.BatchNorm2d(init_channels)
        self.b2 = nn.BatchNorm2d(init_channels*2)
        self.b3 = nn.BatchNorm2d(init_channels*4)
        self.b4 = nn.BatchNorm2d(init_channels*8)
        self.b5 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(256, 128)
        self.bc1 = nn.BatchNorm1d(128)
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.bc2 = nn.BatchNorm1d(latent_dim)
        
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.bc3 = nn.BatchNorm1d(latent_dim)
        
        self.fc2 = nn.Linear(latent_dim, 256)
        self.bc4 = nn.BatchNorm1d(256)
        
        # decoder 
        
        self.d4 = nn.BatchNorm2d(init_channels)
        self.d3 = nn.BatchNorm2d(init_channels*2)
        self.d2 = nn.BatchNorm2d(init_channels*4)
        self.d1 = nn.BatchNorm2d(init_channels*8)
        self.d5 = nn.BatchNorm2d(init_channels//2)
        
        self.dec1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_channels, out_channels= init_channels//2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec6 = nn.ConvTranspose2d(
            in_channels=init_channels//2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )


        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
    
    def encoder(self, x):
        x = F.relu(self.b1(self.enc1(x)))
        x = F.relu(self.b2(self.enc2(x)))
        x = F.relu(self.b3(self.enc3(x)))
        x = F.relu(self.b4(self.enc4(x)))
        x = F.relu(self.b5(self.enc5(x)))
        #print(x.shape,1)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 2).reshape(batch, -1)
        hidden = self.fc1(x)
        hidden = F.relu(self.bc1(hidden))
        # get `mu` and `log_var`
        mu = F.relu(self.bc2(self.fc_mu(hidden)))
        log_var = F.relu(self.bc3(self.fc_log_var(hidden)))
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = F.relu(self.bc4(self.fc2(z)))
        z = z.view(-1, 256, 1, 1)
        return z, mu, log_var
    
    def decoder(self, x, mu, log_var):
        x = F.relu(self.d1(self.dec1(x)))
        x = F.relu(self.d2(self.dec2(x)))
        x = F.relu(self.d3(self.dec3(x)))
        x = F.relu(self.d4(self.dec4(x)))
        x = F.relu(self.d5(self.dec5(x)))
        reconstruction = F.sigmoid(self.dec6(x))
        return reconstruction, mu, log_var
 
    def forward(self, x):
        # encoding
        x = F.relu(self.b1(self.enc1(x)))
        x = F.relu(self.b2(self.enc2(x)))
        x = F.relu(self.b3(self.enc3(x)))
        x = F.relu(self.b4(self.enc4(x)))
        x = F.relu(self.b5(self.enc5(x)))
        #print(x.shape,1)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 2).reshape(batch, -1)
        hidden = self.fc1(x)
        hidden = F.relu(self.bc1(hidden))
        # get `mu` and `log_var`
        mu = F.relu(self.bc2(self.fc_mu(hidden)))
        log_var = F.relu(self.bc3(self.fc_log_var(hidden)))
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = F.relu(self.bc4(self.fc2(z)))
        z = z.view(-1, 256, 1, 1)
 
        # decoding
        x = F.relu(self.d1(self.dec1(z)))
        x = F.relu(self.d2(self.dec2(x)))
        x = F.relu(self.d3(self.dec3(x)))
        x = F.relu(self.d4(self.dec4(x)))
        x = F.relu(self.d5(self.dec5(x)))
        reconstruction = F.sigmoid(self.dec6(x))
        return reconstruction, mu, log_var