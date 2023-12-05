"""

Utilities for training the DDPM model and UNet

Some parts of the TrainingLoop were ported from Dataflowr with modifications:
https://dataflowr.github.io/website/modules/18a-diffusion/

"""
import torch
from torch import nn
from tqdm.notebook import tqdm
from noise_utils import generate_noise

class TrainingLoop:
    """
    This class sets up the training process and trains the model
    
    :param diffusion_model the DDPM model
    :param network the neural network used - we are using UNet for images
    :param dataset the dataset to be train on
    :param batch_size the size of each batch during the training process
    :param num_epochs the number of iterations
    :param learning_rate the learning rate of the optimizer, default is set to 0.001
    :param num_workers the number of workers when loading the data, default is set to 10
    :param num_timesteps the number of timesteps in the diffusion process, default is set to 1000
    :param noise_type the type of distribution where the noise is sampled from, default is set to Gaussian noise
    :param loss_f the type of loss function used when running the optimizer, default is set to mean-squared-error (MSE) loss
    :param shuffle_data determines whether to shuffle the data when loading data, default is set to True
    """
    def __init__(self, 
                diffusion_model,
                network,
                dataset,
                batch_size, 
                num_epochs, 
                learning_rate = 1e-3,
                num_workers=10, 
                num_timesteps=1000,  
                noise_type="Gaussian", 
                loss_f="MSE", 
                shuffle_data=True):
        
        self.diffusion_model = diffusion_model
        self.network = network
        self.network_params = self.network.parameters()
        self.epochs = num_epochs
        self.timesteps = num_timesteps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_type = noise_type
        self.loss_f = loss_f
        self.global_step = 0
        
        self.optimizer = torch.optim.Adam(self.network_params, lr=learning_rate)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=num_workers)
    
    def run_loop(self):
        for epoch in range(self.epochs):
            self.diffusion_model.train() 
            progress_bar = tqdm(total=len(self.dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(self.dataloader):
                # put batch to device (gpu or cpu) to leverage computational resources
                batch = batch[0].to(self.device)

                # create noise
                actual_noise = generate_noise(self.noise_type, batch.shape, self.device)

                # Generates random timesteps for each sample in the batch. 
                # These timesteps determine at which diffusion step the noise is added. 
                # The num_timesteps parameter specifies the total number of diffusion steps.
                timesteps = torch.randint(0, self.timesteps, (batch.shape[0], )).long().to(self.device)

                # returns x_{t+1}, a noisy image
                noisy = self.diffusion_model.add_noise(batch, actual_noise, timesteps)

                # gives the noisy image and returns a prediction of the noise added
                noise_pred = self.diffusion_model.reverse(noisy, timesteps)

                loss = calc_loss(self.loss_f, noise_pred, actual_noise)

                # reset the gradient to zero before computing the new gradient in the backward pass
                self.optimizer.zero_grad()

                # Computes the gradients of the loss with respect to the model parameters. 
                # These gradients are used to update the model weights during optimization.
                loss.backward()

                # updates the parameters
                self.optimizer.step()

                progress_bar.update(1)
                logs = {"loss": loss.detach(), "step": self.global_step}
                progress_bar.set_postfix(**logs)
                self.global_step += 1

            progress_bar.close()
            
        self.global_step = 0
        
def calc_loss(loss_f, pred, actual):
    """
    Calculates the loss given the type of loss function to be used, the predicted value and the actual value
    """
    if loss_f == "LaplaceNLL":
        criterion = LaplaceNLL()
    elif loss_f == "L1":
        criterion = nn.L1Loss()
    elif loss_f == "MSE":
        criterion = nn.MSELoss()
    elif loss_f == "NLL":
        criterion = nn.NLLLoss()
    elif loss_f == "Sigmoid":
        criterion = nn.BCEWithLogitsLoss()
    
    loss = criterion(pred, actual)
        
    return loss

class LaplaceNLL(nn.Module):
    """
    Calculates Laplace Negative Log Likelihood given a predicted value, x and the actual value, target
    """
    def __init__(self, scale = 1.0):
        super(LaplaceNLL, self).__init__()
        self.scale = scale
    
    def forward(self, x, target):
        log = torch.log(torch.tensor([2*self.scale])).to(device)
        nll_loss = torch.sum(log + torch.abs(x-target)/self.scale, dim=-1)
        nll_loss = torch.mean(nll_loss)
        return nll_loss