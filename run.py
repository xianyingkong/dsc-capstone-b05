import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from tqdm.notebook import tqdm
from image_helpers import show_images, show_images_rescale
from model_helper import DDPM, generate_image#, MyTinyUNet, training_loop, generate_image
from unet import UNet
import argparse

arg_in = argparse.ArgumentParser(prog='Manage a Diffusion Model')

parser.add_argument