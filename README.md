## Exploring Noise Distributions in Diffusion Models: A Comparative Analysis and Application
### FA23 DSC Capstone Project Description
This project aims to explore the effect of non-Gaussian distributions, combined with other loss functions on the outcome of **Denoising Diffusion Probabilistic Model (DDPM)**. More specifically, we implement noise sampled from Laplace distribution and Salt & Pepper (S&P) noise, together with L1 loss, Laplace negative log likelihood, and Sigmoid loss. The datasets used in this project are MNIST and CIFAR10 from Pytorch.

If you are interested in training a mini DDPM model, the data folder contains MNIST test data - 100 MNIST images sampled from MNIST training data. Otherwise, the full training process is expected to take around 45-75 minutes on either the full MNIST training data or CIFAR10.

### Software & Spec
All dependencies is listed in requirements.txt file. Running pip install -r requirements.txt will create an environment that will allow you to run ddpm_notebook.ipynb. For all our experiments, we used 1 GPU, 4 CPUs and 80GB RAM.

### Reference
- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *arXiv preprint arXiv:2006.11239*, 2020
- Eliya Nachmani, Robin San Roman, and Lior Wolf. Non Gaussian Denoising Diffusion Models. *arXiv preprint arXiv:2106.07582*, 2021
- Tien Chen. On the Importance of Noise Scheduling for Diffusion Models. *arXiv:2301.10972* [cs.CV]
