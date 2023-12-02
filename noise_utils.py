"""
Utilities for generating noise distributions. We allow generation of Gaussian, Laplace and Salt & Pepper noise
"""

import torch

def generate_distribution(distribution_name, params=None):
    """
    Returns distribution given the name of distribution (either Gaussian or Laplace)
    
    :param distribution_name the name of distribution
    :param params a dictionary of the parameters of the distribution, which are loc and scale. Without specifying params, the default is scale of 1.0 and loc (or mean) of 0.0
    :returns a distribution
    """
    valid_distribution = {"Gaussian", "Laplace"}
    if distribution_name not in valid_distribution:
        raise Exception(f"'distribution' not of value {valid_distribution}")
    
    loc = 0.0
    scale = 1.0
        
    if params:
        try:
            loc = params['loc']
            scale = params['scale']
        except KeyError as error:
            raise Exception("'params' is not properly initialized. It should contain 'loc' and 'scale' of the distribution")
        except TypeError as error:
            raise Exception("'params' should be a dictionary containing keys 'loc' and 'scale'")
    
    if not isinstance(loc, float):
        raise Exception("'loc' must be float")
        
    if not isinstance(scale, float):
        raise Exception("'scale' must be float")
            
    if distribution_name == "Gaussian":
        # normal gaussian, a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 
        dist = torch.distributions.Normal(torch.tensor([loc]), torch.tensor([scale]))
        
    elif distribution_name == "Laplace":
        dist = torch.distributions.Laplace(torch.tensor([loc]), torch.tensor([scale]))
        
    return dist

def generate_noise(noise_distribution, batch_shape, device, params=None):
    """
    Calls `generate_distribution` to generate noise when noise distribution is Gaussian or Laplace. Otherwise, generates salt & pepper noise with probability 1% for salt and 1% for pepper by default
    
    :param noise_distribution String value stating the distribution of noise to be generated
    :param batch_shape the shape of noise to be expanded to
    :param device the computation device
    :param params a dictionary of the parameters of the distribution, which are loc and scale for Gaussian/Laplace noise, probabilities for Salt & Pepper noise
    :return
    """
    noise_distribution_allowed = {"Gaussian", "Laplace", "S&P"}
    if noise_distribution not in noise_distribution_allowed:
        raise Exception(f"'noise_distribution' not of value {noise_distribution_allowed}")
    
    if noise_distribution == "S&P":
        salt_proba = 0.01
        pepper_proba = 0.01
        
        if params:
            try:
                salt_proba = params['salt_proba']
                pepper_proba = params['papper_proba']
            except KeyError as error:
                raise Exception("'params' is not properly initialized. It should contain 'salt_proba' and 'papper_proba'")
            except TypeError as error:
                raise Exception("'params' should be a dictionary containing keys 'salt_proba' and 'papper_proba'")
            
        salt_mask = torch.randn(batch_shape).to(device) < salt_proba
        pepper_mask = (torch.randn(batch_shape).to(device) < pepper_proba) & ~salt_mask # so that we add pepper on spots where salt isn't added to
        noise = (salt_mask.float() + pepper_mask.float()*-1).to(device)
    
    else:
        noise = generate_distribution(noise_distribution, params).expand(batch_shape).sample().to(device)
        
    return noise