"""

Utilities for training specific datasets, i.e. CIFAR10 and MNIST

"""

import copy
from image_data_utils import make_dataset_by_class
from ddpm import DDPM
from train_utils import TrainingLoop

def train_cifar_by_class(cifar_dataset, class_name, device, base_network, batch_size=512, num_epochs=20, num_workers=4):
    valid_class = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}
    if class_name not in valid_class:
        raise Exception(f"'class_name' not of value {valid_class}")
        
    subset_dataset = make_dataset_by_class(cifar_dataset, class_name)
    subset_network = copy.deepcopy(base_network)
    subset_model = DDPM(subset_network, device=device)
    subset_model.train()
    trainer = TrainingLoop(diffusion_model=subset_model, 
                           network=subset_network, 
                           dataset=subset_dataset, 
                           batch_size=batch_size, 
                           num_epochs=num_epochs, 
                           num_workers=num_workers)
    trainer.run_loop()
    return subset_model, subset_network