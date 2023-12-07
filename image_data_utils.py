"""

Utilities for handling image data
Directly ported from Dataflowr with slight modification

https://github.com/dataflowr/notebooks

"""

import torch
import torchvision
import matplotlib.pyplot as plt

def load_data(data_name, root_dir, train=True, download=True):
    valid_data = {"MNIST", "CIFAR10"}
    if data_name not in valid_data:
        raise Exception(f"'data_name' not of value {valid_data}")
    
    dataset = None
    
    if data_name == "MNIST":
        transform01 = torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor()
                ])
        dataset = torchvision.datasets.MNIST(root=root_dir, train=train, transform=transform01, download=download)
    else:
        transforms01 = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset = torchvision.datasets.CIFAR10(root=root_dir, train=train, transform=transforms01, download=download)
    return dataset

def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""
    images = [im.permute(1,2,0).numpy() for im in images]

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx], cmap="gray")
                plt.axis('off')
                idx += 1
    fig.suptitle(title, fontsize=30)
    
    # Showing the figure
    plt.show()

def show_image_dataset(dataset, class_name = None, num_images = 100):
    dataloader = torch.utils.data.DataLoader(dataset=dataset)
    if class_name is not None:
        dataloader = make_dataloader_by_class(dataset, class_name, batch_size=num_images)
        
    for b in dataloader:
        batch = b[0]
        break
    
    try:
        bn = [b for b in batch[:num_images]] 
        show_images(bn)
    except IndexError as error:
        raise IndexError("'num_images' exceed number of image data, re-adjust 'num_images'")
    
def make_dataset_by_class(dataset, class_name):
    """Create Subset object given a class name of a multiclass dataset"""
    s_indices = []
    s_idx = dataset.class_to_idx[class_name]
    for i in range(len(dataset)):
        current_class = dataset[i][1]
        if current_class == s_idx:
            s_indices.append(i)
    return torch.utils.data.Subset(dataset, s_indices) 
    
def make_dataloader_by_class(dataset, class_name, batch_size = 1, shuffle = True):
    """Create Dataloader object given a class name of a multiclass dataset"""
    subset = make_dataset_by_class(dataset, class_name)
    return torch.utils.data.DataLoader(dataset=subset, batch_size=batch_size, shuffle=shuffle)