"""

Utilities for handling image data
Directly ported from Dataflowr with slight modification

https://github.com/dataflowr/notebooks

"""

import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

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
    
class RestrictedDataset(Dataset):
    """ 
    A classic pytorch dataset that only keeps specified labels in the dataset. 
    
    @param dataset - A pytorch dataset object
    @param labels - A list of either the raw labels (integers) or class label (strings) to select for data.
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        if not isinstance(labels, (list, tuple, set)):
            # Check whether this is a single label
            assert isinstance(labels, (str, int))
            
        labels_cleaned = set()
        # Convert all labels to target idx and a set
        for label in labels:
            assert isinstance(label, (str, int))
            
            # Convert into an integer
            if isinstance(label, str):
                assert label in dataset.class_to_idx, 'Class is not in labels'
                label = dataset.class_to_idx[label]
            
            labels_cleaned.add(label)
        
        
        # Select labels
        in_dataset = np.array([x in labels_cleaned for x in dataset.targets])
        self.mapping = torch.arange(len(dataset))[in_dataset]
        print(f'Dataset created with {self.__len__()} examples')
    
    def __len__(self):
        return len(self.mapping)
    
    def __getitem__(self, idx):
        return self.dataset[self.mapping[idx]]
    
    
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