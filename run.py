from image_data_utils import load_data
from full_unet import FullUNet
from ddpm import DDPM
from train_utils import TrainingLoop

import sys
import torch
import json

if __name__  == '__main__':
    args = sys.argv[1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on " + str(device))
    with open("./config/etl.json", "r") as json_file:
        config = json.loads(json_file.read())
    data = torch.load(config['fp'], map_location=device)
    print('Data Loaded')


    if 'build' in args:
        network = FullUNet().to(device)
        model = DDPM(network, device = device)
        trainer = TrainingLoop(diffusion_model = model,
                            network = network,
                            dataset = data,
                            batch_size = config['batch_size'],
                            num_epochs = config['epochs'],
                            num_workers = config['workers'])
        trainer.run_loop()



