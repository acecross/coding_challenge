import hydra
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import io
from torchvision.transforms.functional import resize

from network import Net


@hydra.main(config_name="infer.yaml", config_path="cfg")
def infer(cfg):
    #get some params from cfg
    #hydra changes cwd and creates and output folder. Nice for tracing results (can be overwritten)
    cwd = get_original_cwd()
    data_folder = cfg.io.data_folder
    image_folder = cfg.io.image_folder
    csv_file = cfg.io.csv_file
    device = cfg.params.device
    torch.manual_seed(cfg.params.seed)

    net = Net(size=(cfg.params.image_resize,cfg.params.image_resize))
    #todo: push this into config
    #use backdoor right to check model
    image = io.read_image(cwd+"/data/imgs/0a794c3a-d161-4cc8-98e5-b1bb3d481460.jpg").float()
    c, h, w = image.shape
    diff = abs(h - w)
    if h < w:
        image = F.pad(image, (0, 0, diff // 2, diff - diff // 2, 0, 0))
    image = resize(image, (cfg.params.image_resize, cfg.params.image_resize))
    #todo: dataloader for eval
    net.eval()
    # no grad needed in evaluation
    checkpoint = torch.load(os.path.join(cwd,cfg.io.training_folder,cfg.io.training_name))
    net.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        #create batch dim for one image
        print(net(image[None,:]))
        plt.imshow(np.moveaxis(np.array(image),0,2))
        plt.show()

if __name__ == '__main__':
    infer()