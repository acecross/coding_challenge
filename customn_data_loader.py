import os
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms,io
import torch
import torch.nn.functional as F

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, image_folder, resize=(256,256)):
        self.data = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transforms.Resize(resize, antialias=True)

    def __getitem__(self, index):
        current = self.data.iloc[index]
        im = io.read_image(os.path.join(self.image_folder, current["filename"]))
        c,h,w = im.shape
        diff =abs(h-w)
        if h<w:
            im = F.pad(im, (0,0,diff//2,diff-diff//2,0,0))
        elif h>w:
            im = F.pad(im, (0,0,0,0,diff//2,diff-diff//2))
        im = self.transform(im).float()
        return  im, torch.tensor(self.data.iloc[index,1:3].astype("float32").values).float()

    def __len__(self):
        return len(self.data)