import torch
from torch import nn

import torch.nn.functional as F


class Net(nn.Module):
    """
    Very simple and lightweight CNN
    """
    def __init__(self, size:tuple=(256,256)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding="same")
        self.conv3 = nn.Conv2d(16, 32, 5, padding="same")
        self.conv4 = nn.Conv2d(32, 64, 5, padding="same")
        #64 is the last hidden dimension
        size_after_pooling = (size[0]//2**4)*(size[1]//2**4)*64
        self.fc1 = nn.Linear(size_after_pooling, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def normalize(self, x:torch.tensor)->torch.tensor:
        """
        normalize inputs by mean->0 and std->1
        :param x:
        :return:
        """
        x -= x.mean()
        x /= x.std()
        return x


    def forward(self, x:torch.tensor)->torch.tensor:
        x = self.normalize(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

