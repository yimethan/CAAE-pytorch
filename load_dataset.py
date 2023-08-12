import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os

from config import Config

class GetDataset(Dataset):

    def __init__(self):

        super(GetDataset, self).__init__()

        self.data_root = Config.data_root

        self.images = []
        self.labels = []

        self.load_from_dir()

    def __getitem__(self, item):

        return {'input', images[item], 'label', labels[item]}

    def construct_data_loaders(self):
        transform = transforms.Compose([transforms.ToTensor()])  # Add any necessary transformations

        train_dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, 'train_label'), transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=4)

        if self.unknown_attack is not None:
            val_unknown_dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, self.unknown_attack),
                                                       transform=transform)
            val_unknown_loader = torch.utils.data.DataLoader(dataset=val_unknown_dataset, batch_size=self.batch_size,
                                                             shuffle=False, num_workers=4)

        return train_loader, val_unknown_loader