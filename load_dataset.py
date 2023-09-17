import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

from config import Config


class GetDataset(Dataset):

    def __init__(self, labeled=True):

        super(GetDataset, self).__init__()

        self.data_root = Config.data_root

        self.labeled = labeled

        self.images = []
        self.labels = []

        self.load_from_dir()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        img = self.images[item]
        img = Image.open(img).convert('L')
        img = transforms.ToTensor()(img)

        lb = self.labels[item]
        lb = torch.tensor(lb)

        return {'input': img, 'label': lb}

    def load_from_dir(self):

        attacks = os.listdir(self.data_root)

        for attack in attacks:

            path_to_attack = self.data_root + attack
            filenames = os.listdir(path_to_attack)

            for file in filenames:

                path_to_filename = os.path.join(path_to_attack, file)
                self.images.append(path_to_filename)

                label = file.split('_')[0]
                label = int(label)
                self.labels.append(label)