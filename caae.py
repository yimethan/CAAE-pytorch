import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fullyConnected(input, output_size):
    batch_size = input.size(0)  # Get the batch size
    try:
        input_features = input.size(1) * input.size(2) * input.size(3)  # Get the number of input features
    except IndexError:
        input_features = input.size(1)
    flattened_input = input.view(batch_size, -1)

    fc_layer = nn.Linear(input_features, output_size).to(device)
    fc_output = fc_layer(flattened_input)

    return fc_output


class Encoder(nn.Module):

    def __init__(self, keep_prob=Config.keep_prob):

        super(Encoder, self).__init__()
        self.n_labels = Config.n_labels
        self.z_dim = Config.z_dim

        self.keep_prob = keep_prob

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu0 = nn.ReLU()
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=self.keep_prob)

    def forward(self, x, supervised=False):
        x = F.pad(x, (1, 2, 1, 2))  # batch * 1 * 32 * 32

        x = self.conv0(x)  # batch * 32 * 32 * 32
        x = self.relu0(x)
        x = self.pool0(x)  # batch * 32 * 16 * 16

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.dropout(x)

        latent = fullyConnected(x, self.z_dim)
        cat_op = fullyConnected(x, self.n_labels)

        if supervised:
            softmax_label = cat_op
        else:
            softmax_label = F.softmax(cat_op, dim=1)

        return softmax_label, latent


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv0 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu0 = nn.ReLU()
        self.up0 = self.upsample
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.up1 = self.upsample
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.up2 = self.upsample
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.up3 = self.upsample

    def upsample(self, x, factor=2):
        x = F.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False)
        return x

    def forward(self, x):

        x = fullyConnected(x, 64 * 2 * 2)
        x = x.view(-1, 64, 2, 2)

        x = self.deconv0(x)  # batch * 64 * 2 * 2
        x = self.relu0(x)
        x = self.up0(x)  # batch * 64 * 4 * 4

        x = self.deconv1(x)  # batch * 4 * 4 * 32
        x = self.relu1(x)
        x = self.up1(x)  # batch * 8 * 8 * 32

        x = self.deconv2(x)  # batch * 8 * 8 * 32
        x = self.relu2(x)
        x = self.up2(x)  # batch * 16 * 16 * 32

        x = self.deconv3(x)  # batch * 16 * 16 * 1
        x = self.relu3(x)
        x = self.up3(x)  # batch * 32 * 32 * 1

        x = x[:, :, 1:30, 1:30]

        return x


class Disgauss(nn.Module):
    def __init__(self):
        super(Disgauss, self).__init__()

        self.z_dim = Config.z_dim
        self.n_labels = Config.n_labels

        self.ds0 = nn.Linear(self.z_dim, 1000)
        self.relu0 = nn.ReLU()
        self.ds1 = nn.Linear(1000, 1000)
        self.relu1 = nn.ReLU()
        self.ds2 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # batch 10

        x = self.ds0(x)
        x = self.relu0(x)
        # batch 1000

        x = self.ds1(x)
        x = self.relu1(x)
        # batch 1000

        x = self.ds2(x)
        # batch 1

        x = self.sigmoid(x)
        # batch 1

        return x


class Discateg(nn.Module):
    def __init__(self):
        super(Discateg, self).__init__()

        self.n_labels = Config.n_labels

        self.ds0 = nn.Linear(self.n_labels, 1000)
        self.relu0 = nn.ReLU()
        self.ds1 = nn.Linear(1000, 1000)
        self.relu1 = nn.ReLU()
        self.ds2 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # batch 2

        x = self.ds0(x)
        x = self.relu0(x)
        # batch 1000

        x = self.ds1(x)
        x = self.relu1(x)
        # batch 1000

        x = self.ds2(x)
        # batch 1

        x = self.sigmoid(x)
        # batch 1

        return x