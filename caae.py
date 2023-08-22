import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config


class Encoder(nn.Module):

    def __init__(self, keep_prob=Config.keep_prob, supervised=False):

        super(Encoder, self).__init__()
        self.n_labels = Config.n_labels
        self.z_dim = Config.z_dim

        self.keep_prob = keep_prob
        self.supervised = supervised

        self.conv0 = nn.ReLU(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3))
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.ReLU(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ReLU(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.ReLU(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.dropout(p=self.keep_prob)

    def fullyConnected(self, input, output_size):
        input_size = input.size()[1:]  # Get input size excluding batch dimension
        input_size = int(np.prod(input_size))  # Compute total input size

        # Define the weight and bias parameters using PyTorch constructs
        W = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_size, output_size, requires_grad=True)))
        b = nn.Parameter(nn.init.xavier_normal_(torch.empty(output_size, requires_grad=True)))

        # Reshape the input tensor to 2D
        input = input.view(-1, input_size)

        # Perform the linear transformation
        out = torch.addmm(b, input, W)

        return out

    def forward(self, x):

        x = x.view(-1, 1, 29, 29)
        x = F.pad(x, (2, 2, 1, 2))

        x = self.conv0(x)
        x = self.pool0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.dropout(x)

        latent = self.fullyConnected(x, self.z_dim)
        cat_op = self.fullyConnected(x, self.n_labels)

        if self.supervised:
            softmax_label = cat_op
        else:
            softmax_label = F.softmax(cat_op, dim=1)

        return softmax_label, latent


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv0 = nn.ReLU(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3))
        self.deconv1 = nn.ReLU(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3))
        self.deconv2 = nn.ReLU(nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3))
        self.deconv3 = nn.ReLU(nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3))

    def upsample(self, x, factor=(2, 2)):
        size = (int(x.shape[2] * factor[0]), int(x.shape[3] * factor[1]))
        out = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return out

    def crop_and_reshape(self, input_tensor, top, left, height, width):
        cropped_tensor = input_tensor[:, top:top + height, left:left + width]
        reshaped_tensor = cropped_tensor.view(-1, height * width)
        return reshaped_tensor

    def forward(self, x):
        x = self.deconv0(x)
        x = self.upsample(x)
        x = self.deconv1(x)
        x = self.upsample(x)
        x = self.deconv2(x)
        x = self.upsample(x)
        x = self.deconv3(x)
        x = self.upsample(x)

        x = self.crop_and_reshape(x, 1, 1, 29, 29)

        return x


class Discriminator(nn.Module):

    def __init__(self, tag='g'):
        super(Discriminator, self).__init__()

        self.z_dim = Config.z_dim
        self.n_labels = Config.n_labels
        self.tag = tag

        self.ds0 = nn.ReLU(nn.Linear(self.z_dim, 1000))
        self.ds1 = nn.ReLU(nn.Linear(1000, 1000))
        self.ds2 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

        self.d0 = nn.ReLU(nn.Linear(self.n_labels, 1000))

    def discriminator_gauss(self, x):
        x = self.ds0(x)
        x = self.ds1(x)
        x = self.ds2(x)
        x = self.sigmoid(x)

        return x

    def discriminator_categorical(self, x):
        x = self.d0(x)
        x = self.ds1(x)
        x = self.ds2(x)
        x = self.sigmoid(x)

        return x

    def forward(self, x):
        if self.tag == 'g':
            return self.discriminator_gauss(x)

        else:
            return self.discriminator_categorical(x)