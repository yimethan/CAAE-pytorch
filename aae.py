import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.input_dim = Config.input_dim
        self.n_l1 = Config.n_l1
        self.n_l2 = Config.n_l2
        self.z_dim = Config.z_dim
        self.n_labels = Config.n_labels

        self.e_dense_1 = nn.Linear(self.input_dim, self.n_l1)
        self.e_dense_2 = nn.Linear(self.n_l1, self.n_l2)
        self.e_latent_variable = nn.Linear(self.n_l2, self.z_dim)
        self.e_label = nn.Linear(self.n_l2, self.n_labels)

    def forward(self, x, supervised=False):
        x = F.pad(x, (1, 2, 1, 2))
        x = x.view(x.size(0), -1)

        x = self.e_dense_1(x)
        e_dense_1 = F.relu(x)
        x = self.e_dense_2(e_dense_1)
        e_dense_2 = F.relu(x)

        latent_variable = self.e_latent_variable(e_dense_2)
        cat_op = self.e_label(e_dense_2)

        if not supervised:
            softmax_label = F.softmax(cat_op, dim=1)
        else:
            softmax_label = cat_op

        return softmax_label, latent_variable


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.input_dim = Config.input_dim
        self.n_l1 = Config.n_l1
        self.n_l2 = Config.n_l2
        self.z_dim = Config.z_dim
        self.n_labels = Config.n_labels

        self.d_dense_1 = nn.Linear(self.z_dim + self.n_labels, self.n_l2)
        self.d_dense_2 = nn.Linear(self.n_l2, self.n_l1)
        self.d_output = nn.Linear(self.n_l1, self.input_dim)

    def forward(self, x):
        x = self.d_dense_1(x)
        d_dense_1 = F.relu(x)
        x = self.d_dense_2(d_dense_1)
        d_dense_2 = F.relu(x)
        x = self.d_output(d_dense_2)
        output = torch.sigmoid(x)

        x = output.view(-1, 1, 32, 32)
        x = x[:, :, 1:30, 1:30]

        return x


class Disgauss(nn.Module):
    def __init__(self):
        super(Disgauss, self).__init__()

        self.input_dim = Config.input_dim
        self.n_l1 = Config.n_l1
        self.n_l2 = Config.n_l2
        self.z_dim = Config.z_dim
        self.n_labels = Config.n_labels

        self.dc_g_den1 = nn.Linear(self.z_dim, self.n_l1)
        self.dc_g_den2 = nn.Linear(self.n_l1, self.n_l2)
        self.dc_g_output = nn.Linear(self.n_l2, 1)

    def forward(self, x):
        x = self.dc_g_den1(x)
        dc_den1 = F.relu(x)
        x = self.dc_g_den2(dc_den1)
        dc_den2 = F.relu(x)
        output = self.dc_g_output(dc_den2)

        return output


class Discateg(nn.Module):
    def __init__(self):
        super(Discateg, self).__init__()

        self.input_dim = Config.input_dim
        self.n_l1 = Config.n_l1
        self.n_l2 = Config.n_l2
        self.z_dim = Config.z_dim
        self.n_labels = Config.n_labels

        self.dc_c_den1 = nn.Linear(self.n_labels, self.n_l1)
        self.dc_c_den2 = nn.Linear(self.n_l1, self.n_l2)
        self.dc_c_output = nn.Linear(self.n_l2, 1)

    def forward(self, x):
        x = self.dc_c_den1(x)
        dc_den1 = F.relu(x)
        x = self.dc_c_den2(dc_den1)
        dc_den2 = F.relu(x)
        output = self.dc_c_output(dc_den2)

        return output
