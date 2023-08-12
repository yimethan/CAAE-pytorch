import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import *

class CAAE(nn.Module):
    def __init__(self, n_labels, z_dim):
        super(CAAE, self).__init__()
        self.n_labels = n_labels
        self.z_dim = z_dim

    def dense(self, x, n1, n2, name):
        weights = nn.Parameter(torch.randn(n1, n2) * 0.01)
        bias = nn.Parameter(torch.zeros(n2))
        out = F.linear(x, weights, bias)
        return out

    def encoder(self, x, keep_prob, reuse=False, supervised=False):
        if reuse:
            self._get_name_scope().reuse_variables()
        with torch.no_grad():
            x_input = x.view(-1, 1, 29, 29)
            x_input = F.pad(x_input, (2, 2, 1, 2))  # Pad the input

            conv1 = F.relu(conv2d(x_input, in_channels=1, out_channels=32, kernel_size=3, name='e_conv1'))
            pool1 = maxpool2d(conv1, kernel_size=2, name='e_pool1')

            conv2 = F.relu(conv2d(pool1, in_channels=32, out_channels=32, kernel_size=3, name='e_conv2'))
            pool2 = maxpool2d(conv2, kernel_size=2, name='e_pool2')

            conv3 = F.relu(conv2d(pool2, in_channels=32, out_channels=64, kernel_size=3, name='e_conv3'))
            pool3 = maxpool2d(conv3, kernel_size=2, name='e_pool3')

            conv4 = F.relu(conv2d(pool3, in_channels=64, out_channels=64, kernel_size=3, name='e_conv4'))
            pool4 = maxpool2d(conv4, kernel_size=2, name='e_pool4')
            drop4 = dropout(pool4, keep_rate=keep_prob)

            latent_variable = fullyConnected(drop4, output_size=self.z_dim, name='e_latent_variable')
            cat_op = fullyConnected(drop4, output_size=self.n_labels, name='e_label')

            if not supervised:
                softmax_label = F.softmax(cat_op, dim=1)
            else:
                softmax_label = cat_op
            return softmax_label, latent_variable

    def decoder(self, x, reuse=False):
        if reuse:
            self._get_name_scope().reuse_variables()
        fc1 = fullyConnected(x, output_size=2 * 2 * 64, name='d_fc1')
        fc1 = fc1.view(-1, 64, 2, 2)

        deconv1 = F.relu(deconv2d(fc1, in_channels=64, out_channels=64, kernel_size=3, name='d_deconv1'))
        up1 = upsample(deconv1, scale_factor=2, name='d_up1')

        deconv2 = F.relu(deconv2d(up1, in_channels=64, out_channels=32, kernel_size=3, name='d_deconv2'))
        up2 = upsample(deconv2, scale_factor=2, name='d_up2')

        deconv3 = F.relu(deconv2d(up2, in_channels=32, out_channels=32, kernel_size=3, name='d_deconv3'))
        up3 = upsample(deconv3, scale_factor=2, name='d_up3')

        deconv4 = F.relu(deconv2d(up3, in_channels=32, out_channels=1, kernel_size=3, name='d_deconv4'))
        up4 = upsample(deconv4, scale_factor=2, name='d_up4')

        out = up4[:, :, 1:30, 1:30]
        out = out.view(-1, 29 * 29)
        return out

    def discriminator_gauss(self, x, reuse=False):
        if reuse:
            self._get_name_scope().reuse_variables()
        dc_den1 = F.relu(self.dense(x, self.z_dim, 1000, name='dc_g_den1'))
        dc_den2 = F.relu(self.dense(dc_den1, 1000, 1000, name='dc_g_den2'))
        output = self.dense(dc_den2, 1000, 1, name='dc_g_output')
        return torch.sigmoid(output)

    def discriminator_categorical(self, x, reuse=False):
        if reuse:
            self._get_name_scope().reuse_variables()
        dc_den1 = F.relu(self.dense(x, self.n_labels, 1000, name='dc_c_den1'))
        dc_den2 = F.relu(self.dense(dc_den1, 1000, 1000, name='dc_c_den2'))
        output = self.dense(dc_den2, 1000, 1, name='dc_c_output')
        return torch.sigmoid(output)
