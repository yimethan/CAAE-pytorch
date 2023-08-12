import torch
import torch.nn as nn
import torch.nn.functional as F

class AAE(nn.Module):
    def __init__(self, input_dim, n_l1, n_l2, z_dim, n_labels):
        super(AAE, self).__init__()
        self.input_dim = input_dim
        self.n_l1 = n_l1
        self.n_l2 = n_l2
        self.z_dim = z_dim
        self.n_labels = n_labels

    def dense(self, x, n1, n2):
        weights = nn.Parameter(torch.randn(n1, n2) * 0.01)
        bias = nn.Parameter(torch.zeros(n2))
        out = F.linear(x, weights, bias)
        return out

    def encoder(self, x, reuse=False, supervised=False):
        if reuse:
            self._get_name_scope().reuse_variables()
        with torch.no_grad():
            e_dense_1 = F.relu(self.dense(x, self.input_dim, self.n_l1, 'e_dense_1'))
            e_dense_2 = F.relu(self.dense(e_dense_1, self.n_l1, self.n_l2, 'e_dense_2'))
            latent_variable = self.dense(e_dense_2, self.n_l2, self.z_dim, 'e_latent_variable')
            cat_op = self.dense(e_dense_2, self.n_l2, self.n_labels, 'e_label')
            if not supervised:
                softmax_label = F.softmax(cat_op, dim=1)
            else:
                softmax_label = cat_op
            return softmax_label, latent_variable

    def decoder(self, x, reuse=False):
        if reuse:
            self._get_name_scope().reuse_variables()
        d_dense_1 = F.relu(self.dense(x, self.z_dim + self.n_labels, self.n_l2, 'd_dense_1'))
        d_dense_2 = F.relu(self.dense(d_dense_1, self.n_l2, self.n_l1, 'd_dense_2'))
        output = torch.sigmoid(self.dense(d_dense_2, self.n_l1, self.input_dim, 'd_output'))
        return output

    def discriminator_gauss(self, x, reuse=False):
        if reuse:
            self._get_name_scope().reuse_variables()
        dc_den1 = F.relu(self.dense(x, self.z_dim, self.n_l1, name='dc_g_den1'))
        dc_den2 = F.relu(self.dense(dc_den1, self.n_l1, self.n_l2, name='dc_g_den2'))
        output = self.dense(dc_den2, self.n_l2, 1, name='dc_g_output')
        return output

    def discriminator_categorical(self, x, reuse=False):
        if reuse:
            self._get_name_scope().reuse_variables()
        dc_den1 = F.relu(self.dense(x, self.n_labels, self.n_l1, name='dc_c_den1'))
        dc_den2 = F.relu(self.dense(dc_den1, self.n_l1, self.n_l2, name='dc_c_den2'))
        output = self.dense(dc_den2, self.n_l2, 1, name='dc_c_output')
        return output


# Usage

# aae = AAE(input_dim, n_l1, n_l2, z_dim, n_labels)
# x = torch.randn(32, input_dim)  # Example input tensor
# softmax_label, latent_variable = aae.encoder(x)
# output = aae.decoder(latent_variable)
