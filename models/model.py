import torch
import torch.nn as nn

from aae import AAE
from caae import CAAE
from config import Config

class Model(nn.Module):

    def __init__(self, model='AAE', unknown_attack=None):
        super(Model, self).__init__()

        self.is_build = False
        self.unknown_attack = unknown_attack
        self.data_root = Config.data_root
        self.input_dim = Config.input_dim
        self.n_l1 = Config.n_l1
        self.n_l2 = Config.n_l2
        self.z_dim = Config.z_dim
        # self.batch_size_unknown = 0
        # self.batch_size = Config.batch_size
        # self.n_epochs = Config.n_epochs
        # self.supervised_lr = Config.supervised_lr
        # self.reconstruction_lr = Config.reconstruction_lr
        # self.regularization_lr = Config.regularization_lr
        # self.beta1_sup = 0.9
        # self.beta1 = 0.5
        # self.beta2 = 0.9
        # self.num_critic = 5
        self.n_labels = 2
        # self.n_labeled = self.data_info['train_label']
        # self.validation_size = self.data_info['validation']
        if model == 'AAE':
            self.model = AAE(self.input_dim, self.n_l1, self.n_l2, self.z_dim, self.n_labels)
            self.model_name = ''
        else:
            self.model = CAAE(self.n_labels, self.z_dim)
            self.model_name = 'CNN_WGAN'

