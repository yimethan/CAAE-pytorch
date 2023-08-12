import torch
import torch.nn as nn
import torch.functional as F


def conv2d(input, name, kshape, strides=[1, 1], pad=0):
    conv_layer = nn.Conv2d(in_channels=kshape[2], out_channels=kshape[3], kernel_size=(kshape[0], kshape[1]), stride=strides, padding=pad)
    with torch.no_grad():
        conv_layer.weight.copy_(torch.from_numpy(np.transpose(W, (0, 1, 3, 2))))
        conv_layer.bias.copy_(torch.from_numpy(b))
    out = conv_layer(input)
    out = F.relu(out)
    return out


def deconv2d(input, name, kshape, n_outputs, strides=[1, 1], pad=0):
    deconv_layer = nn.ConvTranspose2d(in_channels=input.size(1), out_channels=n_outputs, kernel_size=(kshape[0], kshape[1]), stride=strides, padding=pad)
    with torch.no_grad():
        deconv_layer.weight.copy_(torch.from_numpy(np.transpose(W, (3, 2, 0, 1))))
        deconv_layer.bias.copy_(torch.from_numpy(b))
    out = deconv_layer(input)
    out = F.relu(out)
    return out


def maxpool2d(x, name, kshape=[1, 2, 2, 1], strides=[1, 2]):
    pool_layer = nn.MaxPool2d(kernel_size=(kshape[1], kshape[2]), stride=strides)
    out = pool_layer(x)
    return out


def upsample(input, name, factor=[2, 2]):
    out = F.interpolate(input, scale_factor=factor, mode='bilinear', align_corners=False)
    return out


def fullyConnected(input, name, output_size):
    input_size = input.size(1)
    W = nn.Parameter(torch.randn(input_size, output_size) * 0.01)
    b = nn.Parameter(torch.zeros(output_size))
    out = torch.matmul(input, W) + b
    return out


def dropout(input, name, keep_rate):
    out = F.dropout(input, p=1 - keep_rate, training=self.training)
    return out