import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torchvision.utils as vutils


def save_imgs(inputs, filename, size = None, nrow = 8):

    if size is not None:
        inputs = inputs.detach().view(size)
    vutils.save_image(inputs, filename, normalize=True, scale_each=True, nrow=nrow)

def conv2d_nxn(in_dim, out_dim, kernel_size, stride=1, padding=0, bias=True, groups = 1, spec_norm=False):

    conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias, groups = groups)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def conv1d_nxn(in_dim, out_dim, kernel_size, stride=1, padding=0, bias=True, groups=1, spec_norm=False):

    conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, bias=bias, groups=groups)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv

def linear(in_dim, out_dim, bias=True, spec_norm=False):

    fc = nn.Linear(in_dim, out_dim, bias=bias)
    if spec_norm:
        fc = spectral_norm(fc)
    return fc


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


        
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes * 2,3,1,1),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block