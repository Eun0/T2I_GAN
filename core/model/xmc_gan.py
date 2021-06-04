import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv2d_nxn, linear

def disc_arch(img_size, nch):
    assert img_size in [64,128,256]

    if img_size == 256:
        in_channels = [1,2,4,8,8,16]
        out_channels = [1,2,4,8,8,16,16]
        resolution = [128,64,32,16,8,4,4]
        depth = 7
    elif img_size == 128:
        in_channels = [1,2,4,8,8]
        out_channels = [1,2,4,8,8,16]
        resolution = [64,32,16,8,4,4]
        depth = 6
    else:
        in_channels = [1,2,4,8]
        out_channels = [1,2,4,8,8]
        resolution = [32,16,8,4,4]
        depth = 5

    return {
        'in_channels': [3] + [i * nch for i in in_channels],
        'out_channels': [i * nch for i in out_channels],
        'downsample': [True] * depth, 
        'resolution': resolution,
        'depth': depth,
    }


class XMC_DISC(nn.Module):
    def __init__(self, cfg, cond_dim, **kwargs):
        super(XMC_DISC, self).__init__()
        ndf = cfg.DISC.NCH
        spec_norm = cfg.DISC.SPEC_NORM

        arch = disc_arch(img_size = cfg.IMG.SIZE, nch = ndf)

        self.conv_img = conv2d_nxn(in_dim = arch['in_channels'][0], out_dim = arch['out_channels'][0], kernel_size = 3, stride = 1, padding = 1, spec_norm= spec_norm)

        self.downblocks = nn.ModuleList(

            [ResBlockDown(in_dim = arch['in_channels'][i],
                  out_dim = arch['out_channels'][i],
                  downsample = arch['downsample'][i],
                  spec_norm = spec_norm) for i in range(1,arch['depth'])]
        )
        
        self.COND_DNET = PROJD_GET_LOGITS(cfg, cond_dim, in_dim = arch['out_channels'][-1], spec_norm = spec_norm)

    def forward(self, x, **kwargs):

        out = self.conv_img(x)

        for block in self.downblocks:
            out = block(out)
        
        return out

class PROJD_GET_LOGITS(nn.Module):
    def __init__(self, cfg, cond_dim, in_dim, spec_norm = True):
        super(PROJD_GET_LOGITS, self).__init__()
        assert cfg.DISC.UNCOND or cfg.DISC.COND
        self.uncond = cfg.DISC.UNCOND
        self.cond = cfg.DISC.COND
        # GSP
        #self.pool = nn.AvgPool2d(kernel_size=4, divisor_override=1)
        self.pool = nn.AvgPool2d(kernel_size=4)

        self.img_match = cfg.DISC.IMG_MATCH

        if self.img_match:
            self.proj_match = linear(in_dim, cond_dim, spec_norm=spec_norm)
        elif cfg.DISC.SENT_MATCH:
            self.proj_match = linear(cond_dim, in_dim, spec_norm=spec_norm)
        else:
            raise NotImplementedError

        if self.uncond:
            self.proj_logit = linear(in_dim, 1, spec_norm=spec_norm)

    def forward(self, out, sent_embs, **kwargs):
        # out [bs, c_in, 4, 4]
        # sent_embs [bs, cond_dim]
        out = self.pool(out) # [bs, c_in, 1, 1]
        out = out.view(out.size(0),-1) # [bs, c_in]
        if self.img_match:
            out_img = self.proj_match(out) # [bs, cond_dim]
        else:
            sent_embs = self.proj_match(sent_embs) # [bs, c_in]
            out_img = out

        match = 0.
        if self.cond:
            match += torch.einsum('be,be->b', out_img, sent_embs)
        if self.uncond:
            logit = self.proj_logit(out)
            logit = logit.view(-1)
            match += logit

        return [match, out, sent_embs]

class ResBlockDown(nn.Module):

    def __init__(self, in_dim, out_dim, downsample, spec_norm = True):
        super(ResBlockDown,self).__init__()
        self.learnable_sc = (in_dim != out_dim)
        self.downsample = downsample

        self.c1 = conv2d_nxn(in_dim = in_dim, out_dim = out_dim, kernel_size = 3, stride = 1, padding = 1, bias = False, spec_norm = spec_norm)
        self.c2 = conv2d_nxn(in_dim = out_dim, out_dim = out_dim, kernel_size = 3, stride = 1, padding = 1, bias =False, spec_norm = spec_norm)

        if self.learnable_sc:
            self.c_sc = conv2d_nxn(in_dim = in_dim, out_dim = out_dim, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):        
        out = self.residual(x)
        out += self.shortcut(x)
        return out

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            x = F.avg_pool2d(x, kernel_size = 2)
        return x

    def residual(self, x):
        out = F.relu(x, inplace=True)
        out = self.c1(out)
        out = F.relu(out, inplace=True)
        out = self.c2(out)
        if self.downsample:
            out = F.avg_pool2d(out, kernel_size = 2)
        return out