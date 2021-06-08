import torch
import torch.nn as nn
import torch.nn.functional as F
from wandb.sdk_py27.lib.telemetry import context

from .modules import conv2d_nxn, linear


def gen_arch(img_size, nch):
    assert img_size in [64,128,256]

    if img_size == 256:
        in_channels = [16,16,8,8,4,2,1]
        out_channels = [16,8,8,4,2,1,1]
        resolution = [8,16,32,64,128,256,256]
        depth = 7
    elif img_size == 128:
        in_channels = [16,8,8,4,2,1]
        out_channels = [8,8,4,2,1,1]
        resolution = [8,16,32,64,128,128]
        depth = 6
    else:
        in_channels = [8,8,4,2,1]
        out_channels = [8,4,2,1,1]
        resolution = [8,16,32,64,64]
        depth = 5

    return {
        'in_channels': [i * nch for i in in_channels],
        'out_channels': [i * nch for i in out_channels],
        'upsample': [True]*(depth-1) + [False],
        'resolution': resolution,
        'depth': depth,
    }

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


class XMC_GEN(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(XMC_GEN, self).__init__()
        self.ngf = cfg.TRAIN.NCH
        noise_dim = cfg.TRAIN.NOISE_DIM
        text_dim = cfg.TEXT.EMBEDDING_DIM
        nef = cfg.TRAIN.NEF

        arch = gen_arch(img_size = cfg.IMG.SIZE, nch = self.ngf)

        init_size = arch['in_channels'][0] * 4 * 4
        self.proj_sent = nn.Linear(cfg.TEXT.EMBEDDING_DIM, nef)
        self.proj_cond = nn.Linear(nef + noise_dim, init_size)

        self.upblocks = nn.ModuleList(
            [ResBlockUp(in_dim = arch['in_channels'][i],
                    out_dim = arch['out_channels'][i],
                    cond_dim = noise_dim + nef,
                    upsample = arch['upsample'][i] ) for i in range(0,2)]
        )

        self.attn_upblocks = nn.ModuleList(
            [AttnResBlockUp(in_dim = arch['in_channels'][i],
                            out_dim = arch['out_channels'][i],
                            cond_dim = noise_dim + nef + text_dim,
                            text_dim = text_dim,
                            upsample = arch['upsample'][i]) for i in range(2,arch['depth'])]
        )

        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(arch['out_channels'][-1], 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, sent_embs, words_embs, mask, **kwargs):

        noise = self.proj_noise(noise)
        sent_embs = self.proj_sent(sent_embs)

        global_cond = torch.cat([noise,sent_embs], dim=1)

        out = self.proj_cond(global_cond)
        out = out.view(out.size(0), -1 , 4, 4) # [bs, c_in, 4, 4]

        for gblock in self.upblocks:
            out = gblock(out, global_cond = global_cond)

        for attn_gblock in self.attn_upblocks:
            out = attn_gblock(out, global_cond=global_cond, words_embs=words_embs, mask=mask)

        out = self.conv_out(out)
        
        return out


class ResBlockUp(nn.Module):

    def __init__(self, in_dim, out_dim, cond_dim, upsample):
        super(ResBlockUp, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.upsample = upsample

        self.c1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.c2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(in_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.conv_gamma1 = nn.Conv1d(cond_dim, in_dim, 1, 1, 0)
        self.conv_beta1 = nn.Conv1d(cond_dim, in_dim, 1, 1, 0)
        self.conv_gamma2 = nn.Conv1d(cond_dim, out_dim, 1, 1, 0)
        self.conv_beta2 = nn.Conv1d(cond_dim, out_dim, 1, 1, 0)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0)

    def forward(self, x, global_cond, **kwargs):
        
        out = self.residual(x, global_cond)
        out += self.shortcut(x)

        return out 


    def shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, global_cond):
        # global_cond [bs, noise_dim + nef]
        BS = x.size(0)
        out = self.conv_gamma1(global_cond).view(BS,-1,1,1) * self.bn1(x) + self.conv_beta1(global_cond).view(BS,-1,1,1)
        out = F.relu(out, inplace=True)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2)
        
        out = self.c1(out)
        out = self.conv_gamm2(global_cond).view(BS,-1,1,1) * self.bn2(out) + self.conv_beta2(global_cond).view(BS,-1,1,1)
        out = F.relu(out, inplace=True)
        out = self.c2(out)

        return out


class AttnResBlockUp(nn.Module):

    def __init__(self, in_dim, out_dim, cond_dim, text_dim, upsample):
        super(ResBlockUp, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.upsample = upsample

        self.proj_region = nn.Conv1d(in_dim, text_dim, 1, 1, 0, bias = False)

        self.c1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.c2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(in_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.conv_gamma1 = nn.Conv1d(cond_dim, in_dim, 1, 1, 0)
        self.conv_beta1 = nn.Conv1d(cond_dim, in_dim, 1, 1, 0)
        self.conv_gamma2 = nn.Conv1d(cond_dim, out_dim, 1, 1, 0)
        self.conv_beta2 = nn.Conv1d(cond_dim, out_dim, 1, 1, 0)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0)

    def forward(self, x, global_cond, words_embs, mask, **kwargs):
        
        context_embs, attn = self.get_context_embs(x, words_embs=words_embs, mask=mask)
        out = self.residual(x, global_cond = global_cond, context_embs = context_embs)
        out += self.shortcut(x)

        return out
    
    def get_context_embs(self, x, words_embs, mask):
        # x [bs, c_in, h, w]
        # words_embs [bs, text_dim, T]
        # mask [bs, T]

        x = x.view(x.size(0), x.size(1), -1) # [bs, c_in, h*w]
        x = self.proj_region(x) # [bs, text_dim, h*w]

        x_norm = F.normalize(x, p=2, dim= 1)
        words_embs_norm = F.normalize(words_embs, p=2, dim=1)

        attn = torch.matmul(x_norm.permute(0,2,1), words_embs_norm) # [bs, h*w, T]

        attn_mask = mask.view(mask.size(0), 1, -1) # [bs, 1, T]
        attn_mask = attn_mask.repeat(1, attn.size(1), 1) # [bs, h*w, T]

        attn.masked_fill_(attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=2)

        context_embs = torch.matmul(attn, words_embs.permute(0,2,1)) # [bs, h*w, text_dim]
        context_embs = context_embs.permute(0,2,1) # [bs, text_dim, h*w]

        return context_embs, attn

    def shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, global_cond, context_embs):
        # x [bs, c_in, h, w]
        # global_cond [bs, noise_dim + nef]
        # context_embs [bs, text_dim, h*w]

        BS = x.size(0)
        H = W = x.size(-1)

        global_cond = torch.view(BS, 1, -1) # [bs, 1, noise_dim + nef]
        global_cond = torch.repeat(1, H*W, 1) # [bs, h*w, noise_dim + nef]
        global_cond = global_cond.permute(0, 2, 1) # [bs, noise_dim + nef, h*w]
        
        cond = torch.cat([global_cond, context_embs], dim=1) # [bs, noise_dim + nef + text_dim, h*w]

        out = self.conv_gamma1(cond).view(BS, -1, H, W) * self.bn1(x) + self.conv_beta1(cond).view(BS, -1, H, W)
        out = F.relu(out, inplace=True)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2)
        
        out = self.c1(out)
        out = self.conv_gamm2(cond).view(BS, -1, H, W) * self.bn2(out) + self.conv_beta2(cond).view(BS, -1, H, W)
        out = F.relu(out, inplace=True)
        out = self.c2(out)

        return out

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

        return [match, out_img, sent_embs]

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