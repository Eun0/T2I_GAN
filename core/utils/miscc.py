import torch
import numpy as np
from scipy.stats import truncnorm

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_trainable_state_dict(model, file_name):

    state_dict = model.state_dict()
    for name,param in model.named_parameters():
        if not param.requires_grad:
            del state_dict[name]
    
    torch.save(state_dict, file_name)


def truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
  return truncation * values