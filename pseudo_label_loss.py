import torch
import numpy as np
import torch.nn as nn


def pseudo_label_loss(epoch_step, batch_step, x_logits, x_targets, u_logits, u_targets, config):
    x_logits = torch.log_softmax(x_logits, dim=1)
    u_logits = torch.log_softmax(u_logits, dim=1)
    
    loss = - (x_logits * x_targets).sum(dim=1).mean()
    loss += - (u_logits * u_targets).sum(dim=1).mean() * alpha(epoch_step + batch_step / config.iterations)

    return loss

def alpha(t, T1=0, T2=100, alpha_f=3.0):
    if t < T1:
        return 0.0
    if t > T1 and t < T2:
        return (t - T1) / (T2-T1) * alpha_f
    else:
        return alpha_f
