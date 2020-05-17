import torch
import numpy as np


def mix_match_loss(epoch_step, batch_step, x_logits, x_targets, u_logits, u_targets, config):
    x_logits = torch.log_softmax(x_logits, dim=1)
    lx = - (x_logits * x_targets).sum(dim=1).mean()

    if u_logits is None:
        return lx

    u_logits = torch.softmax(u_logits, dim=1)
    mse_loss = torch.nn.MSELoss()
    lu = mse_loss(u_logits, u_targets)

    lambda_u = config.lambda_u * linear_rampup(epoch_step + batch_step / config.iterations)
    loss = lx + lambda_u * lu

    return loss


def pseudo_label_loss(epoch_step, batch_step, x_logits, x_targets, u_logits, u_targets, config):
    x_logits = torch.log_softmax(x_logits, dim=1)
    u_logits = torch.log_softmax(u_logits, dim=1)

    loss = - (x_logits * x_targets).sum(dim=1).mean()
    loss += - (u_logits * u_targets).sum(dim=1).mean() * alpha(epoch_step + batch_step / config.iterations)

    return loss


def mix_up_loss(x_logits, x_targets, u_logits, u_targets):
    x_logits = torch.log_softmax(x_logits, dim=1)
    u_logits = torch.softmax(u_logits, dim=1)

    loss_X = torch.nn.CrossEntropyLoss()
    _, indices = torch.max(x_targets, dim=1)
    loss1 = loss_X(x_logits, indices)

    loss_U = torch.nn.CrossEntropyLoss()
    _, indices = torch.max(u_targets, dim=1)
    loss2 = loss_U(u_logits, indices)

    total_loss = loss1 + loss2
    return total_loss


def linear_rampup(current, rampup_length=50):
    current = np.clip(current / rampup_length, 0.0, 1.0)
    return float(current)


def alpha(t, T1=0, T2=100, alpha_f=3.0):
    if t < T1:
        return 0.0

    if T1 < t < T2:
        return (t - T1) / (T2 - T1) * alpha_f

    return alpha_f
