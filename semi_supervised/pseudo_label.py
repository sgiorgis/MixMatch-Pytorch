import torch
import numpy as np
import torch.nn as nn


class PseudoLabel(nn.Module):
    def __init__(self, config):
        super(PseudoLabel, self).__init__()
        self.config = config

    def forward(self, x_hat, y_one_hot, ub, model, ema_model, config):
        pseudo_labels = torch.eye(config.dataset_classes).to(config.device)
        pseudo_labels = pseudo_labels[model(ub).max(1)[1]]

        return model(x_hat), y_one_hot, model(ub), pseudo_labels
