import torch
import numpy as np
import torch.nn as nn

from semi_supervised.mix_up import MixUp


class MixMatch(nn.Module):
    def __init__(self, config):
        super(MixMatch, self).__init__()
        self.config = config
        self.t = config.t
        self.k = config.k
        self.alpha = config.alpha
        self.mix_up = MixUp(self.alpha)

    def forward(self, x_hat, y_one_hot, ub, model, ema_model, config):

        mixed_inputs, mixed_targets = self.forward_semi_supervised(x_hat, y_one_hot, ub, model, ema_model, config) if ub \
            else self.forward_supervised(x_hat, y_one_hot)

        mixed_inputs = list(torch.split(mixed_inputs, self.config.batch_size))
        mixed_inputs = self.interleave(mixed_inputs, self.config.batch_size)

        logits = [model(mixed_input) for mixed_input in mixed_inputs]
        logits = self.interleave(logits, self.config.batch_size)

        x_logits = logits[0]
        x_targets = mixed_targets[:self.config.batch_size]

        if not ub:
            return x_logits, x_targets, None, None

        u_logits = torch.cat(logits[1:], dim=0)
        u_targets = mixed_targets[self.config.batch_size:]

        return x_logits, x_targets, u_logits, u_targets

    def forward_supervised(self, x_hat, y_one_hot):

        x_hat_shuffled, y_one_hot_shuffled = self.shuffle(x_hat, y_one_hot)
        x_prime, p_prime = self.mix_up(x_hat, y_one_hot, x_hat_shuffled, y_one_hot_shuffled)

        return x_prime, p_prime

    def forward_semi_supervised(self, x_hat, y_one_hot, ub, model, ema_model, config):
        qb = self.guess_labels(model, ema_model, ub, config)
        u_hat_data = torch.cat(ub, dim=0)
        u_hat_labels = torch.cat([qb for _ in range(self.k)], dim=0)

        x_prime, p_prime, u_prime, q_prime = self.apply_mix_up_strategy(x_hat, y_one_hot, u_hat_data, u_hat_labels,
                                                                        config)

        mixed_inputs = torch.cat([x_prime, u_prime], dim=0)
        mixed_targets = torch.cat([p_prime, q_prime], dim=0)

        return mixed_inputs, mixed_targets

    def guess_labels(self, model, ema_model, ub, config):
        with torch.no_grad():
            if config.ema:
                qb_init = torch.stack([ema_model(ub[i]) for i in range(len(ub))], dim=1)
            else:
                qb_init = torch.stack([model(ub[i]) for i in range(len(ub))], dim=1)
            qb_average = torch.softmax(qb_init, dim=2).mean(dim=1)
            qb = self.sharpen(qb_average)

        return qb

    def apply_mix_up_strategy(self, x_hat, y_one_hot, u_hat_data, u_hat_labels, config):
        if config.mix_up == 'all':
            wx = torch.cat([x_hat, u_hat_data], dim=0)
            wy = torch.cat([y_one_hot, u_hat_labels], dim=0)
            wx, wy = self.shuffle(wx, wy)
            x_prime, p_prime = self.mix_up(x_hat, y_one_hot, wx[:x_hat.shape[0]], wy[:x_hat.shape[0]])
            u_prime, q_prime = self.mix_up(u_hat_data, u_hat_labels, wx[x_hat.shape[0]:], wy[x_hat.shape[0]:])
        elif config.mix_up == 'labeled':
            x_hat_shuffled, y_one_hot_shuffled = self.shuffle(x_hat, y_one_hot)
            x_prime, p_prime = self.mix_up(x_hat, y_one_hot, x_hat_shuffled, y_one_hot_shuffled)
            u_prime = u_hat_data
            q_prime = u_hat_labels
        elif config.mix_up == 'unlabeled':
            u_hat_data_shuffled, u_hat_labels_shuffled = self.shuffle(u_hat_data, u_hat_labels)
            x_prime = x_hat
            p_prime = y_one_hot
            u_prime, q_prime = self.mix_up(u_hat_data, u_hat_labels, u_hat_data_shuffled, u_hat_labels_shuffled)
        elif config.mix_up == 'separate':
            x_hat_shuffled, y_one_hot_shuffled = self.shuffle(x_hat, y_one_hot)
            u_hat_data_shuffled, u_hat_labels_shuffled = self.shuffle(u_hat_data, u_hat_labels)
            x_prime, p_prime = self.mix_up(x_hat, y_one_hot, x_hat_shuffled, y_one_hot_shuffled)
            u_prime, q_prime = self.mix_up(u_hat_data, u_hat_labels, u_hat_data_shuffled, u_hat_labels_shuffled)
        else:
            x_prime = x_hat
            p_prime = y_one_hot
            u_prime = u_hat_data
            q_prime = u_hat_labels

        return x_prime, p_prime, u_prime, q_prime

    def sharpen(self, p):
        temperature = p ** (1 / self.t)
        sharpened = temperature / temperature.sum(dim=1, keepdim=True)
        return sharpened

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        return offsets

    @staticmethod
    def shuffle(x, y):
        shuffler = torch.randperm(x.shape[0])
        return x[shuffler], y[shuffler]
