import torch
import numpy as np
import torch.nn as nn


class MixUp(nn.Module):
    def __init__(self, alpha=0.75):
        super(MixUp, self).__init__()
        self.alpha = alpha

    def forward(self, x1_data, x1_labels, x2_data, x2_labels):
        lambda_prelim = np.random.beta(self.alpha, self.alpha)
        lambda_prime = max(lambda_prelim, 1 - lambda_prelim)

        x_prime = lambda_prime * x1_data + (1 - lambda_prime) * x2_data
        p_prime = lambda_prime * x1_labels + (1 - lambda_prime) * x2_labels

        return x_prime, p_prime

class MixUpAlg(nn.Module):
	def __init__(self, config):
		super(MixUpAlg,self).__init__()
		self.config = config
		self.t = config.t
		self.k = config.k
		self.alpha = config.alpha
		self.mix_up = MixUp(self.alpha)

	def forward(self, x_hat, y_one_hot, ub, model):
		with torch.no_grad():
			qb_init = torch.stack([model(ub[i]) for i in range(len(ub))], dim=1)
			qb_average = torch.softmax(qb_init, dim=2).mean(dim=1)
			qb = qb_average
		u_hat_data = torch.cat(ub, dim=0)
		u_hat_labels = torch.cat([qb for _ in range(self.k)], dim=0)

		wx = torch.cat([x_hat, u_hat_data], dim=0)
		wy = torch.cat([y_one_hot, u_hat_labels], dim=0)

		x_prime, p_prime = self.mix_up(x_hat, y_one_hot, wx[:x_hat.shape[0]], wy[:x_hat.shape[0]])
		u_prime, q_prime = self.mix_up(u_hat_data, u_hat_labels, wx[x_hat.shape[0]:], wy[x_hat.shape[0]:])

		mixed_inputs = torch.cat([x_prime, u_prime], dim=0)
		mixed_targets = torch.cat([p_prime, q_prime], dim=0)

		mixed_inputs = list(torch.split(mixed_inputs, self.config.batch_size))
		mixed_inputs = self.interleave(mixed_inputs, self.config.batch_size)

		logits = [model(mixed_input) for mixed_input in mixed_inputs]
		logits = self.interleave(logits, self.config.batch_size)

		x_logits = logits[0]
		u_logits = torch.cat(logits[1:], dim=0)

		x_targets = mixed_targets[:self.config.batch_size]
		u_targets = mixed_targets[self.config.batch_size:]

		return x_logits, x_targets, u_logits, u_targets

	def interleave(self, xy, batch):
		nu = len(xy) - 1
		offsets = self.interleave_offsets(batch, nu)
		xy = [[v[offsets[p]:offsets[p+1]] for p in range(nu + 1)] for v in xy]
		for i in range(1, nu + 1):
			xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
		return [torch.cat(v, dim=0) for v in xy]

	def interleave_offsets(self, batch, nu):
		groups = [batch // (nu + 1)] * (nu + 1)
		for x in range(batch - sum(groups)):
			groups[-x -1] +=1
		offsets = [0]
		for g in groups:
			offsets.append(offsets[-1] + g)
		return offsets
