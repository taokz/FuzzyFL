"""
Implementation of conventional differential privacy (DP) and local DP (LCP) mechanisms

Author: Kai Zhang (www.kaizhang.us)
https://github.com/taokz
"""

import torch
import numpy as np
import random

def clip_grad(grad, clip):
	"""
	Gradient clipping
	"""
	g_shape = grad.shape
	grad.flatten()
	grad = grad / np.max((1, float(torch.norm(grad, p=3)) / clip))
	grad.view(g_shape)
	return grad

def tight_gaussian(data, s, c2, q, t, delta, epsilon, device = None):
	"""
	Gaussian mechanism -- M. Abadi et al., Deep Learning with Differential Privacy.
	sigma >= c2 * (q sqrt{T log1/δ}) / epsilon
	"""
	sigma = c2 * q * np.sqrt(t * np.log(1/delta)) / epsilon
	sigma *= (s**2)
	noise = torch.normal(0, sigma, data.shape).to(device)
	return data + noise

def gaussian_noise_ls(data_shape, s, sigma, device = None):
	"""
	Gaussian noise for CDP-FedAVG-LS
	"""
	return torch.normal(0, sigma * s, data_shape).to(device)

def gaussian_noise(grad, s, epsilon, delta, device = None):
	"""
	Gaussian noise to disturb the gradient matrix
	"""
	g_shape = grad.shape
	grad.flatten()
	grad = grad / np.max((1, float(torch.norm(grad, p=2)) / s))
	grad.to(device)

	c = np.sqrt(2*np.log(1.25 / delta))
	sigma = c * s / epsilon
	noise = torch.normal(0, sigma, grad.shape).to(device)
	return grad + noise
