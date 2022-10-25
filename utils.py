import torch 
import torch.nn as nn
import numpy as np

from torch.distributions import Normal

def get_param_groups(net, weight_decay, norm_suffix='weight_g'):
  """Get two parameter groups from `net`: One named "normalized" which will
  override the optimizer with `weight_decay`, and one named "unnormalized"
  which will inherit all hyperparameters from the optimizer.
  Args:
      net (torch.nn.Module): Network to get parameters from
      weight_decay (float): Weight decay to apply to normalized weights.
      norm_suffix (str): Suffix to select weights that should be normalized.
          For WeightNorm, using 'weight_g' normalizes the scale variables.
      verbose (bool): Print out number of normalized and unnormalized parameters.
  """
  norm_params = []
  unnorm_params = []
  for n, p in net.named_parameters():
    if n.endswith(norm_suffix):
      norm_params.append(p)
    else:
      unnorm_params.append(p)
  param_groups = [{'name': 'normalized', 'params': norm_params, 'weight_decay': weight_decay},
                  {'name': 'unnormalized', 'params': unnorm_params}]
  return param_groups

def get_optimizer(params, model):
  """Returns the optimizer that should be used based on params."""
  param_groups = get_param_groups(model, params.wd, norm_suffix='weight_g')
  if params.optimizer == 'sgd':
    opt = torch.optim.SGD(param_groups, lr=params.lr)
  elif params.optimizer == 'adam':
    opt = torch.optim.Adam(param_groups, lr=params.lr)
  elif params.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=params.lr)
  else:
    raise ValueError("Optimizer was not recognized")
  return opt

def bits_per_dim(x, nll):
  """Get the bits per dimension implied by using model with `loss`
  for compressing `x`, assuming each entry can take on `k` discrete values.

  Args:
      x (torch.Tensor): Input to the model. Just used for dimensions.
      nll (torch.Tensor): Scalar negative log-likelihood loss tensor.

  Returns:
      bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
  """
  dim = np.prod(x.size()[1:])
  bpd = nll / (np.log(2) * dim)
  return bpd


class NLLLoss(nn.Module):
  """Negative log-likelihood loss assuming isotropic gaussian with unit norm.
  Args:
      k (int or float): Number of discrete values in each input dimension.
          E.g., `k` is 256 for natural images.
  See Also:
      Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
  """
  def __init__(self):
    super(NLLLoss, self).__init__()
    self.k = 2**8
    self.dist = Normal(0, 1)


  def forward(self, z, sldj):
    prior_ll = self.dist.log_prob(z)
    prior_ll = prior_ll.flatten(1).sum(-1) - np.log(self.k) * np.prod(z.size()[1:])
    ll = prior_ll + sldj
    nll = -ll
    return nll.mean(), bits_per_dim(z, nll).mean()
  


class AddNoise(nn.Module):
  """ add Gaussian or Uniform Noise to the input """
  def __init__(self):
    super(AddNoise,  self).__init__()
    self.nvals = 2**8

  def forward(self, x):
    x = x * (self.nvals - 1) + torch.rand(x.size())
    x = x / self.nvals
    return x
