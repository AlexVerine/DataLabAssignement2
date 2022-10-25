import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_dim(tensor, dim=None, keepdims=False):
  """Take the mean along multiple dimensions.
  Args:
      tensor (torch.Tensor): Tensor of values to average.
      dim (list): List of dimensions along which to take the mean.
      keepdims (bool): Keep dimensions rather than squeezing.
  Returns:
      mean (torch.Tensor): New tensor of mean value(s).
  """
  if dim is None:
    return tensor.mean()
  else:
    if isinstance(dim, int):
      dim = [dim]
    dim = sorted(dim)
    for d in dim:
      tensor = tensor.mean(dim=d, keepdim=True)
    if not keepdims:
      for i, d in enumerate(dim):
        tensor.squeeze_(d-i)
    return tensor


class ActNorm(nn.Module):
  """Activation normalization for 2D inputs.
  The bias and scale get initialized using the mean and variance of the
  first mini-batch. After the init, bias and scale are trainable parameters.
  Adapted from:
      > https://github.com/openai/glow
  """
  def __init__(self, num_features, scale=1., return_ldj=False):
    super(ActNorm, self).__init__()
    self.register_buffer('is_initialized', torch.zeros(1))
    self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
    self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    self.num_features = num_features
    self.scale = float(scale)
    self.eps = 1e-6
    self.return_ldj = return_ldj

  def initialize_parameters(self, x):
    if not self.training:
      return

    with torch.no_grad():
      bias = -mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
      v = mean_dim((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
      logs = (self.scale / (v.sqrt() + self.eps)).log()
      self.bias.data.copy_(bias.data)
      self.logs.data.copy_(logs.data)
      self.is_initialized += 1.

  def _center(self, x, reverse=False):
    if reverse:
      return x - self.bias
    else:
      return x + self.bias

  def _scale(self, x, sldj, reverse=False):
    logs = self.logs
    if reverse:
      x = x * logs.mul(-1).exp()
    else:
      x = x * logs.exp()

    if sldj is not None:
      ldj = logs.sum() * x.size(2) * x.size(3)
      if reverse:
        sldj = sldj - ldj
      else:
        sldj = sldj + ldj

    return x, sldj

  def forward(self, x, ldj=None, reverse=False):
    if not self.is_initialized:
      self.initialize_parameters(x)

    if reverse:
      x, ldj = self._scale(x, ldj, reverse)
      x = self._center(x, reverse)
    else:
      x = self._center(x, reverse)
      x, ldj = self._scale(x, ldj, reverse)

    if self.return_ldj:
      return x, ldj
    return x

def squeeze(x, reverse=False):
  """Trade spatial extent for channels. In forward direction, convert each
  1x4x4 volume of input into a 4x1x1 volume of output.
  Args:
      x (torch.Tensor): Input to squeeze or unsqueeze.
      reverse (bool): Reverse the operation, i.e., unsqueeze.
  Returns:
      x (torch.Tensor): Squeezed or unsqueezed tensor.
  """
  b, c, h, w = x.size()
  # return x
  if reverse:
    # Unsqueeze
    x = x.view(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(b, c // 4, h * 2, w * 2)
  else:
    # Squeeze
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(b, c * 2 * 2, h // 2, w // 2)
  return x


class LogitTransform(nn.Module):
  """
  The proprocessing step used in Real NVP:
  y = sigmoid(x) - a / (1 - 2a)
  x = logit(a + (1 - 2a)*y)
  """

  def __init__(self):
    nn.Module.__init__(self)
    self.alpha = 1e-6


  def forward(self, x, sldj=None, reverse=False):
    if reverse:
      return self.inverse(x, sldj)
    else:
      s = self.alpha + (1 - 2 * self.alpha) * x
      y = torch.log(s) - torch.log(1 - s)
      if sldj is None:
        return y, None
      return y, sldj + self._logdetgrad(x).view(x.size(0), -1).sum(1, keepdim=True).view(-1)

  def inverse(self, y, sldj=None):
    x = (torch.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
    if sldj is None:
      return x, None
    return x, sldj - self._logdetgrad(x).view(x.size(0), -1).sum(1, keepdim=True)

  def _logdetgrad(self, x):
    s = self.alpha + (1 - 2 * self.alpha) * x
    logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * self.alpha)

    return logdetgrad

  def __repr__(self):
    return ('{name}({alpha})'.format(name=self.__class__.__name__, **self.__dict__))



class Glow(nn.Module):
  """ Invertible Conditionnal Glow for 2D inputs.
  Described in Guided Image Generation with Conditional Invertible Neural Networks
  https://arxiv.org/abs/1907.02392 """

  def __init__(self, in_channels, num_channels, num_levels, num_steps, params):
    super(Glow, self).__init__()
    self.params = params
    self.flows = _Glow(in_channels=in_channels*4, # RGB after squeeze
                       mid_channels=num_channels,
                       num_levels=num_levels,
                       num_steps=num_steps,
                       params=params)
    self.init = LogitTransform()

  def forward(self, x, sldj=None, reverse=False):
    if not reverse:
      x, sldj = self.init(x, sldj, reverse = False)
      x = squeeze(x)
    x, sldj = self.flows(x, sldj, reverse)
    if reverse:
      x = squeeze(x, reverse=True)
      x, sldj = self.init(x, sldj, reverse = True)
    return x, sldj


class _Glow(nn.Module):
  """ Recursive constructor for a Glow model. Each call creates a single level.
  Args:
      in_channels (int): Number of channels in the input.
      mid_channels (int): Number of channels in hidden layers of each step.
      num_levels (int): Number of levels to construct. Counter for recursion.
      num_steps (int): Number of steps of flow for each level.
  """
  def __init__(self, in_channels, mid_channels, num_levels, num_steps, params):
    super(_Glow, self).__init__()
    self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                          mid_channels=mid_channels,
                                          params=params)
                                for _ in range(num_steps)])

    if num_levels > 1:
      self.next = _Glow(in_channels=4*in_channels,
                        mid_channels=mid_channels,
                        num_levels=num_levels - 1,
                        num_steps=num_steps,
                        params=params)
    else:
      self.next = None

  def forward(self, x, sldj, reverse=False):
    if not reverse:
      for step in self.steps:
        x, sldj = step(x, sldj, reverse)
      x = squeeze(x)

    if self.next is not None:
      x, sldj = self.next(x, sldj, reverse)

    if reverse:
      x = squeeze(x, reverse)
      for step in reversed(self.steps):
        x, sldj = step(x, sldj, reverse)
    return x, sldj


class _FlowStep(nn.Module):

  def __init__(self, in_channels, mid_channels, params):
      super(_FlowStep, self).__init__()
      # Activation normalization, invertible 1x1 convolution, affine coupling
      self.norm = ActNorm(in_channels, return_ldj=True)
      self.conv = InvConv(in_channels)
      self.coup = Coupling(in_channels // 2, mid_channels, params)

  def forward(self, x, sldj=None, reverse=False):
    if reverse:
      x, sldj = self.coup(x, sldj, reverse)
      x, sldj = self.conv(x, sldj, reverse)
      x, sldj = self.norm(x, sldj, reverse)
    else:
      x, sldj = self.norm(x, sldj, reverse)
      x, sldj = self.conv(x, sldj, reverse)
      x, sldj = self.coup(x, sldj, reverse)
    return x, sldj



class Coupling(nn.Module):
  """Affine coupling layer originally used in Real NVP and described by Glow.
  Note: The official Glow implementation (https://github.com/openai/glow)
  uses a different affine coupling formulation than described in the paper.
  This implementation follows the paper and Real NVP.
  Args:
      in_channels (int): Number of channels of the input.
      mid_channels (int): Number of channels of the intermediate activation
          in NN.
  """
  def __init__(self, in_channels, mid_channels, params):
    super(Coupling, self).__init__()
    self.nn = NN(in_channels, mid_channels, 2 * in_channels, params)
    self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

  def forward(self, x, ldj, reverse=False):
    x_change, x_id = x.chunk(2, dim=1)

    st = self.nn(x_id)
    s, t = st[:, 0::2, ...], st[:, 1::2, ...]
    s = self.scale * torch.tanh(s)

    # Scale and translate
    if reverse:
      x_change = x_change * s.mul(-1).exp() - t
      if ldj is not None:
        ldj = ldj - s.flatten(1).sum(-1)
    else:
      x_change = (x_change + t) * s.exp()
      if ldj is not None:
        ldj = ldj + s.flatten(1).sum(-1)

    x = torch.cat((x_change, x_id), dim=1)
    return x, ldj


class NN(nn.Module):
  """Small convolutional network used to compute scale and translate factors.
  Args:
      in_channels (int): Number of channels in the input.
      mid_channels (int): Number of channels in the hidden activations.
      out_channels (int): Number of channels in the output.
      use_act_norm (bool): Use activation norm rather than batch norm.
  """
  def __init__(self, in_channels, mid_channels, out_channels, params, conv = True):
    super(NN, self).__init__()

    norm_fn = nn.BatchNorm2d if params.batchnorm else ActNorm

    self.in_norm = norm_fn(in_channels)
    self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
    nn.init.normal_(self.in_conv.weight, 0., 0.05)

    self.mid_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
    nn.init.normal_(self.mid_conv1.weight, 0., 0.05)

    self.mid_norm = norm_fn(mid_channels)
    self.mid_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
    nn.init.normal_(self.mid_conv2.weight, 0., 0.05)

    self.out_norm = norm_fn(mid_channels)
    self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
    nn.init.zeros_(self.out_conv.weight)
    nn.init.zeros_(self.out_conv.bias)


  def forward(self, x):
    x = self.in_norm(x)
    x1 = self.in_conv(x)

    x = self.in_conv(x) 
    x = F.relu(x)

    x = self.mid_conv1(x)
    x = self.mid_norm(x)
    x = F.relu(x)

    x = self.mid_conv2(x)
    x = self.out_norm(x)
    x = F.relu(x)
    x = self.out_conv(x)
    return x




class InvConv(nn.Module):
  """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
  (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.
  Args:
      num_channels (int): Number of channels in the input and output.
  """
  def __init__(self, num_channels):
      super(InvConv, self).__init__()
      self.num_channels = num_channels

      # Initialize with a random orthogonal matrix
      w_init = np.random.randn(num_channels, num_channels)
      w_init = np.linalg.qr(w_init)[0].astype(np.float32)
      self.weight = nn.Parameter(torch.from_numpy(w_init))

  def forward(self, x, sldj, reverse=False):
      if sldj is not None:
          ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)
          if reverse:
              sldj = sldj - ldj
          else:
              sldj = sldj + ldj

      if reverse:
          weight = torch.inverse(self.weight.double()).float()
      else:
          weight = self.weight
          
      weight = weight.view(self.num_channels, self.num_channels, 1, 1)
      z = F.conv2d(x, weight)

      return z, sldj


if __name__ == "__main__":
  model = Glow(in_channels=3, num_channels=32, num_levels=3, num_steps=1)
  print(model)
  img = torch.randn(1, 3, 32, 32) * 3.
  img = img.sigmoid()
  z, ldj = model(img)
  rc, _ = model(z, reverse=True)
  torch.testing.assert_allclose(img, rc)
