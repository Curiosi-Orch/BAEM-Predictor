import torch
from torch import nn
import math

# WeightedSquaredErrorLoss weighted squared error loss function
class WeightedSquaredErrorLoss(nn.Module):
  def __init__(self, mean: float, std: float = 5) -> None:
    """
      input: [batch_size, input_length, input_dimension]
      output: weighted squared loss for dimension 1, and get average for dimension 0 and 2
    """
    super(WeightedSquaredErrorLoss, self).__init__()
    self.mean = mean
    self.std = std
    
  def forward(self, input: torch.Tensor, target: torch.Tensor, base: torch.Tensor = None) -> torch.Tensor:
    return self.wse_loss(input, target, base)

  def wse_loss(self, input: torch.Tensor, target: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    if not (target.size() == input.size()):
      torch.warnings.warn(
        "Using a target size ({}) that is different to the input size ({}). "
        "This will likely lead to incorrect results due to broadcasting. "
        "Please ensure they have the same size.".format(target.size(), input.size()),
        stacklevel=2)
    batch_size = input.shape[0]
    input_length = input.shape[1]
    input_dimension = input.shape[2]
    device = input.device
    
    if base is None:
      base = torch.zeros((batch_size,1,input_dimension)).to(device)
    offset = input[:,0:1,:] - base
    input = input - offset
    
    gaussian_kernel = torch.zeros((input_length,1)).to(device)
    for i in range(input_length):
      gaussian_kernel[i,0] = math.exp(-math.pow((i-self.mean),2)/(2*self.std))/(math.sqrt(2*math.pi)*self.std)
    gaussian_kernel = gaussian_kernel/torch.sum(gaussian_kernel)
    
    error = torch.pow(input - target, 2)
    weighted_error = torch.zeros(error.shape)
    for i in range(batch_size):
      weighted_error[i,:,:] = error[i,:,:] * gaussian_kernel
    # weighted_error = error * gaussian_kernel
    
    return torch.sum(weighted_error)/batch_size/input_dimension
    

