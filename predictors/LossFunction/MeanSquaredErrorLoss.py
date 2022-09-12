import torch
from torch import nn

# MeanSquaredErrorLoss wrapper of MSE Loss
class MeanSquaredErrorLoss(nn.Module):
  def __init__(self) -> None:
    """
      input: [batch_size, input_length, input_dimension]
      output: mse loss
    """
    super(MeanSquaredErrorLoss, self).__init__()
    self.mse = nn.MSELoss()
    
  def forward(self, input: torch.Tensor, target: torch.Tensor, base: torch.Tensor = None) -> torch.Tensor:
    return self.mse_loss(input, target, base)

  def mse_loss(self, input: torch.Tensor, 
                  target: torch.Tensor,
                  base: torch.Tensor) -> torch.Tensor:
    if not (target.size() == input.size()):
      torch.warnings.warn(
        "Using a target size ({}) that is different to the input size ({}). "
        "This will likely lead to incorrect results due to broadcasting. "
        "Please ensure they have the same size.".format(target.size(), input.size()),
        stacklevel=2)
    batch_size, input_length, input_dimension  = input.shape
    device = input.device
    
    if base is None:
      base = torch.zeros((batch_size,1,input_dimension)).to(device)
      offset = input[:,0:1,:] - base
      input = input - offset

    loss = self.mse(input, target)
    return loss
    

