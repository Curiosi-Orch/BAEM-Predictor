import torch
from torch import nn
from predictors.LossFunction.ShapeTimeDistortionLoss.path_soft_dtw import PathDTW
from predictors.LossFunction.ShapeTimeDistortionLoss.soft_dtw import SoftDTW 

def pairwise_distances(x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
  '''
    Input: x [input_length, input_dimension]
           y [input_length, input_dimension]
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = sqrt(||x[i,:]-y[j,:]||^2)
  '''
  if y is None: y = x
  x_norm = (x**2).sum(1).view(-1, 1)
  y_t = torch.transpose(y, 0, 1)
  y_norm = (y**2).sum(1).view(1, -1)
  dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
  return torch.clamp(dist, 0.0, float('inf'))



# ShapeTimeDistortionLoss Shape and Time Distortion Loss
class ShapeTimeDistortionLoss(nn.Module):
  def __init__(self, alpha:float = 0.5, 
               gamma:float = 0.001) -> None:
    """
      input: [batch_size, input_length, input_dimension]
      output: shape and temperal distortion, and get average for each batch
    """
    super(ShapeTimeDistortionLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    
  def forward(self, input: torch.Tensor, target: torch.Tensor, base: torch.Tensor = None) -> torch.Tensor:
    return self.dilate_loss(input, target, base)

  def dilate_loss(self, input: torch.Tensor, 
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
    
    D = torch.zeros((batch_size,input_length,input_length)).to(device)
    for k in range(batch_size):
      D[k,:,:] = pairwise_distances(target[k,:,:],input[k,:,:])
    loss_shape = SoftDTW.apply(D,self.gamma)
    
    path = PathDTW.apply(D,self.gamma)           
    Omega = pairwise_distances(torch.arange(input_length).view(-1,1)).to(device)
    loss_temporal =  torch.sum(path*Omega) / (input_length*input_length) 
    
    loss = self.alpha*loss_shape+(1-self.alpha)*loss_temporal
    return loss
    

