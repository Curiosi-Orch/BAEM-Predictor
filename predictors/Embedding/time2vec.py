import torch.nn as nn
import torch

class Time2Vec(nn.Module):
    def __init__(self, input_dimension: int, 
                 output_dimension: int,
                 periodic_activation_function=torch.sin,
                 dropout: float = 0.0):
      super().__init__()
      self.dropout = nn.Dropout(p=dropout)
      
      # Number of features in the time embedding
      self.periodic_activation_function = periodic_activation_function
      self.w0 = nn.parameter.Parameter(torch.empty(input_dimension, 1))
      self.b0 = nn.parameter.Parameter(torch.empty(1))
      self.w = nn.parameter.Parameter(torch.empty(input_dimension, output_dimension-1))
      self.b = nn.parameter.Parameter(torch.empty(output_dimension-1))

      # Initialization phase and frequency
      bound = 1
      nn.init.uniform_(self.w0, -bound, bound)
      nn.init.uniform_(self.b0, -bound, bound)
      nn.init.uniform_(self.w, -bound, bound)
      nn.init.uniform_(self.b, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
      v1 = self.periodic_activation_function(torch.matmul(input, self.w) + self.b)
      v2 = torch.matmul(input, self.w0) + self.b0
      output = self.dropout(torch.cat([v1,v2], axis=-1))
      return output