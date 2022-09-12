import torch
import torch.nn as nn
import math

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoder(nn.Module):
  """ PositionalEncoder
    give position information of input data to embed sequence order 
  """
  def __init__(self, encoder_dimension: int, 
               dropout: float = 1.0, 
               max_len:int = 5000) -> None:
    super(PositionalEncoder, self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    position_encoder = torch.zeros(max_len, encoder_dimension)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, encoder_dimension, 2).float() 
               * (-math.log(10000.0) / encoder_dimension))
    position_encoder[:, 0::2] = torch.sin(position * div_term)
    position_encoder[:, 1::2] = torch.cos(position * div_term)
    position_encoder = position_encoder.unsqueeze(0).transpose(0, 1)
    self.register_buffer('position_encoder', position_encoder)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    """
      input: Tensor, shape [batch_size, input_length, input_dimension]
    """
    input = input + self.position_encoder[:input.size(1), :]. squeeze(1)
    input = self.dropout(input)
    return input
  
