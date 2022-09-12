import torch
from torch import nn
import math


class CNNBlock(nn.Module):
  """
    input: [batch_size, input_length, input_dimension]
    output: [batch_size, output_length, output_dimension]
  """
  def __init__(self, input_length: int,
               output_length: int, 
               input_dimension: int,
               output_dimension: int,
               conv_kernels: list = [3,4,5,6],
               dropout: int = 0.0) -> None:
    super(CNNBlock, self).__init__()
    # covolution
    # input = [batch_size, input_dimension, input_length]
    # conv1d -> [batch_size, output_dimension, input_length-kernel_size+1]
    # maxpool1d -> [batch_size, output_dimension, input_length-kernel_size+1-(input_length-kernel_size+1)+1]
    if len(conv_kernels) != 0:
      self.use_conv = True
      self.conv_layers = nn.ModuleList(
        [nn.Sequential(nn.Conv1d(in_channels=input_dimension, 
                                out_channels=output_dimension, 
                                kernel_size=kernel_size,
                                padding = 'valid'),
                      nn.BatchNorm1d(num_features=output_dimension), 
                      nn.ReLU(),
                      nn.MaxPool1d(kernel_size=input_length-kernel_size+1))
        for kernel_size in conv_kernels])
      self.output_function_1 = nn.Linear(1,output_length)
      self.output_function_2 = nn.Linear(len(conv_kernels)*output_dimension,output_dimension)   
    else:
      self.use_conv = False
      self.output_function_1 = nn.Linear(input_length,output_length)
      self.output_function_2 = nn.Linear(input_dimension,output_dimension) 
                  
    self.dropout = nn.Dropout(dropout)
    # reset parameters
    self._reset_parameters()
    
  def _reset_parameters(self) -> None:
    """Initiate parameters in the model."""
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    conv_in = input.permute(0,2,1)
    if self.use_conv:
      # convolution
      conv_out = [conv(conv_in) for conv in self.conv_layers]
      conv_out = torch.cat(conv_out, dim=-2)
    else:
      conv_out = conv_in
    # output
    output = self.dropout(conv_out)
    output = self.output_function_1(output)
    output = output.permute(0,2,1)
    output = self.output_function_2(output)
    return output
  
  