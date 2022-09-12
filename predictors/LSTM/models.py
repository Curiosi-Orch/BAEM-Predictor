import torch
from torch import nn
import math


class LSTMBlock(nn.Module):
  """
    input: [batch_size, input_length, input_dimension]
    output: [batch_size, output_length, output_dimension]
  """
  def __init__(self, input_dimension: int, 
               output_dimension: int, 
               hidden_dimension: int, 
               input_length: int,
               output_length: int,
               num_lstm_layers: int, 
               bidirectional: bool,
               dropout: int = 0.0) -> torch.Tensor:
    super(LSTMBlock, self).__init__()
    self.hidden_dimension = hidden_dimension
    self.bidirectional = bidirectional
    self.num_lstm_layers = num_lstm_layers
    self.dropout = nn.Dropout(dropout)
    
    # activation
    self.activation = nn.ReLU()
    # lstm
    if self.num_lstm_layers != 0:
      self.lstm = nn.LSTM(input_dimension, self.hidden_dimension, num_layers=self.num_lstm_layers,
                          batch_first=True, bidirectional=self.bidirectional) 
    # output
    if self.num_lstm_layers != 0:
      if self.bidirectional:
        self.output_function_1 = nn.Linear(self.hidden_dimension*2, output_dimension)
      else:
        self.output_function_1 = nn.Linear(hidden_dimension, output_dimension)
    else:
      self.output_function_1 = nn.Linear(input_dimension, output_dimension)
    self.output_function_2 = nn.Linear(input_length, output_length)
    
    self._reset_parameters()

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    input = self.activation(input)
    input = self.dropout(input) 
    
    if self.num_lstm_layers != 0:
      # init hidden layers of LSTM
      batch_size = input.size(0)
      device = input.device
      hidden_layers = self._init_hidden_layers(batch_size, device)
      # LSTM
      lstm_out, _ = self.lstm(input, hidden_layers)
    else:
      lstm_out = input
    
    # output
    output = lstm_out
    output = self.output_function_1(output)
    output = output.permute(0,2,1)
    output = self.output_function_2(output)
    output = output.permute(0,2,1)
    return output
  
  def _reset_parameters(self) -> None:
    """Initiate parameters in the model."""
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
  
  def _init_hidden_layers(self, batch_size, device):
    if self.bidirectional:
      hidden_layers = (torch.randn(self.num_lstm_layers*2, batch_size, self.hidden_dimension).to(device),  # hidden
                      torch.randn(self.num_lstm_layers*2, batch_size, self.hidden_dimension).to(device))  # cell
    else:
      hidden_layers = (torch.randn(self.num_lstm_layers, batch_size, self.hidden_dimension).to(device),  # hidden
                      torch.randn(self.num_lstm_layers, batch_size, self.hidden_dimension).to(device))  # cell
    return hidden_layers
  
 