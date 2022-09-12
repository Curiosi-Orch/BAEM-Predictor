import torch
import torch.nn as nn
from predictors.Embedding.position_encoder import PositionalEncoder
        
class TransformerBlock(nn.Module):
  """
    input: [batch_size, input_length, input_dimension]
    output: [batch_size, output_length, output_dimension]
  """
  def __init__(self, input_dimension: int,
               output_dimension: int,
               input_length: int,
               output_length: int,
               num_heads: int, 
               num_transformer_layers: int,
               feedforward_dimension: int = 256, 
               dropout: float = 0.0, 
               layer_norm_eps: float = 1e-5) -> torch.Tensor:
    super(TransformerBlock, self).__init__()
    # activation
    self.activation = nn.ReLU()
    self.src_mask = None
    self.num_transformer_layers = num_transformer_layers
    
    # positional encoding
    self.position_encoder = PositionalEncoder(input_dimension, dropout)

    # transformer
    if num_transformer_layers != 0:
      encoder_layer = nn.TransformerEncoderLayer(input_dimension, 
                                                num_heads, feedforward_dimension, 
                                                dropout, layer_norm_eps=layer_norm_eps,
                                                batch_first=True, norm_first=False)
      encoder_norm = nn.LayerNorm(input_dimension, eps=layer_norm_eps)
      self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers, encoder_norm)
    
    # output
    self.output_function_1 = nn.Linear(input_dimension, output_dimension)
    self.output_function_2 = nn.Linear(input_length, output_length)
    
    # reset parameters
    self._reset_parameters()

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
      
  def _reset_parameters(self) -> None:
    """Initiate parameters in the model."""
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
        
  def forward(self, input: torch.Tensor) -> torch.Tensor:
    # initialization of src mask
    if self.src_mask is None or self.src_mask.size(0) != len(input):
      mask = self._generate_square_subsequent_mask(input.size(1)).to(input.device)
      self.src_mask = mask
      
    # Transformer Encoder block
    encode_in = self.activation(input)
    encode_out = self.position_encoder(encode_in)
    if self.num_transformer_layers != 0:
      transformer_out = self.transformer(encode_out, self.src_mask)
    else: 
      transformer_out = encode_out
    
    # Output
    output = transformer_out
    output = self.output_function_1(output)
    output = output.permute(0,2,1)
    output = self.output_function_2(output)
    output = output.permute(0,2,1)
    return output
