import torch.nn.functional as F
from attention import *
import math


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoder(nn.Module):
  """ PositionalEncoder
    give position information of input data to embed sequence order 
  Args:
      d_model (_type_): _description_
  """
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoder, self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    """
      Args:
        x: Tensor, shape [seq_len, batch_size, embedding_dim] ???
    """
    x = x + self.pe[:x.size(1), :]. squeeze(1)
    x = self.dropout(x)
    return x


class TransformerEncoderLayer(torch.nn.Module):
  def __init__(self, dim_value, dim_attention, num_heads=1):
    super(TransformerEncoderLayer, self).__init__()
    self.attention = MultiHeadAttentionBlock(dim_value, dim_attention, num_heads)
    self.func1 = nn.Linear(dim_value, dim_value)
    self.func2 = nn.Linear(dim_value, dim_value)
    self.norm1 = nn.LayerNorm(dim_value)
    self.norm2 = nn.LayerNorm(dim_value)

  def forward(self, x):
    a = self.attention(x)
    x = self.norm1(x + a)
    a = self.func1(F.elu(self.func2(x)))
    x = self.norm2(x + a)
    return x

class TransformerDecoderLayer(torch.nn.Module):
  def __init__(self, dim_value, dim_attention, num_heads=1):
    super(TransformerDecoderLayer, self).__init__()
    self.attention1 = MultiHeadAttentionBlock(dim_value, dim_attention, num_heads)
    self.attention2 = MultiHeadAttentionBlock(dim_value, dim_attention, num_heads)
    self.func1 = nn.Linear(dim_value, dim_value)
    self.func2 = nn.Linear(dim_value, dim_value)
    self.norm1 = nn.LayerNorm(dim_value)
    self.norm2 = nn.LayerNorm(dim_value)
    self.norm3 = nn.LayerNorm(dim_value)

  def forward(self, x, enc):
    a = self.attention1(x)
    x = self.norm1(a + x)
    a = self.attention2(x, kv=enc)
    x = self.norm2(a + x)
    a = self.func1(F.elu(self.func2(x)))
    x = self.norm3(x + a)
    return x


class TransformerBlock(torch.nn.Module):
  def __init__(self, dim_value, dim_attention, dim_input, dim_output,
               decoder_sequence_length, out_sequence_length, 
               num_decoder_layers, num_encoder_layers, 
               num_heads, loss_function, device):
    super(TransformerBlock, self).__init__()
    self.decoder_sequence_length = decoder_sequence_length
    # Initiate encoder and Decoder layers
    self.encoders = nn.ModuleList()
    for i in range(num_encoder_layers):
      self.encoders.append(TransformerEncoderLayer(dim_value, dim_attention, num_heads))
    self.decoders = nn.ModuleList()
    for i in range(num_decoder_layers):
      self.decoders.append(TransformerDecoderLayer(dim_value, dim_attention, num_heads))
    self.position_encoder = PositionalEncoder(dim_value)
    # Dense layers for managing network inputtgts and outputs
    self.encoder_input_func = nn.Linear(dim_input, dim_value)
    self.decoder_input_func = nn.Linear(dim_input, dim_value)
    self.out_func = nn.Linear(decoder_sequence_length * dim_value, 
                              out_sequence_length)
    loss_function.to(device)
    self.device = device
    self.loss_function = loss_function
    self.dim_output = dim_output
    self.out_sequence_length = out_sequence_length
    
  def forward(self, x, label=None):
    # encoder
    encode = self.encoder_input_func(x)
    encode =  self.position_encoder(encode)
    # encode = self.encoders[0](encode)
    for encoder in self.encoders:
      encode = encoder(encode)
    # decoder
    decode = self.decoder_input_func(x[:, -self.decoder_sequence_length:])
    # decode = self.decoders[0](decode, encode)
    for decoder in self.decoders:
      decode = decoder(decode, encode)
    # output
    # outdata = self.out_func(d)
    outdata = self.out_func(decode.flatten(start_dim=1))
    outdata = outdata.view((-1,self.out_sequence_length,self.dim_output))
    loss = 0
    if label is not None:
      loss = self.loss_function(outdata, label)
    return outdata, loss