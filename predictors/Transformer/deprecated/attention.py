import torch
import torch.nn as nn

""" attention
  formulus to culculate attention value:
    Attention(query,key,value) = softmax(query*(key^T) / sqrt(dimension)) * value
    * sqrt(d) is used to adjust the scale of the product
    query dimesion:  [batch_size, sequence_length, attention_length]
    key dimension:   [batch_size, sequence_length, attention_length]
    value dimension: [batch_size, sequence_length, value_length]
    
    attention dimension: [batch_size, value_length, attention_length]
"""
def attention(query, key, value, mask=None, dropout=None):
  dimension = torch.tensor(query.shape[-1]).float()
  scores = torch.matmul(query, key.transpose(-2,-1)) / torch.sqrt(dimension)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  p_attention = torch.softmax(scores, dim = -1) # (batch_size, dim_attn, seq_length)
  if dropout is not None:
    p_attention = dropout(p_attention)
  attention = torch.matmul(p_attention, value) #(batch_size, seq_length, seq_length)
  return attention

# 
class AttentionBlock(torch.nn.Module):
  def __init__(self, dim_value, dim_attention):
    super(AttentionBlock, self).__init__()
    self.value = Value(dim_value, dim_value)
    self.key = Key(dim_value, dim_attention)
    self.query = Query(dim_value, dim_attention)
    
  def forward(self, x, kv = None):
    if(kv is None):
      # Attention with x connected to Q,K and V (For encoder)
      return attention(self.query(x), self.key(x), self.value(x))
    # Attention with x as Q, external vector kv as K an V (For decoder)
    return attention(self.query(x), self.key(kv), self.value(kv))

class MultiHeadAttentionBlock(torch.nn.Module):
  def __init__(self, dim_value, dim_attention, num_heads):
    super(MultiHeadAttentionBlock, self).__init__()
    self.heads = []
    for i in range(num_heads):
      self.heads.append(AttentionBlock(dim_value, dim_attention))
    self.heads = nn.ModuleList(self.heads)
    self.func = nn.Linear(num_heads * dim_value, dim_value, bias = False)
             
  def forward(self, x, kv = None):
    a = []
    for h in self.heads:
      a.append(h(x, kv = kv))
    a = torch.stack(a, dim = -1) #combine heads
    a = a.flatten(start_dim = 2) #flatten all head outputs
    x = self.func(a)
    return x

class Value(torch.nn.Module):
  def __init__(self, dim_input, dim_value):
    super(Value, self).__init__()
    self.dim_val = dim_value
    self.func1 = nn.Linear(dim_input, dim_value, bias = False)
  
  def forward(self, x):
    x = self.func1(x)
    return x


class Key(torch.nn.Module):
  def __init__(self, dim_input, dim_attention):
    super(Key, self).__init__()
    self.dim_attn = dim_attention
    self.func1 = nn.Linear(dim_input, dim_attention, bias = False)
  
  def forward(self, x):
    x = self.func1(x)
    return x

class Query(torch.nn.Module):
  def __init__(self, dim_input, dim_attention):
    super(Query, self).__init__()
    self.dim_attn = dim_attention
    self.func1 = nn.Linear(dim_input, dim_attention, bias = False)
  
  def forward(self, x):
    x = self.func1(x)
    return x