import torch
import torch.nn as nn
from predictors.LSTM.models import LSTMBlock
from predictors.Transformer.models import TransformerBlock
from predictors.CNN.models import CNNBlock
from predictors.Embedding.time2vec import Time2Vec


class PredictorEncoder(nn.Module):
  def __init__(self, input_dimension: int,
               output_dimension: int,
               lstm_hidden_dimension: int,
               transformer_feedforward_dimension: int,
               input_length: int,
               output_length: int,
               num_transformer_heads: int,
               num_transformer_layers: int,
               num_lstm_layers: int,
               conv_kernels: list = [3,4,5,6],
               lstm_bidirectional: bool = True,
               dropout: int = 0.0) -> None:
    super(PredictorEncoder, self).__init__()
    # tranformer
    self.transformer = TransformerBlock(input_dimension = input_dimension,
                                        output_dimension = output_dimension,
                                        input_length = input_length,
                                        output_length = output_length,
                                        num_heads = num_transformer_heads, 
                                        num_transformer_layers = num_transformer_layers, 
                                        feedforward_dimension = transformer_feedforward_dimension,
                                        dropout = dropout)
    # lstm
    self.lstm = LSTMBlock(input_dimension = input_dimension, 
                          output_dimension = output_dimension, 
                          hidden_dimension = lstm_hidden_dimension,
                          input_length = input_length,
                          output_length = output_length,
                          num_lstm_layers = num_lstm_layers, 
                          bidirectional = lstm_bidirectional,
                          dropout = dropout)
    # cnn
    self.cnn = CNNBlock(input_length = input_length, 
                        output_length = output_length,
                        input_dimension = input_dimension,
                        output_dimension = output_dimension,
                        conv_kernels = conv_kernels,
                        dropout = dropout)
    # add & norm
    self.norm = nn.LayerNorm(output_dimension)
    
  def forward(self, input: torch.Tensor) -> torch.Tensor:
    # Transformer encoder block
    transformer_out = self.transformer(input)
    
    # LSTM encoder block
    lstm_out = self.lstm(input)  
    
    # CNN encoder block
    cnn_out = self.cnn(input)
    
    # Add & Norm block
    output = self.norm(transformer_out+lstm_out+cnn_out)
    return output
    

class PredictorDecoder(nn.Module):
  def __init__(self, input_dimension: int,
               output_dimension: int,
               input_length: int,
               output_length: int,
               dropout: int = 0.0) -> None:
    super(PredictorDecoder, self).__init__()
    self.activation_function = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
    self.output_function_1 = nn.Linear(input_dimension, output_dimension)
    self.output_function_2 = nn.Linear(input_length, output_length)
  
  def forward(self, input: torch.Tensor) -> torch.Tensor:
    decode_in = self.activation_function(input)
    decode_in = self.dropout(decode_in)
    # Output
    output = self.output_function_1(decode_in)
    output = output.permute(0,2,1)
    output = self.output_function_2(output)
    output = output.permute(0,2,1)
    return output


class Predictor(nn.Module):
  def __init__(self, num_encoder_lstm_layers: int,
               num_encoder_transformer_layers: int,
               num_encoder_transformer_heads: int,
               input_dimension: int,
               output_dimension: int,
               encoder_input_dimension: int,
               encoder_output_dimension: int,
               latent_dimension: int,
               encoder_lstm_hidden_dimension: int,
               encoder_transformer_feedforward_dimension: int,
               input_length: int,
               output_length: int,
               encoder_output_length: int,
               use_VAE: bool,
               encoder_conv_kernels: list = [3,4,5,6],
               encoder_lstm_bidirectional: bool = True,
               dropout: int = 0.0,
               loss_calibration: bool = False,
               loss_function = None, 
               device = None) -> None:
    super(Predictor, self).__init__()
    loss_function.to(device)
    self.loss_function = loss_function
    self.device = device
    self.use_VAE = use_VAE
    self.loss_calibration = loss_calibration
    '''
      [batch_size, input_length, input_dimension] 
               => [input_length, encoder_input_dimension]
    '''
    self.input_embedding_function = Time2Vec(input_dimension = input_dimension,
                                             output_dimension = encoder_input_dimension,
                                             dropout = dropout)
    
    ''' encoder
      [batch_size, input_length, encoder_input_dimension] 
               => [encoder_output_length, encoder_output_dimension]
    '''
    self.encoder = PredictorEncoder(input_dimension = encoder_input_dimension, 
                                    output_dimension = encoder_output_dimension,
                                    lstm_hidden_dimension = encoder_lstm_hidden_dimension, 
                                    transformer_feedforward_dimension = encoder_transformer_feedforward_dimension,
                                    input_length = input_length, 
                                    output_length = encoder_output_length, 
                                    num_transformer_heads = num_encoder_transformer_heads, 
                                    num_transformer_layers = num_encoder_transformer_layers, 
                                    num_lstm_layers = num_encoder_lstm_layers, 
                                    conv_kernels = encoder_conv_kernels, 
                                    lstm_bidirectional = encoder_lstm_bidirectional,
                                    dropout = dropout)
    ''' reparameterize: mu, logvar
      [batch_size, encoder_output_length, encoder_output_dimension] 
               => [encoder_output_length, latent_dimension]
    '''
    if (use_VAE):
      self.mu_function = nn.Linear(encoder_output_dimension, latent_dimension)
      self.logvar_function = nn.Linear(encoder_output_dimension, latent_dimension)
    else:
      self.link_function = nn.Linear(encoder_output_dimension, latent_dimension)
    
    ''' decoder
      [batch_size, encoder_output_length, latent_dimension] 
               => [output_length, output_dimension]
    '''
    self.decoder = PredictorDecoder(input_dimension = latent_dimension,
                                    output_dimension = output_dimension, 
                                    input_length = encoder_output_length,
                                    output_length = output_length,
                                    dropout = dropout)
      
  def reparameterize(self, encode_out: torch.Tensor) -> torch.Tensor:
    """
      mu: (Tensor) Mean of the latent Gaussian
      logvar: (Tensor) Standard deviation of the latent Gaussian
    """
    mu = self.mu_function(encode_out)
    logvar = self.logvar_function(encode_out)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu
  
  def forward(self, input: torch.Tensor, label: torch.Tensor = None):
    """
        input (Tensor): [batch_size, input_length, input_dimension]
        label (Tensor, optional): [batch_size, output_length, output_dimension], 
                                  if is not None, training mode, calculate loss,
                                  if is None, testing mode.
    """
    # Input block
    encode_in = self.input_embedding_function(input)
    # Encoder block
    encode_out = self.encoder(encode_in)
    # Link block
    if self.use_VAE:
      link_out = self.reparameterize(encode_out)
    else:
      link_out = self.link_function(encode_out)
    # Decoder block
    output = self.decoder(link_out)
    
    # Calculate loss
    loss = 0
    if label is not None:
      if self.loss_calibration:
        loss = self.loss_function(output, label, input[:,-1:,:])
      else:
        loss = self.loss_function(output, label)
      
    return output, loss