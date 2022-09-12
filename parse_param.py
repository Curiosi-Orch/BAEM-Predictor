import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from predictors.LSTM.models import *
from pandas import read_csv
from predictors.utils import *
from predictors.models import Predictor
import os
import math
import datetime
import toml
from predictors.LossFunction.WeightedSquaredErrorLoss import WeightedSquaredErrorLoss
from predictors.LossFunction.ShapeTimeDistortionLoss.ShapeTimeDistortionLoss import ShapeTimeDistortionLoss
from predictors.LossFunction.MeanSquaredErrorLoss import MeanSquaredErrorLoss
from predictors.LossFunction.CrossEntropyLoss import CrossEntropyLoss
from predictors.Scaler.NoneScaler import NoneScaler

def ParameterPreparation(params):
  params["data"]["train_data_file"] = params["project_dir"] + "/data/" \
                                    + params["data"]["train_data_file"]
  params["data"]["test_data_file"]  = params["project_dir"] + "/data/" \
                                    + params["data"]["test_data_file"]    
  date_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  params["result_path"] = params["project_dir"] \
                        + "/results/"+date_time \
                        + "_"+params["data"]["data_type"]
  if not os.path.exists(params["result_path"]):
    os.makedirs(params["result_path"])
  return params


""" GenerateOptimizer
  generate optimizer according to parameters
"""
def GenerateOptimizer(params, model):
  if params["optimizer"]["optimizer_type"] == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=params["hyper_parameters"]["learning_rate"])
  elif params["optimizer"]["optimizer_type"] == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=params["hyper_parameters"]["learning_rate"], 
                          momentum=0.9, dampening=0.5)
  return optimizer

""" GenerateScheduler
  generate scheduler according to parameters
"""
def GenerateScheduler(params, optimizer):
  if params["optimizer"]["scheduler_type"] == "ReduceLROnPlateau":
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode=params["optimizer"]["scheduler_mode"],
                                  factor=params["optimizer"]["scheduler_factor"], 
                                  patience=params["optimizer"]["scheduler_patience"],
                                  verbose=params["optimizer"]["scheduler_verbose"], 
                                  threshold=params["optimizer"]["scheduler_threshold"], 
                                  threshold_mode=params["optimizer"]["scheduler_threshold_mode"],
                                  cooldown=params["optimizer"]["scheduler_cooldown"], 
                                  min_lr=params["optimizer"]["scheduler_min_lr"],
                                  eps=params["optimizer"]["scheduler_eps"])
  return scheduler

""" GenerateLossFunction
  generate loss function according to parameters
"""
def GenerateLossFunction(params):
  if params["model"]["loss_function_type"] == "MeanSquaredError":
    loss_function = MeanSquaredErrorLoss()
  elif params["model"]["loss_function_type"] == "CrossEntropy":
    loss_function = CrossEntropyLoss()
  elif params["model"]["loss_function_type"] == "WeightedSquaredError":
    mean_length = params["data"]["output_length"]/2
    loss_function = WeightedSquaredErrorLoss(mean_length)
  elif params["model"]["loss_function_type"] == "ShapeTimeDistortion":
    loss_function = ShapeTimeDistortionLoss(alpha=params["model"]["shape_distortion_alpha"])
  return loss_function

""" GenerateScaler
  generate scaler according to parameters
"""
def GenerateScaler(params):
  if params["data"]["scaler_type"] == "MaxMin":
    scaler = MinMaxScaler(feature_range=(0,1))
  elif params["data"]["scaler_type"] == "Standard":
    scaler = StandardScaler()
  elif params["data"]["scaler_type"] == "None":
    scaler = NoneScaler()
  return scaler

""" GenerateModel
  generate model according to parameters
"""
def GenerateModel(params, loss_function, device):
  model = Predictor(num_encoder_transformer_layers = params["model"]["encoder"]["num_transformer_layers"],
                    num_encoder_lstm_layers = params["model"]["encoder"]["num_lstm_layers"],
                    num_encoder_transformer_heads = params["model"]["encoder"]["num_transformer_heads"],
                    input_length = params["data"]["input_length"],
                    output_length = params["data"]["output_length"],
                    input_dimension = params["data"]["input_dimension"],
                    output_dimension = params["data"]["output_dimension"],
                    encoder_input_dimension = params["model"]["encoder"]["input_dimension"],
                    encoder_output_dimension = params["model"]["encoder"]["output_dimension"],
                    latent_dimension = params["model"]["latent_dimension"],
                    encoder_transformer_feedforward_dimension = params["model"]["encoder"]["transformer_feedforward_dimension"],
                    encoder_lstm_hidden_dimension = params["model"]["encoder"]["lstm_hidden_dimension"],
                    encoder_lstm_bidirectional = params["model"]["encoder"]["lstm_bidirectional"],
                    encoder_conv_kernels = params["model"]["encoder"]["conv_kernels"],
                    encoder_output_length = params["model"]["encoder"]["output_length"],
                    use_VAE = params["model"]["use_VAE"], 
                    dropout = params["model"]["dropout"],
                    loss_calibration = params["model"]["loss_calibration"],
                    loss_function = loss_function,
                    device = device)
  return model

""" LoadRawData
  load raw data according to parameters
"""
def LoadRawData(params):
  data_type = params["data"]["data_type"]
  train_data_file_path = params["data"]["train_data_file"]
  test_data_file_path = params["data"]["test_data_file"]
  train_dataset = read_csv(train_data_file_path, header=0, parse_dates=False, index_col=False, dtype='float32')
  test_dataset = read_csv(test_data_file_path, header=0, parse_dates=False, index_col=False, dtype='float32')
  
  train_data_raw = []
  test_data_raw = []
  if data_type == "delay":
    train_data_raw.append(train_dataset["tau"].to_numpy())
    test_data_raw.append(test_dataset["tau"].to_numpy())
  elif data_type == "joint":
    train_data_raw.append(train_dataset["q_0"].to_numpy())
    train_data_raw.append(train_dataset["q_1"].to_numpy())
    train_data_raw.append(train_dataset["q_2"].to_numpy())
    train_data_raw.append(train_dataset["q_3"].to_numpy())
    train_data_raw.append(train_dataset["q_4"].to_numpy())
    train_data_raw.append(train_dataset["q_5"].to_numpy())
    train_data_raw.append(train_dataset["q_6"].to_numpy())
    test_data_raw.append(test_dataset["q_0"].to_numpy())
    test_data_raw.append(test_dataset["q_1"].to_numpy())
    test_data_raw.append(test_dataset["q_2"].to_numpy())
    test_data_raw.append(test_dataset["q_3"].to_numpy())
    test_data_raw.append(test_dataset["q_4"].to_numpy())
    test_data_raw.append(test_dataset["q_5"].to_numpy())
    test_data_raw.append(test_dataset["q_6"].to_numpy())
  elif data_type == "position_active":
    train_data_raw.append(train_dataset["x"].to_numpy())
    train_data_raw.append(train_dataset["y"].to_numpy())
    train_data_raw.append(train_dataset["z"].to_numpy())
    test_data_raw.append(test_dataset["x"].to_numpy())
    test_data_raw.append(test_dataset["y"].to_numpy())
    test_data_raw.append(test_dataset["z"].to_numpy())
  elif data_type == "position_passive":
    train_data_raw.append(train_dataset["x_d"].to_numpy())
    train_data_raw.append(train_dataset["y_d"].to_numpy())
    train_data_raw.append(train_dataset["z_d"].to_numpy())
    train_data_raw.append(train_dataset["x"].to_numpy())
    train_data_raw.append(train_dataset["y"].to_numpy())
    train_data_raw.append(train_dataset["z"].to_numpy())
    test_data_raw.append(test_dataset["x_d"].to_numpy())
    test_data_raw.append(test_dataset["y_d"].to_numpy())
    test_data_raw.append(test_dataset["z_d"].to_numpy())
    test_data_raw.append(test_dataset["x"].to_numpy())
    test_data_raw.append(test_dataset["y"].to_numpy())
    test_data_raw.append(test_dataset["z"].to_numpy())
  return train_data_raw, test_data_raw

""" ProcessData
  filter data and transform to 2D array
"""
def ProcessData(data_raw,params):
  if params["data"]["filter_type"] == "None":
    data_filtered = data_raw
  elif params["data"]["filter_type"] == "Average":
    data_filtered = AverageFilter(data_raw, kernel_size = params["data"]["average_kernel_size"])
  elif params["data"]["filter_type"] == "Lowpass":
    data_filtered = ButterLowpassFilter(data_raw, 
                                        frequency_cutoff = params["data"]["frequency_lowpass"],
                                        frequency_sample = params["data"]["frequency_sample"])
  # print(data_raw)
  # from 1D-vector list to 2D-array
  data_processed = np.array([])
  for d in data_filtered:
    if data_processed.size == 0:
      data_processed = d.reshape((-1,1))
    else:
      data_processed = np.concatenate((data_processed,d.reshape((-1,1))),axis=1)
  return data_processed
