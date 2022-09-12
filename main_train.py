import toml
import os
import logging
import joblib
import torch
import torch.nn as nn

from parse_param import *
from predictors.utils import *
from predictors.loops import *

if __name__ == '__main__':
  # load parameters
  project_dir = os.path.dirname(__file__)
  # config_name = "prediction_delay"
  config_name = "prediction_position_active"
  # config_name = "prediction_position_passive" 
  
  params_path = project_dir+"/params/"+config_name+".toml"
  params = toml.load(params_path)
  params["project_dir"] = project_dir
  
  # compute output length, make result directory
  params = ParameterPreparation(params)
  
  # set log configurations
  logging.basicConfig(filename=params["result_path"]+"/debug.log",
                      filemode='a',
                      format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                      datefmt='%H:%M:%S',
                      level=logging.INFO)
  logging.info(f"Loading parameters from {params_path}")
  
  # set batch_size
  params["hyper_parameters"]["learning_rate"] = params["hyper_parameters"]["batch_size"]*params["hyper_parameters"]["lr_batch_rate"]
  
  # loss function
  loss_function = GenerateLossFunction(params)
    
  # scaler
  scaler = GenerateScaler(params)
  
  # model
  params["hyper_parameters"]["num_gpu"] = torch.cuda.device_count()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = GenerateModel(params,loss_function,device)
  model = model.to(device)
  if params["hyper_parameters"]["num_gpu"] > 1:
    model = nn.DataParallel(model, list(range(params["hyper_parameters"]["num_gpu"])))
    
  # optimizer
  optimizer = GenerateOptimizer(params, model)
  
  # scheduler
  scheduler = GenerateScheduler(params, optimizer)
  
  # load data
  train_data_raw,test_data_raw = LoadRawData(params)
  
  # filter data
  train_data_processed = ProcessData(train_data_raw, params)
  test_data_processed = ProcessData(test_data_raw, params)
  
  # normalize/standardize the data
  train_data_scaled = scaler.fit_transform(train_data_processed)
  test_data_scaled = scaler.transform(test_data_processed)
  
  # generate data sequences for training
  if params["data"]["data_type"] == "position_passive":
    train_data = GetTrainData_passive(train_data_scaled,
                                      input_length=params["data"]["input_length"],
                                      input_sample_step=params["data"]["input_sample_step"],
                                      output_length=params["data"]["output_length"],
                                      output_sample_step=params["data"]["output_sample_step"], 
                                      batch_size=params["hyper_parameters"]["batch_size"], 
                                      relative=params["data"]["relative"],
                                      device=device)
    test_data =  GetTrainData_passive(test_data_scaled,
                                      input_length=params["data"]["input_length"],
                                      input_sample_step=params["data"]["input_sample_step"],
                                      output_length=params["data"]["output_length"],
                                      output_sample_step=params["data"]["output_sample_step"], 
                                      batch_size=params["hyper_parameters"]["batch_size"], 
                                      relative=params["data"]["relative"],
                                      device=device)
  else:
    train_data = GetTrainData(train_data_scaled, 
                              target_indices=params["data"]["target_indices"], 
                              input_length=params["data"]["input_length"], 
                              output_length=params["data"]["output_length"], 
                              batch_size=params["hyper_parameters"]["batch_size"], 
                              input_sample_step=params["data"]["input_sample_step"], 
                              output_sample_step=params["data"]["output_sample_step"],
                              relative=params["data"]["relative"],
                              device=device)
    test_data  = GetTrainData(test_data_scaled, 
                              target_indices=params["data"]["target_indices"], 
                              input_length=params["data"]["input_length"], 
                              output_length=params["data"]["output_length"], 
                              batch_size=params["hyper_parameters"]["batch_size"], 
                              input_sample_step=params["data"]["input_sample_step"], 
                              output_sample_step=params["data"]["output_sample_step"],
                              relative=params["data"]["relative"],
                              device=device)
  
  if params["data"]["train_data_file"] == params["data"]["test_data_file"]:
    train_ratio = params["hyper_parameters"]["train_ratio"]
    data_train = train_data[0:math.floor(len(train_data)*train_ratio)]
    data_test = train_data[math.floor(len(train_data)*train_ratio):]
  else:
    data_train = train_data
    data_test = test_data
  
  # save scaler
  joblib.dump(scaler, params["result_path"]+'/scaler') 
  
  # # illustration
  # input_length = params["data"]["input_length"]
  # input_sample_step = params["data"]["input_sample_step"]
  # output_length = params["data"]["output_length"]
  # output_sample_step = params["data"]["output_sample_step"]
  # time = np.arange(0,(input_length-1)*input_sample_step+1,1)
  # n = 0
  # batch = 0
  # axis = 0
  # pyplot.plot(time[0:len(time):input_sample_step],data_test[n][0][batch,:,axis],'ro-')
  # pyplot.plot(time[len(time)-1-(output_length-1)*output_sample_step:len(time):output_sample_step],data_test[n][1][batch,:,axis],'bo-')
  # pyplot.show()
  
  # training
  train(model, optimizer, scheduler, data_train, data_test, params)
  