import toml
import joblib
from parse_param import *
import numpy as np

def get_prediction_model(model_dir_path):
  params_path = model_dir_path+"/params.toml"
  params = toml.load(params_path)
  
  # loss function
  loss_function = GenerateLossFunction(params)

  # load scaler
  scaler = GenerateScaler(params)
  scaler = joblib.load(model_dir_path+'/scaler')

  # load model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = GenerateModel(params,loss_function,device)
  if params["hyper_parameters"]["num_gpu"] > 1:
    model = nn.DataParallel(model)
  model.load_state_dict(torch.load(model_dir_path+'/model'))
  model = model.to(device)
  model.eval()
  
  return model, scaler, params, device


def predict(model, scaler, params, input_data, device):
  # process data
  raw_data = []
  for in_data in input_data:
    raw_data.append(np.array(in_data))
  output_dim = params["data"]["output_dimension"]
  output_length = params["data"]["output_length"]
  input_length = params["data"]["input_length"]
  input_sample_step = params["data"]["input_sample_step"]
  output_sample_step = params["data"]["output_sample_step"]
  
  processed_data = ProcessData(raw_data,params)
  # scale or standardize data
  scaled_data = scaler.transform(processed_data)
  data, base = GetTestData(scaled_data, input_length, input_sample_step, device, relative=params["data"]["relative"])
  
  # prediction
  data_predict, _ = model(data)
  data_predict = data_predict.cpu().detach().numpy().reshape(output_length,output_dim)
  if params["data"]["relative"]:
    data_predict = data_predict + base
  if params["data"]["data_type"] == "position_passive":
    data_predict = np.concatenate([np.zeros(data_predict.shape),data_predict],axis=1)
    data_predict_recovered = scaler.inverse_transform(data_predict)
    data_out = data_predict_recovered
  else:
    data_predict_recovered = scaler.inverse_transform(data_predict)
    offset = data_predict_recovered[0,:]-processed_data[-1,:]
    data_predict_recovered = data_predict_recovered-offset
    time = np.arange(0,(output_length-1)*output_sample_step+1,output_sample_step)
    _, data_out = ResampleData(data_predict_recovered, time, (output_length-1)*output_sample_step)
  
  return data_out



