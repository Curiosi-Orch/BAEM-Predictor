"""Define the training and evaluation loops."""

from cmath import inf
import datetime
import logging
import math
import toml
import torch
from tqdm import tqdm
import matplotlib.pyplot as pyplot
import numpy as np
from predictors.utils import *
import os
import pandas as pd

def train(model, optimizer, scheduler, data_train, data_test, params):
  data_type = params["data"]["data_type"]
  
  logging.info('=== Start training ' + data_type + ' ===')
  loss_train_list = list()
  loss_test_list = list()
  best_loss = inf
  best_model = model
  patience = 0
  
  fig = pyplot.figure()
  fig_loss = fig.add_subplot(111)
  fig_loss.set_title("Loss for "+params["data"]["data_type"])
  fig_loss.set_xlabel("epoch")
  fig_loss.set_ylabel("loss")
  is_legend_shown = False
  pyplot.ion()

  num_epoch = params["hyper_parameters"]["num_epoch"]
  pbar_epoch = tqdm(total=num_epoch,unit='it',ncols=100,leave=True)
  for epoch in range(num_epoch):
    model.train()
    pbar_train = tqdm(data_train,unit='it',ncols=50,leave=False,desc="[Train]")
    pbar_test = tqdm(data_test,unit='it',ncols=50,leave=False,desc="[Test]")
    loss_train = 0
    loss_test = 0
    for _, (input, output) in enumerate(pbar_train):
      optimizer.zero_grad()
      _, loss = model(input, output)
      loss.backward()
      optimizer.step()
      loss_train += loss.item()
    with torch.no_grad():
      model.eval()
      for _, (input, output) in enumerate(pbar_test):
        _, loss = model(input, output)
        loss_test += loss.item()
    loss_train = loss_train/len(data_train)
    loss_test = loss_test/len(data_test)
    scheduler.step(loss_train)
    loss_train_list.append(loss_train)
    loss_test_list.append(loss_test)

    pbar_epoch.set_description(f"Epoch:{epoch+1}/{num_epoch} Loss_train: {loss_train:10.8f} Loss_test: {loss_test:10.8f}")
    pbar_epoch.update(1)
    
    fig_loss.plot(loss_train_list,color='b',label="train_loss")
    fig_loss.plot(loss_test_list,color='r',label="test_loss")
    if not is_legend_shown:
      fig_loss.legend(loc='upper left')
      is_legend_shown = True
    fig.canvas.draw()
    pyplot.pause(0.1)
  
    # summary_writer.add_scalars("loss",{"train": loss_train,"test": loss_test}, epoch+1)
    # Keep best model
    if epoch < params["hyper_parameters"]["warm_up_epoch"] and epoch:
      logging.info(f"Epoch:{epoch+1}/{num_epoch} \
                     warming up {epoch}/{params['hyper_parameters']['warm_up_epoch']}")
    elif (loss_test) < best_loss:
      logging.info(f"Epoch:{epoch+1}/{num_epoch} \
                     Validation improved from {best_loss} to {loss_test}")
      best_loss = loss_test
      best_model = model
      patience = 0
    else:
      logging.info(f"Epoch:{epoch+1}/{num_epoch} \
                     Validation did not improve. Patience: {patience} (max: {params['hyper_parameters']['max_patience']})")
      patience += 1
    # Early stopping
    if patience >= params["hyper_parameters"]["max_patience"]:
      logging.info("Triggered early stopping.")
      break

  # save model
  torch.save(best_model.cpu().state_dict(), params["result_path"]+'/model')
  # summary_writer.add_hparams(
  #   {k:v.__str__() if isinstance(v, list) else v for k, v in params.items()},
  #   {"test_loss": best_loss})
  logging.info(f"Training done. Epoch:{epoch+1}/{num_epoch} ")
  
  if not os.path.exists(params["result_path"]+"/figures"):
    os.makedirs(params["result_path"]+"/figures")
  pyplot.savefig(params["result_path"]+"/figures/loss.png")
  
  # save configurations
  with open(params["result_path"]+"/params.toml", 'w') as file:
    r = toml.dump(params, file)
    logging.info(r)
  
  df_loss_train = pd.DataFrame(loss_train_list,columns={"loss_train"})
  df_loss_test = pd.DataFrame(loss_test_list,columns={"loss_test"})
  df_loss = pd.concat([df_loss_train,df_loss_test],axis=1)
  df_loss.to_csv(params["result_path"]+"/loss.csv",index=False)
  
  print("\n=== Training done ===")
  pyplot.ioff()
  pyplot.show()
  
  
  
def evaluate(model, scaler, data_raw, data_processed, device, params):
  total_length = np.size(data_processed, 0)
  # data_mean = np.mean(data_processed,0)
  mean_length = params["data"]["output_length"]/2
  length_std = params["data"]["output_length_std"]
  
  # scale or standardize data
  data_scaled = scaler.transform(data_processed)
  
  data_raw_processed = np.array([])
  for d in data_raw:
    if data_raw_processed.size == 0:
      data_raw_processed = d.reshape((-1,1))
    else:
      data_raw_processed = np.concatenate((data_raw_processed,d.reshape((-1,1))),axis=1)
  
  output_dim = params["data"]["output_dimension"]
  output_length = params["data"]["output_length"]
  input_length = params["data"]["input_length"]
  input_sample_step = params["data"]["input_sample_step"]
  # TODO. need to fix the computation of error when output_sample_step is not 1
  output_sample_step = params["data"]["output_sample_step"]
  num_epoch = total_length-input_length
  # a record of each prediction from each start point, 
  # axis 0 is the dimension of output
  # axis 1 is the start point, from [start_of_raw_data+input_length+output_length-1] to [end_of_raw_data]
  # axis 2 is the prediction step
  errors = np.zeros((output_dim,
                     total_length-output_length-input_length+1,
                     output_length))
  pbar_epoch = tqdm(total=num_epoch,unit='it',ncols=100,leave=True)
  
  time_start = datetime.datetime.now()
  for i in range(num_epoch):
    data = data_scaled[:input_length*input_sample_step+i,:]
    data = GetTestData(data, input_length, input_sample_step, device)
    
    # prediction
    data_predict, _ = model(data)
    data_predict_recovered = scaler.inverse_transform(
      data_predict.cpu().detach().numpy().reshape(output_length,output_dim))
    # record the errors
    for j in range(output_dim):
      for k in range(max(0,output_length-i-1),min(output_length,total_length-input_length-i)):
        errors[j,i+k-output_length+1,k] = data_predict_recovered[k,j] - data_raw_processed[i+input_length+k,j]
    pbar_epoch.update(1)
  time_finish = datetime.datetime.now()
  time_execute = (time_finish-time_start).seconds*1000.0/num_epoch
  params["results"]["time_execution"] = time_execute
  
  # multiply weight of normal distribution   
  gaussian_kernel = np.zeros((output_length))
  for i in range(output_length):
    gaussian_kernel[i] = math.exp(-math.pow((i-mean_length),2)/(2*length_std))/(math.sqrt(2*math.pi)*length_std)
  gaussian_kernel = gaussian_kernel/np.sum(gaussian_kernel)
  
  computed_error = np.zeros((output_dim,total_length-output_length-input_length+1))
  for i in range(np.size(computed_error,0)):
    for j in range(np.size(computed_error,1)):
      computed_error[i,j] = np.sum(errors[i,j,:]*gaussian_kernel)
      
  prediction_expect = data_raw_processed[input_length+output_length-1:total_length,:].transpose()+computed_error
  
  df_error = pd.DataFrame(computed_error.transpose(),columns={"error"})
  df_filtered = pd.DataFrame(data_processed[input_length+output_length-1:total_length,:],columns={"filtered"})
  df_raw = pd.DataFrame(data_raw_processed[input_length+output_length-1:total_length,:],columns=["raw"])
  df_analysis = pd.concat([df_error,df_filtered,df_raw],axis=1)
  df_analysis.to_csv(params["result_path"]+"/analysis.csv",index=False)
  
  error_mean = np.mean(np.abs(computed_error),axis=1)
  error_std = np.std(computed_error,axis=1)
  params["results"]["error_mean"] = error_mean
  params["results"]["error_std"] = error_std
  with open(params["result_path"]+"/params.toml",'w') as file:
    r = toml.dump(params, file)
    logging.info(r)
  
  if not os.path.exists(params["result_path"]+"/figures"):
    os.makedirs(params["result_path"]+"/figures")
  fig_predict = pyplot.figure(0).gca()

  time = np.array(range(total_length))/1000.*params["data"]["frequency_sample"]
  fig_predict.plot(time[input_length+output_length-1:total_length], 
                  data_raw_processed[input_length+output_length-1:total_length,0], 
                  color="grey", alpha=0.5, label="raw")
  fig_predict.plot(time[input_length+output_length-1:total_length], 
                  data_processed[input_length+output_length-1:total_length,0], 
                  color="blue", label="filtered")
  fig_predict.plot(time[input_length+output_length-1:total_length],
                  prediction_expect[0,:],
                  color="red", label="prediction")
  fig_predict.legend()
  fig_predict.set_xlabel("time (s)")
  fig_predict.set_ylabel("value")
  
  pyplot.savefig(params["result_path"]+"/figures/plot.png")
  
  fig_error = pyplot.figure(1).gca()
  
  pyplot.plot(np.abs(computed_error[0,:]))
  fig_error.set_xlabel("time (s)")
  fig_error.set_ylabel("error")
  
  pyplot.savefig(params["result_path"]+"/figures/error.png")
  
  pyplot.show()