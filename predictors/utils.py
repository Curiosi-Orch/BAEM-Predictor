"""Utility functions."""

import numpy as np
import torch
from scipy.signal.signaltools import filtfilt
from scipy.signal import butter
import matplotlib.pyplot as pyplot

""" 
Gaussian filter
"""
def GetGaussianKernel(kernel_size):
  kernel = []
  if kernel_size%2 == 1:
    for i in range(kernel_size//2+1,0,-1):
      kernel.append(np.exp(-i**2/2))
    for j in range(kernel_size//2-1,-1,-1):
      kernel.append(kernel[j])
  else:
    for i in range(kernel_size//2,0,-1):
      kernel.append(np.exp(-i**2/2))
    for j in range(kernel_size//2-1,-1,-1):
      kernel.append(kernel[j])
  kernel = np.array(kernel)
  kernel = kernel / np.sum(kernel)
  return kernel

def Gaussian(kernel_size, data):
  if kernel_size%2==0 and kernel_size<=1:
      print('size shoud be larger than 1, and odd')
      return
  padding_data = []
  mid = kernel_size//2
  for i in range(mid):
    padding_data.append(0)
  padding_data.extend(data.tolist())
  for i in range(mid):
    padding_data.append(0)
  kernel = GetGaussianKernel(kernel_size)
  result = []
  for i in range(len(padding_data)-2*mid):
    temp = 0 
    for j in range(kernel_size):
      temp += kernel[j]*padding_data[i+j]
    result.append(temp)
  result = np.array(result)
  return result

""" Butter filter
  input: a list of numpy arrays
"""
def ButterLowpass(frequency_cutoff, frequency_sample, order=5):
  nyquist_frequency = 0.5 * frequency_sample
  normal_cutoff = frequency_cutoff / nyquist_frequency
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return b, a

def ButterLowpassFilter(data, frequency_cutoff, frequency_sample, order=5):
  result = []
  b, a = ButterLowpass(frequency_cutoff, frequency_sample, order=order)
  for d in data:
    # not use lfilter, because it has phase delay
    _result = filtfilt(b, a, d)
    result.append(_result)
  return result

""" Average filter
  input: a list of numpy arrays
"""
def AverageFilter(data, kernel_size, dim=1):
  result = []
  if dim == 1:
    kernel = np.ones((kernel_size,))/kernel_size
    for d in data:
      _result = (np.convolve(d, kernel, mode='same'))
      result.append(_result)
  # TODO. 2D convolution 
  return result

"""
    Generate sequence data for training
    @param - data: 2 dimension array, 0-sequence, 1-[input output].
    @param - target_indices: list of target indices in data
    @param - sequence_length: length of each sequence for training
    @param - batch_size: batch size for training
    @param - device for training
    @param - relative: is each squeence start with respect to the start point (from zero)
    @output - a list of tuples(input_seq,output_seq), 
              in which input_seq has 3 dimension (batch_size,sequence_length,feature_size),
              whereas output_seq has 2 dimension (batch_size,target_size)
"""
def GetTrainData(data, target_indices, input_length, output_length, 
                 batch_size, input_sample_step, device, output_sample_step=1, relative=False):
  data_seq = list()
  total_length = np.size(data,0)
  feature_size = np.size(data,1)
  target_dimension = len(target_indices)
  n = (total_length-(input_length-1)*input_sample_step-(output_length-1)*output_sample_step)//batch_size
  for i in range(n):
    seq_in = list()
    seq_out = list()
    for j in range(batch_size):
      in_start = batch_size*i+j
      in_end = in_start + (input_length-1)*input_sample_step + 1
      out_start = in_end - 1
      out_end = out_start + (output_length-1)*output_sample_step + 1
      _seq_in = data[in_start:in_end:input_sample_step,:].astype('float32').reshape(-1,feature_size)
      _seq_out = data[out_start:out_end:output_sample_step,target_indices].astype('float32').reshape(-1,target_dimension)
      if relative:
        _seq_in = _seq_in - _seq_in[0,:]
        _seq_out = _seq_out - _seq_in[0,:]
      seq_in.append(_seq_in)
      seq_out.append(_seq_out)
    seq_in = torch.Tensor(np.array(seq_in)).view(batch_size, input_length, -1).to(device)
    seq_out = torch.Tensor(np.array(seq_out)).view(batch_size, output_length, -1).to(device)
    data_seq.append((seq_in,seq_out))
  return data_seq


def GetTrainData_passive(data, input_length, output_length, 
                         batch_size, input_sample_step, output_sample_step, device, relative=False):
  data_seq = list()
  total_length = np.size(data,0)
  n = (total_length-input_length*input_sample_step)//batch_size
  for i in range(n):
    seq_in = list()
    seq_out = list()
    for j in range(batch_size):
      in_start = batch_size*i+j
      in_end = in_start + (input_length-1)*input_sample_step + 1
      out_start = in_end - 1 - (output_length-1)*output_sample_step
      _seq_in = data[in_start:in_end:input_sample_step,:3].reshape(-1,3)
      _seq_out = data[out_start:in_end:output_sample_step,3:].reshape(-1,3)
      if relative:
        _seq_in = _seq_in - _seq_in[0,:]
        _seq_out = _seq_out - _seq_in[0,:]
      seq_in.append(_seq_in)
      seq_out.append(_seq_out)
    seq_in = torch.Tensor(np.array(seq_in)).view(batch_size, input_length, -1).to(device)
    seq_out = torch.Tensor(np.array(seq_out)).view(batch_size, output_length, -1).to(device)
    data_seq.append((seq_in,seq_out))
  return data_seq


"""
    Get the latest several data of the sequence, for test and prediction  
"""
def GetTestData(data, input_length, input_sample_step, device, relative=False):
  total_length = np.size(data,0)
  feature_size = np.size(data,1)
  if total_length - (input_length-1)*input_sample_step - 1 >= 0:
    test_data_start = total_length - (input_length-1)*input_sample_step - 1 
  else:
    test_data_start = 0
  test_data = data[test_data_start:total_length:input_sample_step,:]
  base = test_data[0,:]
  if relative:
    test_data = test_data - base
  test_data = torch.from_numpy(test_data.astype('float32')).view(1,np.size(test_data,0),feature_size).to(device)
  return test_data, base

def ResampleData(data, time, num):
  step = (time[-1] - time[0])/num
  time_resampled = np.arange(time[0],time[-1],step)
  data_resampled = np.zeros([np.size(time_resampled,0),np.size(data,1)])
  for i in range(len(time_resampled)):
    index = np.where(time_resampled[i] <= time)[0][0]
    index = 1 if index <= 0 else index
    rate = (time_resampled[i]-time[index-1])/(time[index]-time[index-1])
    data_resampled[i,:] = (1-rate)*data[index-1,:]+rate*data[index,:]
  return time_resampled, data_resampled


def GenerateSigmoidInputData(batch_size, input_sequence_length, output_sequence_length):
  i = input_sequence_length + output_sequence_length
  t = torch.zeros(batch_size,1).uniform_(0,20 - i).int()
  b = torch.arange(-10, -10 + i).unsqueeze(0).repeat(batch_size,1) + t
  s = torch.sigmoid(b.float())
  return s[:, :input_sequence_length].unsqueeze(-1), s[:,-output_sequence_length:]

def ShowDataSample(data,time,batch_index,sequence_index,params):
  input_end = params["data"]["input_length"]*params["data"]["input_sample_step"]
  output_end = input_end+params["data"]["output_length"]*params["data"]["output_sample_step"]
  pyplot.scatter(time[:input_end:params["data"]["input_sample_step"]],
                 data[batch_index][0][sequence_index].flatten().cpu().detach().numpy())
  pyplot.scatter(time[input_end:output_end:params["data"]["output_sample_step"]],
                 data[batch_index][1][sequence_index].flatten().cpu().detach().numpy())
  pyplot.show()
  

