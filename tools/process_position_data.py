import matplotlib.pyplot as pyplot
import pandas as pd
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append(".")
from predictors.utils import ResampleData

def parse_data_from_string_list(str_list: list):
  data = np.array([]).reshape(-1,3)
  for row in str_list:
    str = row.replace('[','').replace(']','').split(',')
    n = list(map(np.float32, str))
    data = np.append(data, np.array(n).reshape(1,3), axis=0)
  return data

# merge and plot data
if __name__ == '__main__':
  
  # load data
  file_path = os.path.dirname(__file__)
  dir_label = "position_20220825/D2/task/circle/"
  file_label = "1"
  prune_length = 100
  data_master = pd.read_csv(file_path+'/../data/'+dir_label+'/samples/position_master_'+file_label+'.csv',
                            header=0, parse_dates=False, index_col=False)
  data_slave = pd.read_csv(file_path+'/../data/'+dir_label+'/samples/position_slave_'+file_label+'.csv',
                           header=0, parse_dates=False, index_col=False)
  
  time_master = data_master['time'].to_numpy()
  time_slave = data_slave['time'].to_numpy()
  min_time = max([min(time_master),min(time_slave)])
  max_time = min([max(time_master),max(time_slave)])

  data_master = data_master[data_master['time']>=min_time]
  data_master = data_master[data_master['time']<=max_time]
  data_slave = data_slave[data_slave['time']>=min_time]
  data_slave = data_slave[data_slave['time']<=max_time]
  
  data_master = data_master[prune_length:]
  data_slave = data_slave[prune_length:]
  # print("m: ",len(data_master),"s: ",len(data_slave))
  
  data_master.reset_index(drop=True, inplace=True)
  data_slave.reset_index(drop=True, inplace=True)
  
  position = parse_data_from_string_list(data_master['position_command'].to_list())
  position_d = parse_data_from_string_list(data_slave['position_command'].to_list())
  x = position[:,0:1]
  y = position[:,1:2]
  z = position[:,2:]
  x_d = position_d[:,0:1]
  y_d = position_d[:,1:2]
  z_d = position_d[:,2:]
  
  time_master = data_master['time'].to_numpy()
  time_slave = data_slave['time'].to_numpy()
  time_master = (time_master-time_master[0])/1000.
  time_slave = (time_slave-time_slave[0])/1000.
  n = (time_master[-1] - time_master[0])/0.01
  time, x = ResampleData(x,time_master,n)
  _, y = ResampleData(y,time_master,n)
  _, z = ResampleData(z,time_master,n)
  _, x_d = ResampleData(x_d,time_slave,n)
  _, y_d = ResampleData(y_d,time_slave,n)
  _, z_d = ResampleData(z_d,time_slave,n)
  
  data = pd.DataFrame({"time": time,
                       "x_d": x_d.squeeze(),"y_d": y_d.squeeze(),"z_d": z_d.squeeze(),
                       "x": x.squeeze(),"y": y.squeeze(),"z": z.squeeze()})
  

  figure = pyplot.figure()
  # axis = figure.gca(projection = '3d')
  # axis.scatter(x,y,z,s=1,c='r')
  # axis.scatter(x_d,y_d,z_d,s=1,c='b')
  
  axis_2d = figure.gca()
  axis_2d.plot(data["time"],x)
  axis_2d.plot(data["time"],x_d)
  pyplot.show()
  
  # data.to_csv(file_path+'/../data/'+dir_label+'/position_'+file_label+'.csv',index=False)
  