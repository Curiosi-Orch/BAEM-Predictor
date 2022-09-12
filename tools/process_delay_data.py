import matplotlib.pyplot as pyplot
import pandas as pd
import os
# import torch
import numpy as np


# merge and plot data
if __name__ == '__main__':
  # load data
  file_path = os.path.dirname(__file__)
  dir_label = "delay_20220825/D5/"
  file_label = "3"
  data_label = "tau_sm_truth"
  prune_length = 10
  data_master = pd.read_csv(file_path+'/../data/'+dir_label+"samples/delay_master_"+file_label+'.csv',
                          header=0, parse_dates=False, index_col=False, dtype='float64')
  
  data_master = data_master[prune_length:]
  data_master.reset_index(drop=True, inplace=True)
  
  data = pd.DataFrame({"time": (data_master['time']-data_master['time'][0])/1000.,
                       "tau": data_master[data_label]})

  tau = data["tau"].to_numpy()
  time = data["time"].to_numpy()
  
  pyplot.plot(time, tau, label='tau')
  pyplot.legend()
  pyplot.show()
  
  # data.to_csv(file_path+'/../data/'+dir_label+'/delay_'+file_label+'.csv',index=False)
  