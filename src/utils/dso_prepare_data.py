import os
import numpy as np
import pandas as pd
import time
import itertools
from src.utils.systems_collection import systems_collection
from dso import DeepSymbolicOptimizer


data_sizes = ["small", "large"]
snrs = [None, 30, 13]  # [None, 30, 13]
n_data = 4
method = "dso"
exp_version = "e1"
observability = "full"
exp_type = f"sysident_num_{observability}"
data_type = "train"

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
root_dir = ".\\symreg_methods_comparison"

# Get data
# data_size, sys_name, snr, iinit = 'small', 'vdp', 30, 0
for data_size in data_sizes:
    for sys_name in systems_collection:
        for snr in snrs:
            for iinit in range(n_data):

                time_end = 20 if data_size == "large" else 10
                time_step = 0.01 if data_size == "large" else 0.1

                path_data = f"{root_dir}{os.sep}data{os.sep}{data_type}{os.sep}{data_size}{os.sep}{sys_name}{os.sep}"
                fname_data = f"data_{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}.csv"
                data = pd.read_csv(path_data + fname_data)
                data_header = list(data.columns)

                path_data_out = f"{root_dir}{os.sep}data{os.sep}{data_type}{os.sep}{data_size}_for_dso{os.sep}{sys_name}{os.sep}"
                if not os.path.exists(path_data_out):
                    os.makedirs(path_data_out)

                if sys_name == 'lorenz':
                    data_x = data[['x', 'y', 'z', 'dx']]
                    data_y = data[['x', 'y', 'z', 'dy']]
                    data_z = data[['x', 'y', 'z', 'dz']]
                    data_x.to_csv(f"{path_data_out}data_{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}_x.csv", index=False, header=False)
                    data_y.to_csv(f"{path_data_out}data_{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}_y.csv", index=False, header=False)
                    data_z.to_csv(f"{path_data_out}data_{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}_z.csv", index=False, header=False)

                else:
                    data_x = data[['x', 'y', 'dx']]
                    data_y = data[['x', 'y', 'dy']]
                    data_x.to_csv(f"{path_data_out}data_{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}_x.csv", index=False, header=False)
                    data_y.to_csv(f"{path_data_out}data_{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}_y.csv", index=False, header=False)



