import os
import pandas as pd
import numpy as np

method = "lodefind"
exp_type = "partial"
exp_version = "e1"
data_sizes = ["small", "large"]
observability = "full"  # either full, part or all
snrs = [None, 30, 13]
n_data = np.arange(0, 4)
data_type = "train"
sys_names = ['vdp']

main_path = "D:\\Experiments\\symreg_methods_comparison\\data"

# Get data
for data_size in data_sizes:
    time_end = 20 if data_size == "large" else 10
    time_step = 0.01 if data_size == "large" else 0.1
    for sys_name in sys_names:
        path_data_out = f"{main_path}{os.sep}{data_type}\\{data_size}_for_lodefind_gpom\\{sys_name}\\"
        os.makedirs(path_data_out, exist_ok=True)
        for snr in snrs:
            for iinit in n_data:
                print(f"Processing {sys_name} | {data_size} | {snr} dB | init: {iinit}")
                path_data_in = f"{main_path}{os.sep}{data_type}{os.sep}{data_size}{os.sep}{sys_name}{os.sep}"
                data_filename = f"data_{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}.csv"
                data = pd.read_csv(path_data_in + data_filename)
                data_header = list(data.columns)

                if sys_name == 'lorenz':
                    data_new = data[['t', 'x', 'y', 'z']]
                else:
                    data_new = data[['t', 'x', 'y']]

                data_new = data_new.rename(columns={'t': 'time'})
                data_new.to_csv(f"{path_data_out}data_{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}.csv", index=False, header=True)





