
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy.integrate import odeint
from src.generate_data.systems_collection import strogatz, mysystems
from plotting.parestim_sim_plots import plot_trajectories

##
method = "gpomo"
exp_type = 'parobs'

systems = {**strogatz, **mysystems}
data_version = 'all'
exp_version = 'e3'
structure_version = 's0'
analy_version = 'a2'
path_main = f"D:{os.sep}Experiments{os.sep}MLJ23"
path_base_in = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{exp_version}{os.sep}"
path_base_out = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{exp_version}{os.sep}analysis{os.sep}{analy_version}{os.sep}"
os.makedirs(path_base_out, exist_ok=True)

sys_names = ['myvdp']
snrs = ['inf', 30, 13]
set_obs = "full"
n_inits = 4
sim_step = 0.01 if data_version == 'allonger' else 0.1
sim_time = 20 if data_version == 'allonger' else 10
data_length = int(sim_time/sim_step)
plot_trajectories_bool = False

# sys_name, snr, iinit, ieq, eqsym = 'myvdp', 'inf', 0, 0, 'x'
results_gpomo_clean_list = []

for sys_name in sys_names:
    obss = systems[sys_name].get_obs("all")
    path_in = f"{path_base_in}{sys_name}{os.sep}"
    for snr in snrs:
        for iinit in range(n_inits):
            trajectories_true = []
            trajectories_true_inf = []
            trajectories_model = []

            # load true data
            data_filepath = f"{path_main}{os.sep}data{os.sep}lodefind{os.sep}"
            data_filename = f"data_lodefind_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}.csv"
            itr_true = np.array(pd.read_csv(data_filepath + data_filename))
            itr_true_inf = np.array(pd.read_csv(data_filepath +  f"data_lodefind_{sys_name}_{data_version}_len{data_length}_snrinf_init{iinit}.csv"))
            trajectories_true.append(itr_true)
            trajectories_true_inf.append(itr_true_inf)
            init_true = np.array(itr_true)[0, 1:]
            time = np.array(itr_true)[:, 0]

            for iobs in obss:
                iobs_name = "".join(iobs)
                dmax = 1 if len(iobs_name)==2 else 2
                filename_base = f"{method}_{sys_name}_dmax{dmax}_poly3_steps5120_obs{iobs_name}_snr{snr}_init{iinit}"
                results = pd.read_csv(f"{path_in}params_{filename_base}.csv")
                duration_df = pd.read_csv(f"{path_in}duration_{filename_base}.csv")

                try:
                    iparams = np.hstack((np.array(results.iloc[:, 1]), np.array(results.iloc[:, 2])))
                except:
                    print(f"Was not finished succesfully: {sys_name}_{data_version}_snr{snr}_init{iinit}_{iobs_name}")
                    ires = [method, exp_type, data_version, sys_name, snr, iinit, iobs_name] + list(np.full(6, np.nan))
                    results_gpomo_clean_list.append(ires)
                    continue

                def dxdt(t, x, p):
                    return [p[0]*1 + p[1]*x[1] + p[2]*x[1]**2 + p[3]*x[1]**3 + p[4]*x[0] + p[5]*x[0]*x[1] +
                            p[6]*x[0]*x[1]**2 + p[7]*x[0]**2 + p[8]*x[0]**2*x[1] + p[9]*x[0]**3,
                            p[10]*1 + p[11]*x[1] + p[12]*x[1]**2 + p[13]*x[1]**3 + p[14]*x[0] + p[15]*x[0]*x[1] +
                            p[16]*x[0]*x[1]**2 + p[17]*x[0]**2 + p[18]*x[0]**2*x[1] + p[19]*x[0]**3]

                simulation, odeint_output = odeint(lambda t, y: dxdt(t, y, iparams), init_true, time, rtol=1e-12, atol=1e-12, tfirst=True, full_output=True)
                if 'successful' not in odeint_output['message']:
                    TExy = np.nan
                    TExy_inf = np.nan
                else:
                    trajectories_model.append(np.column_stack([time.reshape((len(time), 1)), simulation]))

                    # trajectory error
                    TEx = np.sqrt((np.mean((simulation[:, 0] - itr_true[:, 1]) ** 2))) / np.std(itr_true[:, 1])
                    TEy = np.sqrt((np.mean((simulation[:, 1] - itr_true[:, 2]) ** 2))) / np.std(itr_true[:, 2])
                    TExy = TEx + TEy

                    TEx_inf = np.sqrt((np.mean((simulation[:, 0] - itr_true_inf[:, 1]) ** 2))) / np.std(itr_true_inf[:, 1])
                    TEy_inf = np.sqrt((np.mean((simulation[:, 1] - itr_true_inf[:, 2]) ** 2))) / np.std(itr_true_inf[:, 2])
                    TExy_inf = TEx_inf + TEy_inf

                # parameters
                true_params_unpacked = [j for i in systems[sys_name].gram_params for j in i]
                num_true_params = np.count_nonzero(true_params_unpacked)
                num_model_params = np.count_nonzero(iparams)

                # create expression
                terms = [str(results['Unnamed: 0'][i]).strip() for i in range(10)]
                expr1 = " + ".join([str(np.round(iparams[i], 5))+terms[i] for i in range(10)])
                expr2 = " + ".join([str(np.round(iparams[10+i], 5))+terms[i] for i in range(10)])

                # join results in list
                results_gpomo_clean_list.append([method, exp_type, data_version, sys_name, snr, iinit, iobs_name,
                                                 0, duration_df.x[0], TExy, TExy_inf, num_true_params, num_model_params, [expr1, expr2]])

                # plot phase trajectories in a single figure and save
                if plot_trajectories_bool:
                    plot_filepath = f"{path_base_out}plots{os.sep}{sys_name}{os.sep}"
                    os.makedirs(plot_filepath, exist_ok=True)
                    plot_filename = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_{analy_version}_len{data_length}_" \
                                    f"snr{snr}_init{iinit}_obs{iobs_name}_{{}}.{{}}"

                    plot_trajectories(trajectories_true, trajectories_model, trajectories_true_inf,
                                      fig_path=plot_filepath, fig_name=plot_filename, plot_one_example=True)

# put results in dataframe
column_names = ['method', 'exp_type', 'data_version', 'system', 'snr', 'iinit', 'iobs', 'ibest', 'duration',
                'TExy', 'TExy_inf', 'num_true_params', 'num_model_params', 'expr_model']
results_clean = pd.DataFrame(results_gpomo_clean_list, columns=column_names)

# save all results as a dataframe
results_clean.to_csv(f"{path_base_out}overall_results_table_{data_version}_"
                                 f"{structure_version}_{exp_version}_{analy_version}.csv", sep='\t')
content = tabulate(results_clean.values.tolist(), list(results_clean.columns), tablefmt="plain")
open(f"{path_base_out}overall_results_prettytable_{data_version}_{structure_version}_"
     f"{exp_version}_{analy_version}.csv", "w").write(content)
