# symreg_methods_comparison

The repository contains the code for evaluation of five symbolic regression methods, as was done in the paper _Probabilistic grammars for modeling dynamical systems from coarse, noisy, and partial data_ (in submission). The repository is still undergoing updates.


## Part 0 - Preparation

1. Download/Clone the repository to the prefered location.

2. Pull the singularity container from the SyLabs singularity library. To do that first install singularity (if you have not already), go to the repository location and pull the container from the library using this command:
`singularity pull --arch amd64 library://nomejc/symreg/symreg:latest`. If the command fails, you will have to change singularity's default remote server following these steps:
    * Run command `singularity remote add --no-login SylabsCloud cloud.sylabs.io`
    * Run command `singularity remote use SylabsCloud`
    * Run the same command `singularity pull ...` as above.

4. Download Dynobench benchmark from the Zenodo platform, that is located here: https://zenodo.org/records/10041312. Save the dataset folder `.\dynobench\data\*` inside the symreg_methods_comparison folder

5. Modify the data files for DSO using the script `.\utils\dso_prepare_data.py`. Similarly, modify the data for L-ODEfind and GPoM using the script `.\utils\lodefind_gpom_prepare_data.py`. The modified data files will be saved inside .\data folder.

6. Create candidate structures for ProGED using `.\src\proged_generate_structures.py`.

## Part I - System identification using training datasets
We ran system identification with the methods ProGED, DSO and SINDy on the high-performance computing cluster. To repeat the experiments, follow the instructions below.

3. Copy the symreg.sif container as well as the folders .\src, .\data and .\results to the local node on the cluster. Note that with ProGED, we first create possible structures using grammars locally, using the script `.\src\check_proged\proged_generate_structures.py`. The structures are saved in the path `.\symreg_methods_comparison\results\sysident_num_full\proged\structures` for full observability scenario. The structures folder should also be copied to the cluster, for ProGED to run properly.
  
4. Run the bash shell script that corresponds to the method you want to run (e.g. for SINDy, run `run_sindy.sh`). We call the scripts using slurm's `sbatch` command. For ProGED, run `run_proged_outer.sh`, which will call the `run_proged.sh`. The reason is this way more than one thousand jobs can be submitted to the slurm. Make sure that you also manually create the ./slurm folder for the log files, otherwise the jobs will fail.

We ran other methods, GPoM and L-ODEfind, locally, using the scripts `.\src\check_gpom\gpom_system_identification.R` and `.\src\check_lodefind\lodefind_system_identification.py` 

## Part II - Validation datasets
In this part, we evaluate all the models that were returned by the methods using the validation datasets. We ran the validation on the cluster as well, using the command `sbatch run_validation.sh`. The bash script runs the python code `common1_validation_hpc.py`.

## Part III - Evaluation of best model per method using three metrics - trajectory error on test data, term difference and complexity
To do that, first run the `common2_testing.py` and then the `common3_TD_complexity.py` script. The figures for the paper were created using the `common4_make_figures.py` script.







