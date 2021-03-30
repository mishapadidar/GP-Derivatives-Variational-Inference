import os
import pickle
import subprocess
from datetime import datetime

# write a pickle file with the run info
run_params_dir = "./param_files/"
if os.path.exists(run_params_dir) is False:
  os.mkdir(run_params_dir)
run_params = {}
run_params['num_inducing']                 = 1024
run_params['num_directions']               = 2
run_params['minibatch_size']               = 1024
run_params['num_epochs']                   = 1
run_params['tqdm']                         = False
run_params['inducing_data_initialization'] = False
run_params['use_ngd']                      = False
run_params['use_ciq']                      = False
run_params['learning_rate_hypers']         = 1e-3
run_params['learning_rate_ngd']            = 1e-3
run_params['gamma']                        = 10
# seed and date
now     = datetime.now()
seed    = int("%d%.2d%.2d%.2d%.2d"%(now.month,now.day,now.hour,now.minute,now.second))
barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
run_params['date']  = now
run_params['seed']  = seed
# file name
base_name = f"stell_regress_ni_{run_params['num_inducing']}_nd_{run_params['num_directions']}"+\
            f"_ne_{run_params['num_epochs']}_ngd_{run_params['use_ngd']}"+\
            f"_ciq_{run_params['use_ciq']}_{barcode}"
run_params['base_name']  = base_name
param_filename = run_params_dir + "params_" +base_name + ".pickle"
pickle.dump(run_params,open(param_filename,'wb'))

# write a slurm batch script
slurm_dir  = "./slurm_scripts/"
if os.path.exists(slurm_dir) is False:
  os.mkdir(slurm_dir)
slurm_name = slurm_dir + base_name + ".sub"
#slurm_name = base_name + ".sub"
f = open(slurm_name,"w")
f.write(f"#!/bin/bash\n")
f.write(f"#SBATCH -J  stell\n")
f.write(f"#SBATCH -o ./slurm_output/job_{barcode}.out\n")
f.write(f"#SBATCH -e ./slurm_output/job_{barcode}.err\n")
f.write(f"#SBATCH --get-user-env\n")
f.write(f"#SBATCH -N 1\n")
f.write(f"#SBATCH -n 1\n")
f.write(f"#SBATCH --mem=15000\n")
f.write(f"#SBATCH -t 168:00:00\n")
f.write(f"#SBATCH --partition=default_gpu\n")
f.write(f"#SBATCH --gres=gpu:1\n")
f.write(f"python3 stellarator_regression.py {param_filename}\n")

# write the shell submission script
submit_name = slurm_dir + 'slurm_submit.sh'
f = open(submit_name,"w")
f.write(f"#!/bin/bash\n")
f.write(f"sbatch {slurm_name}")
f.close()

# submit the script
bash_command = f"sbatch {slurm_name}"
subprocess.run(bash_command.split(" "))
