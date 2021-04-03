import os
import pickle
import subprocess
from datetime import datetime
import numpy as np

# flags
write_sbatch =True
submit       =True

#ni_list = [512,1024, 2048, 4096]
#for ni in ni_list:
n_dir_list = [1]
for dd in n_dir_list:

  # write a pickle file with the run info
  run_params_dir = "./param_files/"
  if os.path.exists(run_params_dir) is False:
    os.mkdir(run_params_dir)
  run_params = {}
  run_params['mode']                         = "DSVGP" # or SVGP
  run_params['n']                            = 10000 # number of total data points
  run_params['dim']                          = 5 # problem dimension
  run_params['num_inducing']                 = 512
  run_params['num_directions']               = dd
  run_params['minibatch_size']               = 256
  run_params['num_epochs']                   = 800
  run_params['tqdm']                         = False
  run_params['inducing_data_initialization'] = False
  run_params['use_ngd']                      = False
  run_params['use_ciq']                      = False
  run_params['num_contour_quadrature']       = 10 # gpytorch default=15
  run_params['learning_rate_hypers']         = 0.01  
  run_params['learning_rate_ngd']            = 0.1
  run_params['lr_benchmarks']                = np.array([50,150,300])
  run_params['lr_gamma']                     = 10.0
  run_params['lr_sched']                     = None
  run_params['data_file'] = "../../data/focus_w7x_dataset.csv"
  # seed and date
  now     = datetime.now()
  seed    = int("%d%.2d%.2d%.2d%.2d"%(now.month,now.day,now.hour,now.minute,now.second))
  barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
  run_params['date']  = now
  run_params['seed']  = seed
  # file name
  if run_params['mode'] == "DSVGP":
    base_name = f"synthetic1_DSVGP_ni_{run_params['num_inducing']}_nd_{run_params['num_directions']}"+\
              f"_ne_{run_params['num_epochs']}_ngd_{run_params['use_ngd']}"+\
              f"_ciq_{run_params['use_ciq']}_{barcode}"
  elif run_params['mode'] == "SVGP":
    base_name = f"synthetic1_SVGP_ni_{run_params['num_inducing']}"+\
              f"_ne_{run_params['num_epochs']}_{barcode}"
  run_params['base_name']  = base_name
  param_filename = run_params_dir + "params_" +base_name + ".pickle"
  pickle.dump(run_params,open(param_filename,'wb'))
  print(f"Dumped param file: {param_filename}")
  
  if write_sbatch:
    # write a slurm batch script
    slurm_dir  = "./slurm_scripts/"
    if os.path.exists(slurm_dir) is False:
      os.mkdir(slurm_dir)
    slurm_name = slurm_dir + base_name + ".sub"
    #slurm_name = base_name + ".sub"
    f = open(slurm_name,"w")
    f.write(f"#!/bin/bash\n")
    f.write(f"#SBATCH -J  {run_params['mode']}_{run_params['num_directions']}\n")
    f.write(f"#SBATCH -o ./slurm_output/job_%j.out\n")
    f.write(f"#SBATCH -e ./slurm_output/job_%j.err\n")
    f.write(f"#SBATCH --get-user-env\n")
    f.write(f"#SBATCH -N 1\n")
    f.write(f"#SBATCH -n 1\n")
    f.write(f"#SBATCH --mem=15000\n")
    f.write(f"#SBATCH -t 168:00:00\n")
    f.write(f"#SBATCH --partition=default_gpu\n")
    f.write(f"#SBATCH --gres=gpu:1\n")
    f.write(f"python3 synthetic1.py {param_filename}\n")
    print(f"Dumped slurm file: {slurm_name}")
    
    # write the shell submission script
    submit_name = slurm_dir + 'slurm_submit.sh'
    f = open(submit_name,"w")
    f.write(f"#!/bin/bash\n")
    f.write(f"sbatch --requeue {slurm_name}")
    f.close()
    print(f"Dumped bash script: {submit_name}")
  
  if submit:
    # submit the script
    #bash_command = f"sbatch {slurm_name}"
    bash_command = f"bash {submit_name}"
    subprocess.run(bash_command.split(" "))
