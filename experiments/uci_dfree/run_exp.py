import os
import pickle
import subprocess
from datetime import datetime
import numpy as np

# flags
write_sbatch =True
submit       =True

model = ['DSVGP','DSVGP','SVGP','SVGP']
mll   = ['PLL','ELBO','PLL','ELBO']
dd = np.array([1,1,0,0]) # number of directions (use 0 for SVGP)
M = 800 # inducing matrix size
ni_list = (M/(dd+1)).astype(int) # ensures equal inducing matrix size

for jj,ni in enumerate(ni_list):

  # write a pickle file with the run info
  run_params_dir = "./param_files/"
  if os.path.exists(run_params_dir) is False:
    os.mkdir(run_params_dir)
  run_params = {}

  # select the dataset
  #run_params['data_file']                    = "../../../uci/protein/protein.mat" # use relative import
  #run_params['data_file']                    = "../../../uci/elevators/elevators.mat" # use relative import
  #run_params['data_file']                    = "../../../uci/kin40k/kin40k.mat" # use relative import
  #run_params['data_file']                    = "../../../uci/keggdirected/keggdirected.mat" 
  run_params['data_file']                    = "../../../uci/energy/energy.mat" # use relative import
  run_params['data_dir']                     = "./output/energy/" # save location for output

  run_params['mode']                         = model[jj] # DSVGP, SVGP
  run_params['mll_type']                     = mll[jj] # PLL or ELBO
  run_params['num_inducing']                 = ni_list[jj] # number of inducing
  run_params['num_directions']               = dd[jj] # number of directions
  run_params['minibatch_size']               = 512
  run_params['num_epochs']                   = 700
  run_params['verbose']                      = True
  run_params['inducing_data_initialization'] = False
  run_params['use_ngd']                      = False
  run_params['use_ciq']                      = False
  run_params['num_contour_quadrature']       = 15 # gpytorch default=15
  run_params['learning_rate_hypers']         = 0.01
  run_params['learning_rate_ngd']            = 0.1
  run_params['lr_sched']                     = "MultiStepLR"
  run_params['lr_benchmarks']                = 73*np.array([300,500])
  run_params['lr_gamma']                     = 0.1 # LR decrease rate
  # seed and date
  now     = datetime.now()
  seed    = int("%d%.2d%.2d%.2d%.2d"%(now.month,now.day,now.hour,now.minute,now.second))
  barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
  run_params['date']  = now
  run_params['seed']  = seed
  # file name
  if run_params['mode'] == "DSVGP":
    base_name = f"uci_DSVGP_ni_{run_params['num_inducing']}_nd_{run_params['num_directions']}"+\
              f"_ne_{run_params['num_epochs']}_ngd_{run_params['use_ngd']}"+\
              f"_ciq_{run_params['use_ciq']}_{barcode}"
  elif run_params['mode'] == "SVGP":
    base_name = f"uci_SVGP_ni_{run_params['num_inducing']}"+\
              f"_ne_{run_params['num_epochs']}_{barcode}"
  elif run_params['mode'] == "GradSVGP":
    base_name = f"uci_GradSVGP_ni_{run_params['num_inducing']}_nd_{run_params['num_directions']}"+\
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
    f.write(f"#SBATCH --partition=default_partition\n")
    f.write(f"#SBATCH --gres=gpu:1\n")
    f.write(f"python3 test.py {param_filename}\n")
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
