#!/bin/sh
#SBATCH --time=96:00:00
#SBATCH --mem=60gb
#SBATCH -c 38
#SBATCH --killable

source /vol/ek/Home/orlyl02/working_dir/python3_venv/bin/activate.csh
/vol/ek/Home/orlyl02/working_dir/python3_venv/bin/python3 /vol/ek/Home/orlyl02/working_dir/oligopred/mlp/first_hyp_param_tuning.py > /vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/picasso_esm_weights4.log
