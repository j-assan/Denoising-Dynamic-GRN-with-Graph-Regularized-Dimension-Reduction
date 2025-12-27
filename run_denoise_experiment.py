#!/usr/bin/env python3

# This example demonstrates how to ask sweetsweep to only run one of the experiments.
# This is useful when running an array job in an HPC cluster, as each job of the array will call the same script with one index integer parameter changing.
# An example of how to run array jobs on the Alliance Canada HPC cluster can be found here: https://docs.alliancecan.ca/wiki/Job_arrays
# You can pass the array job index to this script using '--only-exp-id N'. The code below processes it, and will run only the corresponding experiment.
# It's also useful to call this script beforehand with '--get-num-exp', to obtain the number of array jobs that will be submitted, and not entering it manually.

import os
import sys; sys.path += ['./src']; sys.path += ['../denoise-experiment']
import math
import json
import matplotlib.pyplot as plt
import time
import numpy as np

from src.parse import argparser
from sweetsweep import parameter_sweep_parallel, parameter_sweep, get_num_exp
from experiment import experiment
from sweepdict import files


# Create the dictionary of values to sweep for each parameter
# For example:

if '--input' in sys.argv:
    param_sweep, args = argparser(sys.argv[sys.argv.index('--input')+1])
else:
    param_sweep, args = argparser()

seed = args.pop('seed')
num_exp = get_num_exp(param_sweep)
seeds = np.random.default_rng(seed).spawn(num_exp)

## ------------------------------------
# Process argv parameters

# Don't print anything before this line
if '--get-num-exp' in sys.argv:
    print(get_num_exp(param_sweep))
    exit()

only_exp_id = None
direct_seed = None
if '--only-exp-id' in sys.argv:
    only_exp_id = int(sys.argv[sys.argv.index('--only-exp-id')+1])

exp_ids = None
if '--exp-ids' in sys.argv:
     exp_ids = [int(x) for x in (sys.argv[sys.argv.index('--exp-ids')+1]).split(',')]

## ------------------------------------


my_sweep_dir = "my_sweep"   # Default output dir
# Main folder for the sweep
if '-o' in sys.argv:
    my_sweep_dir = sys.argv[sys.argv.index('-o')+1]
if '--outdir' in sys.argv:
    my_sweep_dir = sys.argv[sys.argv.index('--outdir')+1]
parallel = False
if '--parallel' in sys.argv:
    parallel = True
else:
     parallel = False
os.makedirs(my_sweep_dir, exist_ok=True)

start_index = 0
if '--start-index' in sys.argv:
     start_index = int(sys.argv[sys.argv.index('--start-index')+1])


# Name of the image to save
image_filename = files(args['reg_lam'], args['maxrank'], args['n_ranks'])
# Name of the csv file to save (one row per experiment)
csv_filename = "results.csv"

# Save the param_sweep file
params = param_sweep.copy()
# Add parameters for the viewer if you need (see README.md)
params["viewer_filePattern"] = image_filename
# params["viewer_cropLBRT"] = [0, 0, 0, 0]
params["viewer_resultsCSV"] = csv_filename

def tuple_encoder(d):
    encoded = {}
    for k, v in d.items():
        dict_item = v
        if isinstance(v, list):
               dict_item = []
               for i in range(len(v)):
                    list_item = v[i]
                    if isinstance(v[i], tuple):
                         list_item = str(v[i])
                    dict_item.append(list_item)
        encoded[k] = dict_item
    return encoded
json.dump(tuple_encoder(params), open(os.path.join(my_sweep_dir, "sweep.txt"), "w"))

def experiment_wrapper(exp_id, param_dict, exp_dir, seeds=seeds, start_index=start_index):
    print("Experiment #%d:"%exp_id, param_dict)
    seed = seeds[exp_id-start_index] if seeds else None
    all_args = args | param_dict
    return experiment(all_args, exp_dir, seed)

if parallel:
	parameter_sweep_parallel(
		param_sweep,
		experiment_wrapper,
		my_sweep_dir,
		max_workers=4,
        start_index=start_index,
		result_csv_filename="results.csv"
	)
elif exp_ids:
     for exp in exp_ids:
          parameter_sweep(
            param_sweep,
            experiment_wrapper,
            my_sweep_dir,
            result_csv_filename="results.csv",
            only_exp_id=exp
	)
else:
    parameter_sweep(
		param_sweep,
		experiment_wrapper,
		my_sweep_dir,
        start_index=start_index,
		result_csv_filename="results.csv",
        only_exp_id=only_exp_id
	)
