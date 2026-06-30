#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:23:40 2025

@author: smkelley
"""

#%% Imports
import yaml
import os
import copy
import sys
# script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(script_dir)
# sys.path.append(parent_dir)

sys.path.append("/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts")
import preprocess as preproc

#%% Test
import importlib
importlib.reload(preproc)

# config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/config/config_STS_Q.yml"
# config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion_pipeline_regression_test/configs/ref/config_ref_1.yml" 
config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/config/config.yaml"

with open(config_path, 'r') as file:  # open config file
    config = yaml.safe_load(file)
    
cfg_dataset = config['dataset']
cfg_preprocess = config['preprocess']
cfg_hrf = config['hrf_estimation']
mse_amp_thresh = config['groupaverage']['mse']['mse_amp_thresh']

# subjects = cfg_dataset['subject'] 
dirs = os.listdir(cfg_dataset['root_dir'])
subjects = [d.replace("sub-", "") for d in dirs if "sub" in d and d.replace("sub-", "") not in cfg_dataset["subjects_to_exclude"]]
config["dataset"]["subject"] = subjects
if "run_list" in config["dataset"]:
    config["run"] = config["dataset"]["run_list"]
else:
    config["run"] = [f"{i:02d}" for i in range(1, int(config["dataset"]["num_runs"]) + 1)]
runs = config["run"]
tasks = cfg_dataset['task'] 
# runs = cfg_dataset['run']    

# Loop through lists of tasks, subjects, and runs
for subj in subjects:
    for task in tasks:
        for run in runs:
            cfg_preprocess_loop = copy.deepcopy(cfg_preprocess)
            cfg_hrf_loop = copy.deepcopy(cfg_hrf)
            mse_amp_thresh_loop = copy.deepcopy(mse_amp_thresh)
            
            snirf_path = f"{cfg_dataset['root_dir']}/sub-{subj}/nirs/sub-{subj}_task-{task}_run-{run}_nirs.snirf"
            events_path =  f"{cfg_dataset['root_dir']}/sub-{subj}/nirs/sub-{subj}_task-{task}_run-{run}_events.tsv"
            
            save_path = f"{cfg_dataset['root_dir']}/derivatives/cedalion/{cfg_dataset['derivatives_subfolder']}/preprocessed_data/sub-{subj}/"
            
            out_files = {
                "out_snirf" : f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_preprocessed.snirf",
                "out_sidecar": f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_dataquality.nc",
                }
            
            os.makedirs(save_path, exist_ok=True)  # make directory if it doesn't already exist

            print(f"Processing sub-{subj}, task-{task}, run-{run}...")
            
            preproc.preprocess_func(snirf_path, events_path, cfg_dataset, cfg_preprocess_loop, cfg_hrf_loop['stim_lst'], mse_amp_thresh_loop, out_files)
            

# %%

