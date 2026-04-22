#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:37:37 2025

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
import hrf_estimation as hrf


#%%
import importlib
importlib.reload(hrf)

# config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/config/config.yaml"
# config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion_pipeline_regression_test/configs/ref/config_ref_1.yml" 
# config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/config/config.yaml"

# config_path = "/projectnb/nphfnirs/s/datasets/Interactive_Walking_HD/derivatives/cedalion/final_ihope/config_STS_Q.yaml"
config_path = '/projectnb/nphfnirs/s/users/shannon/Data/test_data_cedalion_smk/data/derivatives/cedalion/test_0421/test_1/config_test_1.yml'


with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

cfg_dataset = config['dataset']

cfg_hrf = config['hrf_estimation']
# 
# subjects = cfg_dataset['subject']   # sub idx you want to test
dirs = os.listdir(cfg_dataset['root_dir'])
subjects = [d.replace("sub-", "") for d in dirs if "sub" in d and d.replace("sub-", "") not in config["dataset"]["subjects_to_exclude"]]
tasks = cfg_dataset['task'] #[0]
run = config["run"] = [f"{i:02d}" for i in range(1, int(config["dataset"]["num_runs"]) + 1)]
# run = cfg_dataset['run']
subjects = subjects[:2]  # only test on first 2 subjects for now

# Loop through lists of tasks and subjects
for subj in subjects:
    for task in tasks:
        cfg_hrf_loop = copy.deepcopy(cfg_hrf)
        
        preproc_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", "cedalion", cfg_dataset['derivatives_subfolder'], "preprocessed_data")
        
        run_files = [os.path.join(preproc_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_run-{r}_nirs_preprocessed.snirf") for r in run]
        data_quality_files = [os.path.join(preproc_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_run-{r}_nirs_dataquality.nc") for r in run]
                
        save_path = os.path.join(cfg_dataset['root_dir'], "derivatives", "cedalion", cfg_dataset['derivatives_subfolder'], "hrf_estimate", f"sub-{subj}")
        out_pkl = os.path.join(save_path,f"sub-{subj}_task-{task}_nirs_hrf_estimate_{cfg_hrf['rec_str']}.nc")
        
        os.makedirs(save_path, exist_ok=True)
                
        print(f"Processing sub-{subj}, task-{task} ...")
        
        hrf.hrf_est_func(cfg_hrf_loop, run_files, data_quality_files, out_pkl)
        
        
# %%
