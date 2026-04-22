#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:39:52 2025

@author: smkelley
"""

#%% Imports
import yaml
import os
import sys
# script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(script_dir)
# sys.path.append(parent_dir)
sys.path.append("/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts")

import groupaverage as groupavg

#%% Run func

import importlib
importlib.reload(groupavg)

# root_dir = "/projectnb/nphfnirs/s/users/shannon/Data/test_data_cedalion_smk/data/"
# config_path = os.path.join(root_dir, 'derivatives', 'cedalion', 'test', 'test_3', 'config_test_3.yml')
# config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/config/config.yaml"
config_path = '/projectnb/nphfnirs/s/users/shannon/Data/test_data_cedalion_smk/data/derivatives/cedalion/test_0421/test_1/config_test_1.yml'

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# CHANGE
blockavg = True  # if true, group average done on blockavg, false, done on img recon

cfg_dataset = config['dataset']
cfg_hrf = config['hrf_estimation']
cfg_groupaverage = config['groupaverage']
# flag_prune_channels = config['preprocess']['steps']['prune']['enable']


dirs = os.listdir(cfg_dataset['root_dir'])
# subject_list = [d for d in dirs if "sub" in d and d not in cfg_dataset["subjects_to_exclude"]]
subject_list = [d.replace("sub-", "") for d in dirs if "sub" in d and d not in cfg_dataset["subjects_to_exclude"]]


subjects = subject_list #cfg_dataset['subject']
task = cfg_dataset['task'][0]

blockavg_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", "cedalion", cfg_dataset['derivatives_subfolder'], "hrf_estimate")  #, f"sub-{subj}")
if blockavg:
    blockavg_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_hrf_estimate_{cfg_hrf['rec_str']}.nc") for subj in subjects ]
else:
    file_name = (
        #f"Xs_sub-{subj}_{task}"
        f"_cov_alpha_spatial_{config['image_recon']['alpha_spatial']}"
        + f"_alpha_meas_{config['image_recon']['alpha_meas']}"
        + (f"_recon_mode_{config['image_recon']['recon_mode']}")
        + ("_Cmeas" if config["image_recon"]["Cmeas"]["enable"] else "_noCmeas")
        + ("_SB" if config["image_recon"]["spatial_basis"]["enable"] else "_noSB")
        + ("_mag" if config["image_recon"]["mag"]["enable"] else "_ts")
        + (f"_{config['image_recon']['mag']['t_win'][0]}_{config['image_recon']['mag']['t_win'][1]}" if config["image_recon"]["mag"]["enable"] else "")
        + ".pkl.gz"
    )
    image_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", "cedalion", cfg_dataset['derivatives_subfolder'], "image_results")
    image_files =  [os.path.join(image_dir, f"sub-{subj}", (f"Xs_sub-{subj}_{task}" + file_name)) for subj in subjects ]


# data_quality_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_dataquality.json") for subj in subjects ]
# geo_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_geo.sidecar") for subj in subjects ]


if blockavg:
    save_path = os.path.join(cfg_dataset['root_dir'], "derivatives", "cedalion", cfg_dataset['derivatives_subfolder'], "groupaverage")
    out = os.path.join(save_path, f"task-{task}_nirs_groupaverage_chanspace_{cfg_hrf['rec_str']}.nc")
else:
    save_path = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'cedalion', cfg_dataset['derivatives_subfolder'], 'image_results')
    file_name = (
        f"Xs_groupavg_{task}"
        + f"_cov_alpha_spatial_{config['image_recon']['alpha_spatial']}"
        + f"_alpha_meas_{config['image_recon']['alpha_meas']}"
        + (f"_recon_mode_{config['image_recon']['recon_mode']}")
        + ("_Cmeas" if config["image_recon"]["Cmeas"]["enable"] else "_noCmeas")
        + ("_SB" if config["image_recon"]["spatial_basis"]["enable"] else "_noSB")
        + ("_mag" if config["image_recon"]["mag"]["enable"] else "_ts")
        + (f"_{config['image_recon']['mag']['t_win'][0]}_{config['image_recon']['mag']['t_win'][1]}" if config["image_recon"]["mag"]["enable"] else "")
        #+"_20subs"
        + ".pkl"
    )
    out = os.path.join(save_path, file_name)
der_dir = os.path.join(save_path)
if not os.path.exists(der_dir):
    os.makedirs(der_dir)

if blockavg:    
    groupavg.groupaverage_func(cfg_dataset, cfg_groupaverage, cfg_hrf, blockavg_files, out)
else:
    groupavg.groupaverage_func(cfg_dataset, cfg_groupaverage, cfg_hrf, image_files, out)


# %%

