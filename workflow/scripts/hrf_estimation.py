# -*- coding: utf-8 -*-
"""
Perform blockaverage to get HRF

Created on Thu Jun  5 09:40:42 2025

@author: shank
"""

#%% Imports

import os
import cedalion
import numpy as np
import xarray as xr
import pint
from cedalion import units
from cedalion.dataclasses.geometry import PointType
import gzip
import pickle
import json
import pandas as pd
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(script_dir, 'modules')
sys.path.append(modules_path)

import module_hrf_est as mhrf


#%% Block average func

def hrf_est_func(cfg_hrf, run_files, data_quality_files, out_file): 
    print(f'run_files: {run_files}')
        
    # update units 
    cfg_hrf['t_pre']= units(cfg_hrf['t_pre'])
    cfg_hrf['t_post']= units(cfg_hrf['t_post'])

    if cfg_hrf['GLM']['enable']:
        cfg_GLM = cfg_hrf['GLM']
        cfg_GLM["basis_func_params"]
        for param in cfg_GLM["basis_func_params"]:
            cfg_GLM["basis_func_params"][param] = units(cfg_GLM["basis_func_params"][param])
        if cfg_GLM['distance_threshold']:
            cfg_GLM['distance_threshold'] = units(cfg_GLM['distance_threshold'])

    # Loop through files
    pruned_chans_lst = []
    bad_channels_runs = []
  
    for file_idx, run in enumerate(run_files):        # loop through files and concatinate runs for GLM and epochs for blockaverage
        
        # if file path/ current run does not exist for this file, continue without it  (i.e. subj dropped out)
        if not os.path.isfile(run):  
            continue   
        
        # # Load in snirf for curr subj and run
        records = cedalion.io.read_snirf(fname = run, time_units = 'second' ) #FIXME: HARD CODED TIME UNITS
        rec = records[0]
        rec_str = cfg_hrf['rec_str']
        if rec_str not in rec.timeseries and rec_str == 'od_02' and 'od' in rec.timeseries:
            print("WARNING: hrf_estimation rec_str='od_02' is deprecated; using cleaned preprocessed 'od' instead.")
            rec_str = 'od'
        ts = rec[rec_str].copy()
        stim = rec.stim.copy() # select the stim for the given file 

        # Load in data quality info for current run
        ds = xr.open_dataset(data_quality_files[file_idx])
        pruned_channels = ds['pruned_channels'].values
        bad_channels = ds['bad_channels'].values
        geo2d = ds['geo2d']
        geo3d = ds['geo3d']
        ds.close()

        geo2d = geo2d.pint.quantify().rename({'pos2d': 'pos'}) # re-cast type coord from string back to PointType enum 
        geo2d['type'] = xr.DataArray(pd.Series(geo2d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
            dims=geo2d['type'].dims)
        geo3d = geo3d.pint.quantify().rename({'pos3d': 'pos'})
        geo3d['type'] = xr.DataArray(pd.Series(geo3d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
            dims=geo3d['type'].dims)
        
        # check if ts has dimension chromo
        if 'chromo' in ts.dims:
            ts = ts.transpose('chromo', 'channel', 'time')  # !!! try transpose(..., 'channel', 'time') to get rid of if statement
        else:
            ts = ts.transpose('wavelength', 'channel', 'time')
            
        ts = ts.assign_coords(samples=('time', np.arange(len(ts.time))))
        ts['time'] = ts.time.pint.quantify(units.s) # !!! already is s? do we need this HARD CODING SECONDS. FIX IN CEDALION FUNCS
                
        # get the epochs
        #FIXME: ADD IF GLM OR BLOCKAVG
        epochs_tmp = ts.cd.to_epochs(
                                    stim,  # stimulus dataframe
                                    set(stim[stim.trial_type.isin(cfg_hrf['stim_lst'])].trial_type), # select events  
                                    before = cfg_hrf['t_pre'],  # seconds before stimulus
                                    after = cfg_hrf['t_post'],  # seconds after stimulus
                                )
        #FIXME: IF GLM OR BLOCK
        if file_idx == 0:
            epochs_all = epochs_tmp
            all_runs = []
            all_runs.append( rec )

        else:
            epochs_all = xr.concat([epochs_all, epochs_tmp], dim='epoch')  # concatenate epochs from all runs
            all_runs.append( rec )


        # Concatenate all data qual stuff
        pruned_chans_lst.append(pruned_channels)
        bad_channels_runs.append(bad_channels)

        # DONE LOOP OVER FILES
    
    # Flatten list of bad channels and take only unique chan values
    bad_channels_flat = [x for xs in bad_channels_runs for x in xs]
    bad_channels_tmp = list(set(bad_channels_flat))

    # bad_chans_sat_flat = [x for xs in bad_chans_sat_runs for x in xs]
    # bad_chans_amp_flat = [x for xs in bad_chans_amp_runs for x in xs]
    # bad_chans_sat = list(set(bad_chans_sat_flat))
    # bad_chans_amp = list(set(bad_chans_amp_flat))
    

    if cfg_hrf['GLM']['enable']:
        print('Running GLM HRF estimation')
        glm_results, hrf_estimate, hrf_mse, bad_chans_mse_lst = mhrf.GLM(all_runs, cfg_hrf, geo3d, pruned_chans_lst)
    else:
        print('Running Block Average HRF estimation')
        hrf_estimate, hrf_mse, bad_chans_mse_lst = mhrf.blockaverage(epochs_all, cfg_hrf)
        glm_results = None

    #weights = glm_results.sm.
    
    bad_chans_mse_flat = [x for xs in bad_chans_mse_lst for x in xs]
    bad_chans_mse = list(set(bad_chans_mse_flat))

    bad_channels_all = np.unique(np.concat([bad_channels_tmp, bad_chans_mse]))
    
    # Save results as xr dataset to netcdf file
    ds_results = xr.Dataset()
    ds_results['hrf_est'] = hrf_estimate.pint.dequantify()  # dequant to save, will re-quant in groupaverage
    ds_results['mse_t'] = hrf_mse.pint.dequantify() # dequant to save, will re-quant in groupaverage
    ds_results['bad_channels'] = xr.DataArray(bad_channels_all, dims='bad_channel')
    geo2d_clean = rec.geo2d.pint.dequantify().rename({'pos': 'pos2d'}) # dequant to save, and rename pos to pos2d to avoid confusion with geo3d pos coords
    geo2d_clean['type'] = geo2d_clean['type'].astype(str) # convert type to str
    ds_results['geo2d'] = geo2d_clean
    geo3d_clean = rec.geo3d.pint.dequantify().rename({'pos': 'pos3d'}) # dequant to save, and rename pos to pos3d to avoid confusion with geo2d pos coords
    geo3d_clean['type'] = geo3d_clean['type'].astype(str) # convert type to str
    ds_results['geo3d'] = geo3d_clean

    # SAVE AS NETCDF FILE
    ds_results.to_netcdf(out_file, mode='w')
    
    print(f"Hrf estimation data saved successfully to {out_file}!")

    


def replace_bad_vals(data_array, bad_chans_amp, bad_chans_sat, bad_chans_mse, replacement_val, trial_type):
    # Change bad values to predetermined set val

    data_array.loc[dict(trial_type=trial_type, channel=bad_chans_amp)] = replacement_val
    data_array.loc[dict(trial_type=trial_type, channel=bad_chans_sat)] = replacement_val
    data_array.loc[dict(trial_type=trial_type, channel=bad_chans_mse)] = replacement_val

    return data_array

    
#%%

def main():
    
    cfg_hrf = snakemake.params.cfg_hrf
    run_files = snakemake.input.preproc  #.preproc_runs
    data_quality_files = snakemake.input.quality
    
    out_file = snakemake.output.net_hrf
    # out_json = snakemake.output.json
    # out_geo = snakemake.output.geo
    
    hrf_est_func(cfg_hrf, run_files, data_quality_files, out_file) #, out_json, out_geo)  #, out_blkavg_nc, out_epoch_nc)
    
   
    
if __name__ == "__main__":
    main()

