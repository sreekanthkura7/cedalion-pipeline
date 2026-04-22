#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 06:12:33 2025

@author: smkelley
"""
#%% Imports

import os
import gzip
import pickle
import matplotlib.pyplot as plt
import cedalion
import xarray as xr
from cedalion import io, nirs, units
import cedalion.vis.misc.plot_probe_gui as vPlotProbe
import cedalion.geometry.registration as registration
from cedalion.dataclasses.geometry import PointType
from cedalion.physunits import units
import pint
import pandas as pd


#%%
# path2results = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS/derivatives/Shannon/cedalion"
path2results = "/projectnb/nphfnirs/s/datasets/Interactive_Walking_HD/derivatives/cedalion/new"
#path2results = "/projectnb/nphfnirs/s/users/shannon/Data/reg_test_data/ground_truth" #test_data/derivatives/cedalion"
#path2results = "/projectnb/nphfnirs/s/users/shannon/Data/reg_test_data/test_data/derivatives/cedalion"
path2results = "/projectnb/nphfnirs/s/datasets/Interactive_Walking_HD/derivatives/cedalion/final_ihope"

task = "IWHD"

filname = "task-" + task + "_nirs_groupaverage_chanspace_conc.nc"
filepath_bl = os.path.join(os.path.join(path2results, "group_results") , filname)
    


if os.path.exists(filepath_bl):
    # with open(filepath_bl, 'rb') as f:
    #     groupavg_results = pickle.load(f)
    groupavg_results = xr.open_dataset(filepath_bl)
        
    groupaverage = groupavg_results['group_average_weighted']
    groupaverage_unweighted = groupavg_results['group_average']
    blockaverage_stderr = groupavg_results['total_stderr']
    tstat = groupavg_results['tstat']
    geo2d = groupavg_results['geo2d']
    geo3d = groupavg_results['geo3d']
    print("Group average file loaded successfully!")

else:
    raise ValueError(f"Error: File '{filepath_bl}' not found!")
    
# groupaverage = groupaverage_unweighted_orig  # plot unweighted group averagegroupaverage_conc.sel(channel=ch_name)
# geo2d = registration.simple_scalp_projection(geo3d)

geo2d = geo2d.pint.quantify().rename({'pos2d': 'pos'}) # re-cast type coord from string back to PointType enum 
geo2d['type'] = xr.DataArray(pd.Series(geo2d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
    dims=geo2d['type'].dims)
geo3d = geo3d.pint.quantify().rename({'pos3d': 'pos'})
geo3d['type'] = xr.DataArray(pd.Series(geo3d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
    dims=geo3d['type'].dims)

#%% Convert to conc if in OD

# If blockaverage in OD, convert to conc

if 'wavelength' in groupaverage.dims:
    dpf = xr.DataArray(
        [1, 1],
        dims="wavelength",
       coords={"wavelength": groupaverage.wavelength},
    )

    # blockavg
    groupaverage = groupaverage.rename({'reltime':'time'})
    groupaverage.time.attrs['units'] = units.s
    groupaverage_conc = nirs.od2conc(groupaverage, geo3d, dpf, spectrum="prahl")
    groupaverage_conc = groupaverage_conc.rename({'time':'reltime'})
    
    # stderr
    blockaverage_stderr = blockaverage_stderr.rename({'reltime':'time'})
    blockaverage_stderr.time.attrs['units'] = units.s
    blockaverage_stderr_conc = nirs.od2conc(blockaverage_stderr, geo3d, dpf, spectrum="prahl")
    blockaverage_stderr_conc = blockaverage_stderr_conc.rename({'time':'reltime'})
else:
    groupaverage_conc = groupaverage.copy()
   
#%% Plot


vPlotProbe.run_vis(blockaverage = groupaverage_conc, geo2d = geo2d, geo3d = geo3d)



# %%
