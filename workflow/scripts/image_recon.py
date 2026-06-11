#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 07:17:55 2025

@author: smkelley
"""


import os
import cedalion
import cedalion.nirs

from cedalion.physunits import units
import xarray as xr
from cedalion.dataclasses.geometry import PointType
from cedalion.sigproc.quality import measurement_variance
import cedalion.io as io
from cedalion.dot import image_recon as ir
from cedalion.vis.anatomy import image_recon_multi_view
import numpy as np
import gzip
import pickle
import sys
import pandas as pd

import pyvista as pv

pv.OFF_SCREEN = True

# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')


#%%

def img_recon_func(cfg_img_recon, cfg_hrf, file_name, Adot_path, out, SB=[], root_dir=None, derivatives_subfolder=None):

    # Convert str vals to units from config
    cfg_mse = cfg_img_recon['mse']
    cfg_sb = cfg_img_recon['spatial_basis']
    cfg_sb["threshold_brain"] = units(cfg_sb["threshold_brain"])
    cfg_sb["threshold_scalp"] = units(cfg_sb["threshold_scalp"])
    cfg_sb["sigma_brain"] = units(cfg_sb["sigma_brain"])
    cfg_sb["sigma_scalp"] = units(cfg_sb["sigma_scalp"])
    if isinstance(cfg_mse["mse_min_thresh"], str):
        cfg_mse["mse_min_thresh"] = float(eval(cfg_mse["mse_min_thresh"]))
    if isinstance(cfg_mse["hrf_val"], str):
                cfg_mse["hrf_val"] = float(cfg_mse["hrf_val"])
    if isinstance(cfg_mse["mse_val_for_bad_data"], str):
                cfg_mse["mse_val_for_bad_data"] = float(cfg_mse["mse_val_for_bad_data"])

    if isinstance(cfg_img_recon["alpha_meas"], str):
        cfg_img_recon["alpha_meas"] = float(cfg_img_recon["alpha_meas"])
    if isinstance(cfg_img_recon["alpha_spatial"], str):
        cfg_img_recon["alpha_spatial"] = float(cfg_img_recon["alpha_spatial"])

    if isinstance(cfg_img_recon["lambda_spatial_depth"], str):
            cfg_img_recon["lambda_spatial_depth"] = float(eval(cfg_img_recon["lambda_spatial_depth"]))
    
        
    #%% Load sensitivity matrix 

    Adot = io.forward_model.load_Adot(Adot_path)
    ec = cedalion.nirs.get_extinction_coefficients(cfg_img_recon['spectrum'], Adot.wavelength)
    head = cedalion.dot.get_standard_headmodel(cfg_img_recon['generate_sensitivity']['head_model'])

    #%% run image recon
    
    """
    do the image reconstruction of each subject independently 
    - this is the unweighted subject block average magnitude 
    - then reconstruct their individual MSE
    - then get the weighted average in image space 
    - get the total standard error using between + within subject MSE 
    """
    
    # load files
    if 'hrf' in file_name:
        results = xr.open_dataset(file_name) # load in data
        ts = results['hrf_est'].pint.quantify() 
        mse_t = results['mse_t'].pint.quantify()      
        if 'bad_channels' in results.keys(): # this is only results for hrf_est 
            bad_channels = results['bad_channels']

        geo2d = results['geo2d'] # grab geometry vals
        geo3d = results['geo3d']
        geo2d = geo2d.pint.quantify().rename({'pos2d': 'pos'}) # re-cast type coord from string back to PointType enum 
        geo2d['type'] = xr.DataArray(pd.Series(geo2d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
            dims=geo2d['type'].dims)
        geo3d = geo3d.pint.quantify().rename({'pos3d': 'pos'})
        geo3d['type'] = xr.DataArray(pd.Series(geo3d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
            dims=geo3d['type'].dims)

    elif 'preprocess' in file_name:
        records = cedalion.io.read_snirf(fname = file_name, time_units = 'second' ) #FIXME: HARD CODED TIME UNITS
        rec = records[0]
        if 'od_corrected' in rec.timeseries.keys(): #naming conventions change based on what file we load
            ts = rec['od_corrected'].copy()
        else:
             ts = rec['od_02'].copy()  # name if saved as snirf
        mse_t = None

        # FIXME: add bad_indices to file that also has rec in it (FOR IMG RECON ON PREPROC TIME SERIES)
            # FIXME: ADD optional input to img recon rule in snakefile 
            # this also has geo coords so no longer have to save in with sens mat
            # ds = xr.open_dataset(data_quality_files)
            # pruned_channels = ds['pruned_channels'].values
            # bad_channels = ds['bad_channels'].values
            # geo2d = ds['geo2d']
            # geo3d = ds['geo3d']
            # ds.close()

    print(f'Performing image recon on subject = {file_name}')

    # Loop through trial types
    all_trial_Xs = None
    for trial_type in cfg_hrf['stim_lst']: #NOTE: do we still need to loop through trial types here?
        
        print( f'   Getting images for trial type = {trial_type}')       

        ts_trial = ts.sel(trial_type=trial_type) 
        if mse_t is not None: # if hrf data loaded in
            mse_trial = mse_t.sel(trial_type=trial_type)

        # Convert conc to od and units for cfg
        if 'chromo' in ts.dims:
            dpf = xr.DataArray(
                    [1, 1],
                    dims="wavelength",
                    coords={"wavelength": Adot.wavelength},
                    )
            od_ts =  cedalion.nirs.cw.conc2od(ts_trial, geo3d, dpf)
            if mse_t is not None:  # would not need this in theory bc if loading in ts, then conc should not be there 
                od_mse = xr.dot(ec**2, mse_trial, dim =['chromo']) * 1 * units.mm**2  
            else:
                od_mse = measurement_variance(od_ts, calc_covariance=False) #NOTE: CHECK DIMS 
        # already in OD space
        else:
            od_ts = ts_trial.copy()
            if mse_t is not None: # if mse variable exists, i.e. loading in hrf not ts
                od_mse = mse_trial.copy()
            else:
                mse = measurement_variance(od_ts, calc_covariance=False) #NOTE: CHECK DIMS 
                od_mse = mse.sel(trial_type=trial_type)

        # replace bad vals #FIXME: this will fail if running on preprocessed time series and not hrf
        od_ts.loc[dict(channel=bad_channels)] = cfg_mse['hrf_val']
        od_mse.loc[dict(channel=bad_channels)] = cfg_mse['mse_val_for_bad_data']
        od_mse = xr.where(od_mse < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], od_mse)  # !!! maybe can be removed when we have the between subject mse
        
        # if doing magnitude image
        if cfg_img_recon['mag']['enable']:
            if 'reltime' in od_ts.dims:
                od_ts_mag = od_ts.sel(reltime=slice(cfg_img_recon['mag']['t_win'][0], cfg_img_recon['mag']['t_win'][1])).mean('reltime')
            else:
                od_ts_mag = od_ts.sel(time=slice(cfg_img_recon['mag']['t_win'][0], cfg_img_recon['mag']['t_win'][1])).mean('time')
        else:
            od_ts_mag = od_ts.copy()


        if mse_t is not None: # if hrf loaded in, get mse magnitude
            if 'reltime' in od_ts.dims:
                od_mse_mag = od_mse.mean('reltime')
            else:
                od_mse_mag = od_mse.mean('time')
        else:
             od_mse_mag = od_mse.copy() # if mse not loaded in, copy od_mse

        C_meas = od_mse_mag.pint.dequantify()
       
        #FIXME: save G (spatial basis) in derivatives/cedalion/forward_model  -> for brain and scalp separately and sigma
       
        if cfg_sb['enable'] and SB:  # do I need both
            #fil_path, after = Adot_path.split("fw", 1)
            #print(  'Performing image recon with SB')
            with gzip.open(SB, 'rb') as f:
                sbf = pickle.load(f)

            recon_obj = ir.ImageRecon(
                    Adot,
                    recon_mode=cfg_img_recon['recon_mode'],
                    brain_only = cfg_img_recon['BRAIN_ONLY']['enable'],
                    alpha_meas = cfg_img_recon['alpha_meas'],
                    alpha_spatial = cfg_img_recon['alpha_spatial'],
                    lambda_R_conc = cfg_img_recon['lambda_spatial_depth'],
                    apply_c_meas = cfg_img_recon['Cmeas']['enable'],
                    spatial_basis_functions = sbf,
                )
        else:
             sbf = None
             #print(  'Performing image recon without SB')
             recon_obj = ir.ImageRecon(
                    Adot,
                    recon_mode=cfg_img_recon['recon_mode'],  # conc is direct, mua2conc is indirect
                    brain_only = cfg_img_recon['BRAIN_ONLY']['enable'],
                    alpha_meas = cfg_img_recon['alpha_meas'],
                    alpha_spatial = cfg_img_recon['alpha_spatial'],
                    lambda_R_conc = cfg_img_recon['lambda_spatial_depth'],
                    apply_c_meas = cfg_img_recon['Cmeas']['enable'],
                    spatial_basis_functions = None,
                )
        # reconstruct image with or without C_meas, depending on config
        if cfg_img_recon['Cmeas']['enable']:
            Xs = recon_obj.reconstruct(od_ts_mag, c_meas=C_meas)
        else:
            Xs = recon_obj.reconstruct(od_ts_mag)
        
        # calculate image noise
        #FIXME: HAVE OPTION IN CONFIG IF CALCULATING NORMAL OR POSTERIOR?
            # is the old way just wrong or can it still be an option?
        if cfg_img_recon['noise_est_method'] == 'posterior':
            X_mse = recon_obj.get_image_noise_posterior(C_meas) #FIXME: this gets rid of trial type coord somewhere
        elif cfg_img_recon['noise_est_method'] == 'measurement':
            X_mse = recon_obj.get_image_noise(C_meas)
        else:
            X_mse = recon_obj.get_image_noise(C_meas) # estimate from this if not specified. 
        
        if 'trial_type' not in X_mse.coords:
            X_mse = X_mse.assign_coords(trial_type=trial_type) # add back trial type coord  

        X_mse = X_mse.pint.to('molar**2')
        Xs = Xs.pint.to('molar')
        
        if all_trial_Xs is None:
            all_trial_Xs = Xs.expand_dims(trial_type=[trial_type])
            all_trial_X_mse = X_mse.expand_dims(trial_type=[trial_type])
        else:
            all_trial_Xs = xr.concat([all_trial_Xs, Xs], dim='trial_type')
            all_trial_X_mse = xr.concat([all_trial_X_mse, X_mse], dim='trial_type')


    # IF loading in time series, save in parcel space instead of vertex for smaller file size
    if mse_t is None: # if time series data loaded in
         Xs_parcel_weighted = (
                    (all_trial_Xs / all_trial_X_mse)            # numerator weights: X * (1/var)
                    .groupby("parcel")
                    .sum("vertex")
                    /
                    (1 / all_trial_X_mse)
                    .groupby("parcel")
                    .sum("vertex")
                )

    # END OF TRIAL TYPE LOOP

    geo2d_clean = geo2d.pint.dequantify().rename({'pos': 'pos2d'}) # dequant to save, and rename pos to pos2d to avoid confusion with geo3d pos coords
    geo2d_clean['type'] = geo2d_clean['type'].astype(str) # convert type to str
    geo3d_clean = geo3d.pint.dequantify().rename({'pos': 'pos3d'}) # dequant to save, and rename pos to pos3d to avoid confusion with geo2d pos coords
    geo3d_clean['type'] = geo3d_clean['type'].astype(str) # convert type to str

    ds_results = xr.Dataset()
    if mse_t is not None: 
        ds_results['Xs'] = all_trial_Xs.pint.dequantify()  
        ds_results['X_mse'] = all_trial_X_mse.pint.dequantify() 
        ds_results['geo2d'] = geo2d_clean
        ds_results['geo3d'] = geo3d_clean
    else:
        ds_results['Xs'] = Xs_parcel_weighted.pint.dequantify()   # IF loading in time series, save in parcel space instead of vertex for smaller file size
        ds_results['X_mse'] = all_trial_X_mse.pint.dequantify() 
        ds_results['geo2d'] = geo2d_clean
        ds_results['geo3d'] = geo3d_clean

    ds_results.to_netcdf(out, mode='w') # save as netcdf file 

    #NOTE: we have Xmse for if using Cmeas or not, so do we save for both cases?
        # group avg would fail without. 
        # we just did not take in account the covariance of the data when reconstructing the image
    
    
    # #%% build and save plots

    if cfg_img_recon['plot_image']['enable']:
        save_dir_tmp = os.path.join(root_dir, derivatives_subfolder, 'cedalion', 'plots', 'image_recon')
        os.makedirs(save_dir_tmp, exist_ok=True)

        folder_name = out.removeprefix("Xs_").removesuffix(".nc")
        suffix = out.split("Xs_")[1].split("_BS")[0]

        intensity = np.log10(Adot[:,:,0].sum('channel')) # make non-sensitive vertices NaN / gray in image
        mask = intensity > -2
        sensitivity_mask = mask.drop_vars('wavelength')

        all_trial_X_stderr = np.sqrt(all_trial_X_mse)
        all_trial_X_tstat = all_trial_Xs / all_trial_X_stderr

        all_trial_Xs_plot = all_trial_Xs.where(sensitivity_mask)
        all_trial_X_stderr              = all_trial_X_stderr.where(sensitivity_mask)
        all_trial_X_tstat               = all_trial_X_tstat.where(sensitivity_mask)

        plot_img = cfg_img_recon['plot_image']

        flag_hbo_list = plot_img['flag_hbo_list']  
        flag_brain_list = plot_img['flag_brain_list']
        flag_img_list = plot_img['flag_img_list'] 
            
        flag_condition_list = cfg_hrf['stim_lst']
        
        for flag_hbo in flag_hbo_list:
            
            for flag_brain in flag_brain_list: 
                
                for flag_condition in flag_condition_list:
                    
                    for flag_img in flag_img_list:
                        
                        if flag_hbo in ['hbo', 'HbO']:
                            title_str = flag_condition + ' ' + 'HbO'
                            hbx_brain_scalp = 'hbo'
                        else:
                            title_str = flag_condition + ' ' + 'HbR'
                            hbx_brain_scalp = 'hbr'
                        
                        if flag_brain in ['brain', 'Brain']:
                            title_str = title_str + ' brain'
                            hbx_brain_scalp = hbx_brain_scalp + '_brain'
                        else:
                            title_str = title_str + ' scalp'
                            hbx_brain_scalp = hbx_brain_scalp + '_scalp'
                        
                        if len(flag_condition_list) > 1:
                            if flag_img == 'tstat':
                                foo_img = all_trial_X_tstat.sel(trial_type=flag_condition).copy()
                                title_str = title_str + ' t-stat'
                            elif flag_img == 'mag':
                                foo_img = all_trial_Xs_plot.sel(trial_type=flag_condition).copy()
                                title_str = title_str + ' magnitude'
                            elif flag_img == 'noise':
                                foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
                                title_str = title_str + ' noise'
                        else:
                            if flag_img == 'tstat':
                                foo_img = all_trial_X_tstat.copy()
                                title_str = title_str + ' t-stat'
                            elif flag_img == 'mag':
                                foo_img = all_trial_Xs_plot.copy()
                                title_str = title_str + ' magnitude'
                            elif flag_img == 'noise':
                                foo_img = all_trial_X_stderr.copy()
                                title_str = title_str + ' noise'
                
                        if 'reltime' in foo_img.dims:
                            foo_img = foo_img.rename({"reltime": "time"})
                            foo_img = foo_img.transpose("vertex", "chromo", "time")
                        clim = (-foo_img.sel(chromo='HbO').max(), foo_img.sel(chromo='HbO').max())

                        filename = f'IMG_{flag_condition}_{flag_img}_{hbx_brain_scalp}'
                        # create overall folder for current image recon params 
                        save_dir_tmp_ful = os.path.join(save_dir_tmp, folder_name)
                        os.makedirs(save_dir_tmp_ful, exist_ok=True) 


                        save_dir_full = os.path.join(save_dir_tmp_ful, suffix)
                        os.makedirs(save_dir_full, exist_ok=True)
                        save_file_path = os.path.join(save_dir_full, filename )
        
                        print('plotting: ', filename)
                        image_recon_multi_view(   #FIXME: add off_screen option to this function
                            foo_img,  # time series data; can be 2D (static) or 3D (dynamic)
                            head,
                            cmap='jet',
                            clim=clim,
                            view_type=hbx_brain_scalp,
                            title_str=f'{filename} / uM',
                            filename=save_file_path,
                            SAVE=True,
                            #time_range=(foo_img.time.values[0],foo_img.time.values[-1],0.5)*units.s,
                            fps=12,
                            geo3d_plot = None, #  geo3d_plot
                            wdw_size = (1024, 768)
                        )
                     #              
                     
    


#%%
def main():    
    # get params
    cfg_img_recon = snakemake.params.cfg_img_recon
    cfg_hrf = snakemake.params.cfg_hrf
    
    hrf_data = snakemake.input.hrf_data
    Adot_path = snakemake.input.Adot
    SB_path = snakemake.input.SB
    root_dir = snakemake.params.root_dir
    derivatives_subfolder = snakemake.params.derivatives_subfolder

    Adot_path = str(Adot_path) if not isinstance(Adot_path, str) else Adot_path
    
    out = snakemake.output[0]
    
    img_recon_func(cfg_img_recon, cfg_hrf, hrf_data, Adot_path, out, SB_path, root_dir, derivatives_subfolder)
    
            
if __name__ == "__main__":
    main()


