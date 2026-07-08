# -*- coding: utf-8 -*-
"""

This function will load all the data for the specified subject and file IDs, and preprocess the data.
This function will also create several data quality report (DQR) figures that are saved in /derivatives/plots.
The function will return the preprocessed data and a list of the filenames that were loaded, both as 
two dimensional lists [subj_idx][file_idx].
The data is returned as a recording container with the following fields:
  timeseries - the data matrices with dimensions of ('channel', 'wavelength', 'time') 
     or ('channel', 'HbO/HbR', 'time') depending on the data type. 
     The following sub-fields are included:
        'amp' - the original amplitude data slightly processed to remove negative and NaN values and to 
           apply a 3 point median filter to remove outliers.
        'amp_pruned' - the 'amp' data pruned according to the SNR, SD, and amplitude thresholds.
        'od' - the optical density data
        'od_tddr' - the optical density data after TDDR motion correction is applied
        'conc_tddr' - the concentration data obtained from 'od_tddr'
        'od_splineSG' and 'conc_splineSG' - returned if splineSG motion correction is applied (i.e. flag_do_splineSG=True)
  stim - the stimulus data with 'onset', 'duration', and 'trial_type' fields and more from the events.tsv files.
  aux_ts - the auxiliary time series data from the SNIRF files.
     In addition, the following aux sub-fields are added during pre-processing:
        'gvtd' - the global variance of the time derivative of the 'od' data.
        'gvtd_tddr' - the global variance of the time derivative of the 'od_tddr' data.
        
        
Created on Tue Jun  3 11:39:55 2025

@author: shank
"""

import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.motion as motion_correct
import xarray as xr
import numpy as np
import pandas as pd
import pint
from cedalion.physunits import units
import json
import pickle
import gzip

import sys
# sys.path.append('scripts/modules/')
script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append("/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/modules/")
modules_path = os.path.join(script_dir, 'modules')
sys.path.append(modules_path)

import module_plot_DQR as plot_dqr
import module_imu_glm_filter as imu_filt
import module_preprocess as preproc

#%% Load in data for current subject/task/run

def _parse_dpf_values(dpf_config, wavelengths):
    """Parse DPF values from YAML and align them to the wavelength coordinate."""
    if dpf_config is None:
        dpf_config = [1, 1]

    if not isinstance(dpf_config, (list, tuple)):
        dpf_config = [dpf_config]

    dpf_values = [float(value) for value in dpf_config]
    if len(dpf_values) == 1:
        dpf_values = dpf_values * len(wavelengths)

    if len(dpf_values) != len(wavelengths):
        raise ValueError(
            f"DPF must have one value per wavelength: got {len(dpf_values)} "
            f"values for {len(wavelengths)} wavelengths."
        )

    return dpf_values


def preprocess_func(snirf_path, events_path, root_dir, derivatives_subfolder, cfg_preprocess, stim_lst, out_files):
    cedalion.xrutils.unit_stripping_is_error(True)
    # Load in snirf file
    
    records = cedalion.io.read_snirf( snirf_path, time_units = 'second') #FIXME: HARD CODED TIME UNITS
    rec = records[0]

    # Load in events.tsv file 
    if not os.path.exists( events_path ):  
        print( f"Error: File {events_path} does not exist" )
    else:
        stim_df = pd.read_csv( events_path, sep='\t' )
        rec.stim = stim_df

    # Load in json sidecar file if it exists
    file_json_path = snirf_path.replace('.snirf', '.json')
    if not os.path.exists( file_json_path ):  
        print( f"File {file_json_path} does not exist" )
    else:
        with open( file_json_path, 'r' ) as f:
            file_json = json.load(f)
    
    # get filename for plots
    filename = os.path.basename(snirf_path)
    filnm, _ext = os.path.splitext(filename)
    
    # Replace negatives and nans with a small pos value
    rec['amp'] = rec['amp'].where( rec['amp']>0, 1e-18 ) 
    rec['amp'] = rec['amp'].where( ~rec['amp'].isnull(), 1e-18 ) 

    # if first value is 1e-18 then replace with second value
    indices = np.where(rec['amp'][:,0,0] == 1e-18)
    rec['amp'][indices[0],0,0] = rec['amp'][indices[0],0,1]
    indices = np.where(rec['amp'][:,1,0] == 1e-18)
    rec['amp'][indices[0],1,0] = rec['amp'][indices[0],1,1]
    
    #%% Preprocess
    
    for step_name, params in cfg_preprocess["steps"].items():
        
        # If this step is disabled, skip it
        if not (params.get("enable", False))  and (step_name != "prune"): 
            continue
       
        if step_name == "bs_preproc":   # !!! only for BS data 
            recordings = cedalion.io.read_snirf(params['probe_dir'] + params['snirf_name_probe'])
            rec = recordings[0]
            meas_list = rec._measurement_lists['amp']
            chans = list(dict.fromkeys(meas_list.channel))  # select chans we care about from probe snirf meas list
            rec['amp'] = rec['amp'].sel(channel=chans) 
            
        elif step_name == "median_filter":
            rec = preproc.median_filt( rec, params['order'] )
        
        elif step_name == "prune":
            # change from str to pint object
            cfg_preprocess['steps']['prune']["sd_thresh_min"] = units(cfg_preprocess['steps']['prune']["sd_thresh_min"]) 
            cfg_preprocess['steps']['prune']["sd_thresh_max"] = units(cfg_preprocess['steps']['prune']["sd_thresh_max"])
            cfg_preprocess['steps']['prune']["window_length"] = units(cfg_preprocess['steps']['prune']["window_length"])
            if isinstance(cfg_preprocess['steps']['prune']["amp_thresh_min"], str):
                cfg_preprocess['steps']['prune']["amp_thresh_min"] = float(cfg_preprocess['steps']['prune']["amp_thresh_min"])
            if isinstance(cfg_preprocess['steps']['prune']["amp_thresh_max"], str):
                cfg_preprocess['steps']['prune']["amp_thresh_max"] = float(cfg_preprocess['steps']['prune']["amp_thresh_max"])            

            rec, chs_pruned = preproc.pruneChannels( rec, cfg_preprocess['steps']['prune'])
            pruned_chans = chs_pruned.where(chs_pruned != 0.58, drop=True).channel.values # get array of channels that were pruned
            
            
        # Calculate OD 
        # if flag pruned channels is True, then do rest of preprocessing on pruned amp, if not then do preprocessing on unpruned data
        elif step_name == "int2od":
            if cfg_preprocess['steps']['prune']['enable']:
                rec["od"] = cedalion.nirs.cw.int2od(rec['amp_pruned'])                
            else:
                rec["od"] = cedalion.nirs.cw.int2od(rec['amp'])
            
            rec["od_corrected"] = rec["od"]
            units_od = rec["od"].pint.units

            # calc slope and gctd of od ts before any correction
            slope_base = preproc.quant_slope(rec, "od") # Get the slope of 'od' before motion correction and bpf 
            rec.aux_ts["gvtd"], _ = quality.gvtd(rec['amp_pruned'])  # Calculate GVTD on pruned data
            rec.aux_ts["gvtd"].name = "gvtd"
        
        # # Walking filter 
        # elif step_name == 'imu_glm': 
        #     rec["od_corrected"] = imu_filt.filterWalking(rec, "od", params, filnm, cfg_dataset) 
            
            
        #%% MOTION CORRECTION: 
        # tddr
        elif step_name in ("tddr", "motion_correct_tddr"):
            rec['od_corrected'] = motion_correct.tddr( rec['od_corrected'] )  
            rec['od_corrected'] = rec['od_corrected'].where( ~rec['od_corrected'].isnull(), 1e-18 ) # 0  # replace any NaNs after TDDR # !!! make a step?
        
            
        # !!! add spline
        # if step_name in ("spline", "motion_correct_spline"):
        #     if 'od_corrected' in rec.timeseries.keys():
        #         rec['od_corrected'] = motion_correct.motion_correct_spline( rec['od_corrected'] )  
        #     else:   # do tddr on uncorrected od
        #         rec['od_corrected'] = motion_correct.motion_correct_spline( rec['od'] ) 
        
        
        # splineSG
        elif step_name in ("splineSG", "motion_correct_splineSG"):
            rec['od_corrected'] = motion_correct.motion_correct_splineSG( rec['od_corrected'], params['p'], params['frame_size'] )  
            
        
        # !!! add PCA
        # elif step_name in ("PCA", "motion_correct_PCA"):
        
        # pca recurse
        elif step_name in ("PCA_recurse", "motion_correct_PCA_recurse"):
            rec['od_corrected'] = motion_correct.motion_correct_PCA_recurse( rec['od_corrected'], params['t_motion'], 
                                                                                   params['t_mask'],  params['stdev_thresh'], params['amp_thresh'],
                                                                                   params['nSV'], params['maxIter'])  
        
        
        # wavelet
        elif step_name in ("wavelet", "motion_correct_wavelet"):
            rec['od_corrected'] = motion_correct.motion_correct_PCA_recurse( rec['od_corrected'], params['iqr'], 
                                                                                   params['wavelet'], params['level'])
    

        #%%
        
        # Bandpass filter od
        elif step_name == "freq_filter":
            # change from str to pint object
            cfg_preprocess["steps"]["freq_filter"]["fmin"] = units(cfg_preprocess["steps"]["freq_filter"]["fmin"])
            cfg_preprocess["steps"]["freq_filter"]["fmax"] = units(cfg_preprocess["steps"]["freq_filter"]["fmax"])
            
            rec['od_unfiltered'] = rec['od_corrected'].copy() # save unfiltered od for dqr

            rec['od_corrected'] = cedalion.sigproc.frequency.freq_filter(rec['od_corrected'], 
                                                                            params['fmin'], 
                                                                            params['fmax'])  
        # Convert OD to Conc
        elif step_name == "od2conc":
            dpf_values = _parse_dpf_values(params.get("dpf", [1, 1]), rec['amp'].wavelength)
            dpf = xr.DataArray(
                dpf_values,
                dims="wavelength",
                coords={"wavelength": rec['amp'].wavelength},
            )
            print(f"Using DPF values for od2conc: {dpf_values}")
            rec['conc'] = cedalion.nirs.cw.od2conc(rec['od_corrected'], rec.geo3d, dpf, spectrum="prahl")
        
        
        # Plot DQR
        elif step_name in ("DQR_plot", "plot_DQR", "plot_dqr", "dqr_plot"):
            lambda0 = rec['amp_pruned'].wavelength[0].wavelength.values
            lambda1 = rec['amp_pruned'].wavelength[1].wavelength.values
            snr0, _ = quality.snr(rec['amp_pruned'].sel(wavelength=lambda0), cfg_preprocess['steps']["prune"]['snr_thresh'])
            snr1, _ = quality.snr(rec['amp_pruned'].sel(wavelength=lambda1), cfg_preprocess['steps']["prune"]['snr_thresh'])
            snr0 = np.nanmedian(snr0.pint.dequantify().values)
            snr1 = np.nanmedian(snr1.pint.dequantify().values)

            # calc slope and gvtd after correction but before bandpass
            if 'od_unfiltered' in rec.timeseries.keys():
                ts_name = 'od_unfiltered'
            else:
                ts_name = 'od_corrected'
            rec = preproc.get_gvtd(rec, pruned_chans, ts_name, 'gvtd_corrected')    
            slope_corrected = preproc.quant_slope(rec, ts_name)  # Get slopes after correction before bandpass filtering
            if 'od_unfiltered' in rec.timeseries.keys():
                del rec.timeseries["od_unfiltered"]

            plot_dqr.plotDQR( rec, chs_pruned, cfg_preprocess['steps'], filnm, root_dir, derivatives_subfolder, stim_lst) #, out_files['out_dqr'], out_files['out_gvtd'] )
            
            # if MA correction was performed, plot slope b4 and after
            if not (rec["od_corrected"].data == rec["od"].data).all():
                plot_dqr.plot_slope(rec, [slope_base, slope_corrected], cfg_preprocess['steps'], filnm, root_dir, derivatives_subfolder) #, out_files['out_slope'])
            if os.path.exists( file_json_path ):
                plot_dqr.plotDQR_sidecar(file_json, rec, root_dir, derivatives_subfolder, filnm)
    
        # If you get here, that means this step is enabled but unrecognized. 
        else:
            raise ValueError(f"Unknown preprocessing step: {step_name}")
            
    if not rec['od_corrected'].pint.units:
        rec['od_corrected'] = rec['od_corrected'].pint.quantify(units_od)  # make sure od has units 
        
    min_amp_thresh = cfg_preprocess['steps']['prune']["amp_thresh_min"]
    if isinstance(min_amp_thresh,str):
        min_amp_thresh = float(min_amp_thresh)  
    idx_sat = np.where(chs_pruned == 0.92)[0]
    sat_ch_coords = chs_pruned.channel[idx_sat].values  # get channel coords
    amp = rec['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
    idx_amp = np.where(amp < min_amp_thresh)[0]   
    amp_ch_coords = chs_pruned.channel[idx_amp].values

    # Concat bad indices
    bad_channels = np.unique(np.concat([amp_ch_coords, sat_ch_coords]))

    # build a data quality xr dataset 
    ds = xr.Dataset()
    if cfg_preprocess["steps"]["plot_dqr"]["enable"]:  # only add snr,gvtd_corr, and slope_corr if DQR is plotted, since snr is calculated in that step
        ds.attrs['snr0'] = float(snr0)
        ds.attrs['snr1'] = float(snr1)
        ds['gvtd_corrected'] = rec.aux_ts['gvtd_corrected'].pint.dequantify()
        ds = xr.merge([ds,slope_base.rename({'slope': 'slope_base'}),
                    slope_corrected.rename({'slope': 'slope_corrected'})])
    ds['chs_pruned'] = chs_pruned  
    ds['bad_channels'] = xr.DataArray(bad_channels, dims='bad_channels') # both sat and amp bad channels
    ds['pruned_channels'] = xr.DataArray(pruned_chans, dims='pruned_channels')  # same as chs_pruned but just the coordinates of bad channels 
    ds['bad_chans_amp'] = xr.DataArray(amp_ch_coords, dims='bad_chans_amp') # make xarrays and add
    ds['bad_chans_sat'] = xr.DataArray(sat_ch_coords, dims='bad_chans_sat')
    ds['idx_amp'] = xr.DataArray(idx_amp, dims='idx_amp')
    ds['idx_sat'] = xr.DataArray(idx_sat, dims='idx_sat')
    ds['gvtd_base']      = rec.aux_ts['gvtd'].pint.dequantify()
    geo2d_clean = rec.geo2d.pint.dequantify().rename({'pos': 'pos2d'}) # dequant to save, and rename pos to pos2d to avoid confusion with geo3d pos coords
    geo2d_clean['type'] = geo2d_clean['type'].astype(str) # convert type to str
    ds['geo2d'] = geo2d_clean
    geo3d_clean = rec.geo3d.pint.dequantify().rename({'pos': 'pos3d'}) # dequant to save, and rename pos to pos3d to avoid confusion with geo2d pos coords
    geo3d_clean['type'] = geo3d_clean['type'].astype(str) # convert type to str
    ds['geo3d'] = geo3d_clean

    # save quality info as netcdf file
    ds.to_netcdf(out_files['out_sidecar'],  mode='w')


    # Publish only user-facing preprocessed series. Cedalion's SNIRF reader maps
    # repeated dOD/amplitude data blocks to od_02/amp_02, so store the final OD
    # as "od" and keep pruning details in the quality sidecar.
    if 'od_corrected' in rec.timeseries:
        rec['od'] = rec['od_corrected']
        del rec.timeseries['od_corrected']
    if 'amp_pruned' in rec.timeseries:
        del rec.timeseries['amp_pruned']
    if 'od_unfiltered' in rec.timeseries:
        del rec.timeseries['od_unfiltered']

    # # Save preprocessed data as a snirf file
    cedalion.io.snirf.write_snirf(out_files['out_snirf'], rec)  
        # this has a bug - units for time do not save

    # file = gzip.GzipFile(out_files['out_snirf'], 'wb')
    # file.write(pickle.dumps([rec]))
    
    print("Snirf file saved successfuly")



#%% 
def main():    
    snirf_path = snakemake.input.snirf
    events_path = snakemake.input.events
    
    root_dir = snakemake.params.root_dir
    derivatives_subfolder = snakemake.params.derivatives_subfolder
    cfg_preprocess = snakemake.params.cfg_preprocess
    stim_lst = snakemake.params.stim_lst    
    
    out_files = {
        "out_snirf" : snakemake.output.snirf,
        "out_sidecar": snakemake.output.sidecar,
        }
    
    preprocess_func(snirf_path, events_path, root_dir, derivatives_subfolder, cfg_preprocess, stim_lst, out_files)
 
    
if __name__ == "__main__":
    main()

