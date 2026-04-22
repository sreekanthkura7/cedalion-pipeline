# module GLM
# functions to perform GLM

#%% Imports
import cedalion
import cedalion.nirs
import cedalion.sigproc.frequency as frequency
import cedalion.models.glm as glm
import xarray as xr
import matplotlib.pyplot as p
from cedalion import units
import numpy as np
import pandas as pd
from functools import reduce
import operator

#%% Functions

def blockaverage(epochs_all, cfg_hrf_estimation):
    all_trial_blockaverage = None
    # Block Average
    baseline = epochs_all.sel(reltime=(epochs_all.reltime < 0)).mean('reltime')
    epochs = epochs_all - baseline  # baseline subtract
    blockaverage_ep = epochs.groupby('trial_type').mean('epoch') # mean across all epochs 

    epochs_zeromean = epochs - blockaverage_ep   # zeromean the epochs - residual
    
    bad_chans_mse_lst = []
    # LOOP OVER TRIAL TYPES
    for idxt, trial_type in enumerate(cfg_hrf_estimation['stim_lst']): 
        epochs_zeromean_tmp = epochs_zeromean.copy()
        blockaverage_tmp = blockaverage_ep.copy()
        # select current trial type
        epochs_zeromean_tmp = epochs_zeromean_tmp.where(epochs_zeromean_tmp.trial_type == trial_type, drop=True)
        blockaverage1 = blockaverage_tmp.sel(trial_type=trial_type)  # select current trial type
        blockaverage1 = blockaverage1.expand_dims('trial_type')  # readd trial type coord
        blockaverage = blockaverage1.copy()
        
        if 'chromo' in blockaverage.dims:
            epochs_zeromean_tmp = epochs_zeromean_tmp.stack(measurement=['channel','chromo']).sortby('chromo')
            blockaverage = blockaverage.transpose('trial_type', 'channel', 'chromo', 'reltime')
        else:
            epochs_zeromean_tmp = epochs_zeromean_tmp.stack(measurement=['channel','wavelength']).sortby('wavelength')
            blockaverage = blockaverage.transpose('trial_type', 'channel', 'wavelength', 'reltime')
    
        n_epochs = len(epochs_zeromean_tmp.epoch)

        epochs_zeromean_tmp = epochs_zeromean_tmp.transpose('trial_type', 'measurement', 'reltime', 'epoch')  
        # calc mse
        mse_t = (epochs_zeromean_tmp**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean
                
        # retrieve channels where mse_t = 0
        bad_mask = mse_t.sel(trial_type=trial_type).data == 0
        bad_any = bad_mask.any(axis=1)
        bad_chans_mse = mse_t.channel[bad_any].values
    
        bad_chans_mse_lst.append(bad_chans_mse)

        if 'measurement' in mse_t.dims and isinstance(mse_t.get_index('measurement'), pd.MultiIndex):
            mse_t = mse_t.unstack('measurement')
        
        if all_trial_blockaverage is None:
            all_trial_blockaverage = blockaverage
            all_trial_mse = mse_t
        else:
            all_trial_blockaverage = xr.concat([all_trial_blockaverage, blockaverage], dim='trial_type') 
            all_trial_mse = xr.concat([all_trial_mse, mse_t], dim='trial_type')

    # DONE LOOP OVER TRIAL_TYPES

    return all_trial_blockaverage, all_trial_mse, bad_chans_mse_lst


def GLM(runs, cfg_hrf_estimation, geo3d, pruned_chans_list):
    cfg_GLM = cfg_hrf_estimation['GLM']
    rec_str = cfg_hrf_estimation['rec_str']

    # 1. need to concatenate runs 
    Y_all, stim_df_tmp, runs_updated = concatenate_runs(runs, rec_str)

    target_units = Y_all.pint.units # grab units from data 
    target_units_time = str(Y_all.time.attrs['units'])

    # grab only trial type of interest
    stim_df = stim_df_tmp[stim_df_tmp['trial_type'].isin(cfg_hrf_estimation['stim_lst'])].reset_index(drop=True)
    
    # basis function options
    BASIS_FUNCTIONS = {
    "gaussian_kernels": glm.GaussianKernels,
    #"gaussian_kernels_with_tails": glm.GaussianKernelsWithTails, #FIXME: not currently in glm code?
    "gamma": glm.Gamma,
    # "gamma_deriv": glm.GammaDeriv, #FIXME: not currently in glm code?
    # "afni_gamma": glm.AFNIGamma, #FIXME: not currently in glm code?
    "dirac_delta": glm.DiracDelta, 
    }
    
    basis_cls = BASIS_FUNCTIONS[cfg_GLM["basis_func"]] # basis func class
    if basis_cls is None:
        raise ValueError(f"Unsupported basis function: {cfg_GLM['basis_func']}")
    if cfg_GLM["basis_func_params"] is None:
        raise ValueError(f"Missing parameters for basis function: {cfg_GLM['basis_func']}")

    if cfg_GLM['basis_func'] == "gaussian_kernels_with_tails" or cfg_GLM['basis_func'] == "gaussian_kernels":
        t_pre = cfg_hrf_estimation['t_pre']
        t_post = cfg_hrf_estimation['t_post']
    basis_params = cfg_GLM["basis_func_params"] # basis func params
    basis_func = basis_cls(t_pre=t_pre, t_post=t_post, **basis_params)  # basis function with params

    # 2. define design matrix
    dms = glm.design_matrix.hrf_regressors(
                                    Y_all,
                                    stim_df,
                                    basis_func
                                )

    # Combine drift and short-separation regressors (if any)
    if cfg_GLM['do_drift'] == 'polynomial': 
        drift_regressors = get_drift_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    elif cfg_GLM['do_drift'] == 'legendre':  
        drift_regressors = get_drift_legendre_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors) # adds iteratively to dm 

    if cfg_GLM['do_short_sep']:
        ss_regressors = get_short_regressors(runs_updated, pruned_chans_list, geo3d, cfg_GLM)
        dms &= reduce(operator.and_, ss_regressors)

    dms.common = dms.common.fillna(0)

    # 3. get betas and covariance
    results = glm.fit(Y_all, dms, noise_model=cfg_GLM['noise_model'])  # fit GLM to get betas and covariance
    betas = results.sm.params  # this is the beta estimates for each regressor in the design matrix, it has dimensions regressor and measurement, 
    cov_params = results.sm.cov_params() # this is the covariance of the beta estimates, which we can use to get MSE of the HRF estimate. It has dimensions regressor_r and regressor_c, ctions

    # 4. estimate HRF and MSE
    basis_hrf = basis_func(Y_all)

    trial_type_list = cfg_hrf_estimation['stim_lst']

    hrf_mse_list = []
    hrf_estimate_list = []
    bad_chans_mse_lst = []

    for trial_type in trial_type_list:
        print(trial_type)
        betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
        hrf_estimate = estimate_HRF_from_beta(betas_hrf, basis_hrf)
        
        cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                            regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
                                    )
        hrf_mse = estimate_HRF_cov(cov_hrf, basis_hrf)

        # get bad mse channels 
        if 'chromo' in hrf_mse.dims:
            bad_mask = (hrf_mse == 0).any(dim=["time", "chromo"])
        else:
            bad_mask = (hrf_mse == 0).any(dim=["time", "wavelength"])
        bad_chans_mse = hrf_mse.channel[bad_mask].values

        hrf_estimate = hrf_estimate.expand_dims({'trial_type': [ trial_type ] })
        hrf_mse = hrf_mse.expand_dims({'trial_type': [ trial_type ] })

        hrf_estimate_list.append(hrf_estimate)
        hrf_mse_list.append(hrf_mse)
        bad_chans_mse_lst.append(bad_chans_mse)

    hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
    hrf_estimate = hrf_estimate.pint.quantify(target_units)

    hrf_mse = xr.concat(hrf_mse_list, dim='trial_type')
    hrf_mse = hrf_mse.pint.quantify(target_units**2)  

    # set universal time so that all hrfs have the same time base 
    fs = frequency.sampling_rate(runs[0][rec_str]).to('Hz')
    before_samples = int(np.ceil((cfg_hrf_estimation['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_hrf_estimation['t_post'] * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = target_units_time 

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = target_units_time  

    return results, hrf_estimate, hrf_mse, bad_chans_mse_lst


def estimate_HRF_cov(cov, basis_hrf):

    basis_hrf = basis_hrf.rename({'component':'regressor_c'})
    basis_hrf = basis_hrf.assign_coords(regressor_c=cov.regressor_c.values)

    tmp = xr.dot(cov, basis_hrf, dims='regressor_c')

    tmp = tmp.rename({'regressor_r':'regressor'})
    basis_hrf = basis_hrf.rename({'regressor_c':'regressor'})

    mse_t = xr.dot(basis_hrf, tmp, dims='regressor')

    return mse_t

def estimate_HRF_from_beta(betas, basis_hrf):
        
    basis_hrf = basis_hrf.rename({'component':'regressor'})
    basis_hrf = basis_hrf.assign_coords(regressor=betas.regressor.values)

    hrf_estimate = xr.dot(betas, basis_hrf, dims='regressor')

    hrf_estimates_blcorr = hrf_estimate - hrf_estimate.sel(time = hrf_estimate.time[hrf_estimate.time<0]).mean('time')

    return hrf_estimates_blcorr

def get_drift_regressors(runs, cfg_GLM):
    
    drift_regressors = []
    i=0
    for i, run  in enumerate(runs):

        drift = glm.design_matrix.drift_regressors(run, cfg_GLM['drift_order'])
        drift.common = drift.common.assign_coords({'regressor': [f'Drift {x} run {i}' for x in range(cfg_GLM['drift_order']+1)]})
        drift_regressors.append(drift)
        
    return drift_regressors

def get_drift_legendre_regressors(runs, cfg_GLM):

    drift_regressors = []
    i=0
    for i, run  in enumerate(runs):

        drift = glm.design_matrix.drift_legendre_regressors(run, cfg_GLM['drift_order'])
        drift.common = drift.common.assign_coords({'regressor': [f'Drift {x} run {i}' for x in range(cfg_GLM['drift_order']+1)]})
        drift_regressors.append(drift)

    return drift_regressors

def get_short_regressors(runs, pruned_chans_list, geo3d, cfg_GLM):
    ss_regressors = []
    i=0
    for run, pruned_chans in zip(runs, pruned_chans_list):

        rec_pruned = prune_mask_ts(run, pruned_chans) # !!! how is this affected when using pruned data
        ts_long, ts_short = cedalion.nirs.split_long_short_channels(
                                rec_pruned, geo3d, distance_threshold= cfg_GLM['distance_threshold']  # !!! change to rec_pruned once NaN prob fixed
                                )
        # SSR options
        SSR_method = {
        "average": glm.design_matrix.average_short_channel_regressor(ts_short),
        "mean":  glm.design_matrix.average_short_channel_regressor(ts_short),
        "closest": glm.design_matrix.closest_short_channel_regressor(ts_long, ts_short, geo3d), 
        #'max_corr': glm.design_matrix.max_corr_short_channel_regressor(ts_long, ts_short), #FIXME: fails with NaNs
        }

        short = SSR_method[cfg_GLM['short_channel_method']]
        short.common = short.common.reset_coords('samples', drop=True)
        short.common = short.common.assign_coords({'regressor': [f'short run {i}']})
        ss_regressors.append(short)
        i = i+1

    return ss_regressors

def concatenate_runs(runs, rec_str):

    CURRENT_OFFSET = 0
    runs_updated = []
    stim_updated = []

    for rec in runs:

        ts = rec[rec_str]
        time = ts.time.values
        new_time = time + CURRENT_OFFSET

        ts_new = ts.copy(deep=True)
        ts_new = ts_new.pint.to('molar')
        ts_new = ts_new.assign_coords(time=new_time)

        stim = rec.stim
        stim_shift = stim.copy()
        stim_shift['onset'] += CURRENT_OFFSET

        stim_updated.append(stim_shift)
        runs_updated.append(ts_new)

        CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])  # updating time offset

    Y_all = xr.concat(runs_updated, dim='time')
    Y_all.time.attrs['units'] = ts.time.units 
    stim_df = pd.concat(stim_updated, ignore_index = True)

    return Y_all, stim_df, runs_updated


def prune_mask_ts(ts, pruned_chans):
    '''
    Function to mask pruned channels with NaN .. essentially repruning channels
    Parameters
    ----------
    ts : data array
        time series from rec[rec_str].
    pruned_chans : list or array
        list or array of channels that were pruned prior.

    Returns
    -------
    ts_masked : data array
        time series that has been "repruned" or masked with data for the pruned channels as NaN.

    '''
    mask = np.isin(ts.channel.values, pruned_chans)
    
    if ts.ndim == 3 and ts.shape[0] == len(ts.channel):
        mask_expanded = mask[:, None, None]  # (chan, wav, time)
    elif ts.ndim == 3 and ts.shape[1] == len(ts.channel):
        mask_expanded = mask[None, :, None]  # (chrom, chan, time)
    else:
        raise ValueError("Expected input shape to be either (chan, dim, time) or (dim, chan, time)")

    ts_masked = ts.where(~mask_expanded, np.nan)
    return ts_masked



