# Script to run forward model and calculate sensitivity matrix of your probe
#%% Imports

import os
import cedalion
import cedalion.dot as dot
import cedalion.io as io
import glob


def generate_Adot_func(cfg_Adot, root_dir, sub, head_model, save_dir_Adot):
    #%%
    # Load recording obj from first subject/task/run
    #task = task[0]  # assuming only 1 task is listed
    snirf_path = find_first_snirf(root_dir, sub)
    #snirf_path = os.path.join(root_dir, f"sub-{sub}", "nirs", f"sub-{sub}_task-{task}_run-01_nirs.snirf")
    recordings = io.read_snirf(snirf_path)
    rec = recordings[0]
    geo3d_meas = rec.geo3d
    meas_list = rec._measurement_lists["amp"]

    # Load head model
    head = dot.get_standard_headmodel(head_model)
    # head_ras = head.apply_transform(head.t_ijk2ras) # change between coord systems

    geo3d_snapped_ijk = head.align_and_snap_to_scalp(geo3d_meas) # optode registration, snap optodes to nearest vertex on scalp

    # Construct forward model
    fwm = dot.ForwardModel(head, geo3d_snapped_ijk, meas_list)

    #%% Run the simulation
    save_dir_fl = save_dir_Adot.split("sensitivity")[0]

    # calculate fluence
    print ('Calculating fluence')
    fluence_fname = os.path.join(save_dir_fl, "fluence.h5")

    if cfg_Adot['forward_model'] == "MCX":
        fwm.compute_fluence_mcx(fluence_fname)
    elif cfg_Adot['forward_model'] == "NIRFASTER":
        fwm.compute_fluence_nirfaster(fluence_fname)


    # Calculate the sensitivity matrix
    print('Calculating the sensitivity matrix')
    sensitivity_fname = os.path.join(save_dir_Adot)
    fwm.compute_sensitivity(fluence_fname, sensitivity_fname)


def find_first_snirf(root_dir, sub):
    """Discover any available snirf run for this subject without relying on config task."""
    pattern = os.path.join(root_dir, f"sub-{sub}", "nirs", f"sub-{sub}_task-*_run-01_nirs.snirf")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No snirf files found for sub-{sub} under {root_dir}")
    return matches[0]  # use the first one found


#%%

def main():
    
    # get params
    cfg_Adot = snakemake.params.cfg_Adot
    root_dir = snakemake.params.root_dir
    head_model = snakemake.params.head_model
    #task = snakemake.params.task

    dirs = sorted(d.replace("sub-", "")for d in os.listdir(root_dir)if d.startswith("sub-")) # get list of subject folders
    sub = dirs[0] # grab first subject

    out_Adot = snakemake.output.Adot
    
    generate_Adot_func(cfg_Adot, root_dir, sub, head_model, out_Adot)
    
            
if __name__ == "__main__":
    main()

