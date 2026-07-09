#%% Import necessary libraries
import os
import re
import pickle
import gzip
import time_series_gui
from tkinter import filedialog
import tkinter as tk
#%% Define paths and extract data information

# ========================================
# STEP 1: Select BIDS root directory
# ========================================
print("\n" + "="*70)
print("STEP 1: SELECT BIDS ROOT DIRECTORY")
print("="*70)
print("Please select the BIDS root directory (where sub-XXX folders are located)")
print("(The dialog window may be in the background - check your taskbar)")
print("="*70 + "\n")

current_dir = os.getcwd()
root = tk.Tk()
root.withdraw()  # Hide the main window

root_dir = filedialog.askdirectory(
    title="Step 1: Select BIDS Root Directory (where sub-XXX folders are)",
    initialdir=current_dir
)
root.destroy()

# Check if user cancelled the dialog
if not root_dir:
    print("No directory selected. Exiting...")
    exit()

print(f"✓ BIDS root directory: {root_dir}")

# Change working directory to root_dir
os.chdir(root_dir)
print(f"✓ Changed working directory to: {os.getcwd()}")

# ========================================
# STEP 2: Check/create derivatives/cedalion
# ========================================
print("\n" + "="*70)
print("STEP 2: CHECKING DERIVATIVES STRUCTURE")
print("="*70)

derivatives_path = os.path.join(root_dir, 'derivatives')
cedalion_path = os.path.join(derivatives_path, 'cedalion')

if not os.path.exists(cedalion_path):
    print(f"Creating: {cedalion_path}")
    os.makedirs(cedalion_path, exist_ok=True)
    print("✓ Created derivatives/cedalion/ directory")
else:
    print(f"✓ Found existing: {cedalion_path}")

# ========================================
# STEP 3: Select/create pipeline folder
# ========================================
print("\n" + "="*70)
print("STEP 3: SELECT OR CREATE PIPELINE FOLDER")
print("="*70)
print("Select an existing pipeline folder OR create a new one")
print("inside 'derivatives/cedalion/'")
print("(The dialog window may be in the background - check your taskbar)")
print("="*70 + "\n")

root = tk.Tk()
root.withdraw()  # Hide the main window

path_to_data = filedialog.askdirectory(
    title="Step 3: Select or Create Pipeline Folder in derivatives/cedalion/",
    initialdir=cedalion_path
)
root.destroy()

# Check if user cancelled the dialog
if not path_to_data:
    print("No pipeline folder selected. Exiting...")
    exit()

print(f"✓ Pipeline folder: {path_to_data}")

# Verify the selected folder is within derivatives/cedalion
# Normalize paths for cross-platform comparison
normalized_path_to_data = os.path.normpath(path_to_data)
normalized_cedalion_path = os.path.normpath(cedalion_path)

if not normalized_path_to_data.startswith(normalized_cedalion_path):
    print("\n⚠️  WARNING: Selected folder is not inside derivatives/cedalion/")
    print(f"   Selected: {normalized_path_to_data}")
    print(f"   Expected to be inside: {normalized_cedalion_path}")
    print("   Continuing anyway, but this may cause issues...")

print("\n" + "="*70)
print("FOLDER SETUP COMPLETE")
print("="*70)
print(f"BIDS Root:       {root_dir}")
print(f"Pipeline Folder: {path_to_data}")
print("="*70 + "\n")

# Construct the path to the preprocessed data
if os.path.basename(path_to_data) == 'preprocessed_data':
    preprocessed_path = path_to_data
else:
    preprocessed_path = os.path.join(path_to_data, 'Outputs', 'preprocessed_data')

# List to store the extracted information
all_data_info = []

# Scan for SNIRF files in root_dir
print(f"Scanning for SNIRF files in: {root_dir}")

for subject_folder in os.listdir(root_dir):
    if subject_folder.startswith('sub-'):
        subject_path = os.path.join(root_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        subject_name = subject_folder.replace('sub-', '')
        
        # Check for session folders
        session_folders = [d for d in os.listdir(subject_path) 
                          if d.startswith('ses-') and os.path.isdir(os.path.join(subject_path, d))]

        if session_folders:
            # Sessions exist, iterate through them
            for session_folder in session_folders:
                session_path = os.path.join(subject_path, session_folder)
                session_name = session_folder.replace('ses-', '')
                
                # Look for nirs folder
                nirs_path = os.path.join(session_path, 'nirs')
                if os.path.isdir(nirs_path):
                    for filename in os.listdir(nirs_path):
                        if filename.endswith('.snirf'):
                            # Extract task and run from filename
                            task_match = re.search(r'task-([^_]+)', filename)
                            run_match = re.search(r'run-([^_]+)', filename)
                            
                            task_name = task_match.group(1) if task_match else None
                            run_name = run_match.group(1) if run_match else None
                            
                            snirf_file_path = os.path.join(nirs_path, filename)
                            
                            # Construct expected preprocessed SNIRF path
                            pkl_filename = snirf_filename.replace('.snirf', '_preprocessed.snirf')
                            if session_folder:
                                pkl_path = os.path.join(preprocessed_path, subject_folder, session_folder, pkl_filename)
                            else:
                                pkl_path = os.path.join(preprocessed_path, subject_folder, pkl_filename)
                            
                            # Check if pkl exists
                            pkl_exists = os.path.exists(pkl_path) if preprocessed_path else False
                            
                            all_data_info.append({
                                'subject': subject_name,
                                'session': session_name,
                                'task': task_name,
                                'run': run_name,
                                'snirf_path': snirf_file_path,
                                'pkl_path': pkl_path if pkl_exists else None
                            })
        else:
            # No sessions, look for nirs folder directly in subject folder
            nirs_path = os.path.join(subject_path, 'nirs')
            if os.path.isdir(nirs_path):
                for filename in os.listdir(nirs_path):
                    if filename.endswith('.snirf'):
                        # Extract task and run from filename
                        task_match = re.search(r'task-([^_]+)', filename)
                        run_match = re.search(r'run-([^_]+)', filename)
                        
                        task_name = task_match.group(1) if task_match else None
                        run_name = run_match.group(1) if run_match else None
                        
                        snirf_file_path = os.path.join(nirs_path, filename)
                        
                        # Construct expected preprocessed SNIRF path
                        pkl_filename = filename.replace('.snirf', '_preprocessed.snirf')
                        pkl_path = os.path.join(preprocessed_path, subject_folder, pkl_filename)
                        
                        
                        # Check if pkl exists
                        pkl_exists = os.path.exists(pkl_path) if preprocessed_path else False
                        
                        all_data_info.append({
                            'subject': subject_name,
                            'session': None,
                            'task': task_name,
                            'run': run_name,
                            'snirf_path': snirf_file_path,
                            'pkl_path': pkl_path if pkl_exists else None
                        })

# Create combined labels and a file map
file_map = {}
combined_subjects_set = set()

# This dictionary will map combined subject labels to a set of their available combined run labels
subject_to_runs_map = {}

print(f"Found {len(all_data_info)} SNIRF files")

for info in all_data_info:
    # Create combined subject label
    subject_part = f"sub-{info['subject']}"
    session_part = f"ses-{info['session']}" if info['session'] else None
    combined_subject = f"{subject_part}_{session_part}" if session_part else subject_part
    combined_subjects_set.add(combined_subject)

    # Create combined run label
    run_part = f"run-{info['run']}" if info['run'] else None
    task_part = f"task-{info['task']}" if info['task'] else None
    
    if run_part and task_part:
        combined_run = f"{task_part}_{run_part}"
    elif run_part:
        combined_run = run_part
    elif task_part:
        combined_run = task_part
    else:
        combined_run = 'default' # Fallback

    # Populate the file_map and the subject_to_runs_map
    if combined_subject not in file_map:
        file_map[combined_subject] = {}
        subject_to_runs_map[combined_subject] = set()

    # Store both SNIRF and PKL paths
    file_map[combined_subject][combined_run] = {
        'snirf_path': info['snirf_path'],
        'pkl_path': info['pkl_path']
    }
    subject_to_runs_map[combined_subject].add(combined_run)
    
    if info['pkl_path']:
        print(f"  {combined_subject}/{combined_run}: SNIRF + PKL")
    else:
        print(f"  {combined_subject}/{combined_run}: SNIRF only (no processed data)")

# Convert sets to sorted lists for stable ordering in the GUI
subjects = sorted(list(combined_subjects_set))
for subj in subject_to_runs_map:
    subject_to_runs_map[subj] = sorted(list(subject_to_runs_map[subj]))


print("Combined Subjects:", subjects)
print("Subject to Runs Map:", subject_to_runs_map)

# Create a dictionary to pass to the GUI
gui_data = {
    "subjects": subjects,
    "subject_to_runs_map": subject_to_runs_map,
    "file_map": file_map,
    "path_to_data": path_to_data  # Pass the selected derivatives/cedalion/XXX folder
}

# Visualize the data
if all_data_info:
    print("Starting visualization...")
    time_series_gui.run_vis(gui_data)
else:
    print("No SNIRF files found to process.")    