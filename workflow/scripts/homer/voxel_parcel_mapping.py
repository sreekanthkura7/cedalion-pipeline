import cedalion.data
import numpy as np
import nibabel as nib
from scipy import sparse
import os

print("Loading Colin27 head model and parcellation data...")
print("="*60)

# Load head model files
hmfiles = cedalion.data.get_colin27_headmodel_files()

# Load FreeSurfer directory
fs_dir = cedalion.data.get_colin27_freesurfer_directory()

# Load voxel-to-vertex mapping
print("\n1. Loading voxel-to-vertex mapping...")
v2v = hmfiles.load_voxel_to_vertex_mapping()
print(f"   Voxel-to-vertex matrix shape: {v2v.shape}")
print(f"   Maps {v2v.shape[0]:,} voxels to {v2v.shape[1]:,} vertices")
print(f"   Non-zero entries: {v2v.nnz:,}")
print(f"   Matrix format: {type(v2v).__name__}")

# Convert to CSC format for efficient column access
print("   Converting to CSC format for efficient vertex lookup...")
v2v_csc = v2v.tocsc()

# Load gray matter mask to get voxel dimensions
print("\n2. Loading gray matter mask for voxel coordinates...")
gm_mask_file = os.path.join(hmfiles.basedir, hmfiles.mask_files['gm'])
gm_nii = nib.load(gm_mask_file)
gm_data = gm_nii.get_fdata()
print(f"   Gray matter volume shape: {gm_data.shape}")
print(f"   Affine transformation:\n{gm_nii.affine}")

# Load parcel annotation (FreeSurfer format)
print("\n3. Loading FreeSurfer parcellation annotations...")
label_dir = os.path.join(fs_dir, 'label')

# Use the 600-parcel atlas to match parcel_colors.json
lh_annot_file = os.path.join(label_dir, 'lh.Schaefer2018_600Parcels_17Networks_order.annot')
rh_annot_file = os.path.join(label_dir, 'rh.Schaefer2018_600Parcels_17Networks_order.annot')

if os.path.exists(lh_annot_file):
    print(f"   Using Schaefer2018 600-parcel 17-network atlas")
else:
    print(f"   600-parcel atlas not found, listing available files:")
    annot_files = [f for f in os.listdir(label_dir) if f.endswith('.annot')]
    for f in sorted(annot_files):
        print(f"     {f}")
    # Fall back to any Schaefer atlas
    lh_files = [f for f in annot_files if f.startswith('lh.Schaefer')]
    if lh_files:
        lh_annot_file = os.path.join(label_dir, lh_files[0])
        rh_annot_file = os.path.join(label_dir, lh_files[0].replace('lh.', 'rh.'))
        print(f"\n   Using: {os.path.basename(lh_annot_file)}")

# Read the annotation files
try:
    lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot(lh_annot_file)
    rh_labels, rh_ctab, rh_names = nib.freesurfer.read_annot(rh_annot_file)
    
    print(f"\n   Left hemisphere: {len(lh_labels):,} vertices, {len(np.unique(lh_labels))} unique labels")
    print(f"   Right hemisphere: {len(rh_labels):,} vertices, {len(np.unique(rh_labels))} unique labels")
    print(f"   Total parcel labels: {len(lh_names) + len(rh_names)}")
    
    # Get parcel names
    parcel_names_list = [name.decode() if isinstance(name, bytes) else name for name in lh_names] + \
                        [name.decode() if isinstance(name, bytes) else name for name in rh_names]
    
    print(f"\n4. Computing voxel-to-parcel mapping...")
    print("   Mapping chain: Voxels → Vertices → Parcels")
    
    # Note: v2v has 25,000 vertices total
    # We need to figure out which vertices correspond to LH vs RH
    n_vertices_total = v2v.shape[1]
    print(f"   Total vertices in mapping: {n_vertices_total:,}")
    print(f"   LH annotation vertices: {len(lh_labels):,}")
    print(f"   RH annotation vertices: {len(rh_labels):,}")
    
    # The voxel-to-vertex mapping uses a downsampled surface
    # We'll need to map between the annotation vertices and the downsampled vertices
    # For now, let's just show the structure
    
    print(f"\n5. Example: Finding voxels for first few parcels...")
    print("="*60)
    
    # For demonstration, let's find voxels for parcels using a simpler approach
    # We'll show parcels from the parcel_colors.json which should match the names
    parcel_colors = hmfiles.load_parcel_colors()
    parcel_names_from_colors = sorted(parcel_colors.keys())
    
    print(f"\nNOTE: The voxel-to-vertex mapping uses a downsampled surface ({n_vertices_total:,} vertices)")
    print(f"      The FreeSurfer annotation uses the full resolution surface ({len(lh_labels) + len(rh_labels):,} vertices)")
    print(f"      Direct mapping requires resampling the parcellation to match the downsampled surface.")
    print(f"\nShowing structure for first few parcels from parcel_colors.json:")
    
    for i, parcel_name in enumerate(parcel_names_from_colors[:5]):
        color = parcel_colors[parcel_name]
        print(f"\nParcel: {parcel_name}")
        print(f"  Color (RGB): {color}")
        print(f"  To find voxels:")
        print(f"    1. Find vertices labeled with this parcel (from annotation)")
        print(f"    2. Resample to downsampled surface (25,000 vertices)")
        print(f"    3. Find voxels that map to those vertices (from v2v matrix)")
    
    # Show an example with actual voxel counting
    print(f"\n\n{'='*60}")
    print("EXAMPLE: Counting voxels per parcel (simplified approach)")
    print("="*60)
    print("\nUsing the voxel-to-vertex mapping to show voxel distribution:")
    print(f"Total non-zero voxel-vertex connections: {v2v.nnz:,}")
    
    # Count voxels that map to each vertex
    voxels_per_vertex = np.diff(v2v_csc.indptr)
    print(f"Average voxels per vertex: {voxels_per_vertex.mean():.1f}")
    print(f"Max voxels per vertex: {voxels_per_vertex.max()}")
    print(f"Min voxels per vertex: {voxels_per_vertex.min()}")
    
    # Show which voxels map to a few example vertices
    print(f"\nExample vertex-to-voxel mappings:")
    for vertex_idx in [0, 100, 1000, 10000]:
        voxel_indices = v2v_csc[:, vertex_idx].nonzero()[0]
        print(f"  Vertex {vertex_idx}: {len(voxel_indices)} voxels")
        if len(voxel_indices) > 0:
            # Convert first voxel to 3D coordinates
            i, j, k = np.unravel_index(voxel_indices[0], gm_data.shape)
            print(f"    First voxel at (i,j,k): ({i}, {j}, {k})")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total voxels in volume: {v2v.shape[0]:,}")
    print(f"Total brain surface vertices: {v2v.shape[1]:,}")
    print(f"Total parcels: {len(parcel_names_list)}")
    print("\nMapping chain: Voxels → Vertices → Parcels")
    print("Each parcel contains multiple vertices, each vertex may have multiple voxels")

except Exception as e:
    print(f"\nError loading annotation files: {e}")
    import traceback
    traceback.print_exc()
