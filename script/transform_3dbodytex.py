import os
import pandas as pd
import numpy as np
import trimesh
import pymeshlab
import argparse

import sys
sys.path.append("../")
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../skin3d')))
from skin3d.bodytex import BodyTexDataset

# Script to transform the 3dbodytex data by: (for convenience since the parameters used in this work is in mm) 
# NOTE: The script also transform the data to align with the convention used in [3D-CODED](https://github.com/ThibaultGROUEIX/3D-CODED).

# argument set-up
parser = argparse.ArgumentParser(description="Transform 3dbodytex data (highres)")
parser.add_argument("-i", "--input", type=str, help="Input directory of the 3dbodytex data")
parser.add_argument("-o", "--output", type=str, default="../data/3dbodytex_long_data", help="Output folder of the 3dbodytex data")

# Parse the command line arguments to an object
args = parser.parse_args()
if not args.input:
    print("No input folder is provided.")
    print("For help type --help")
    exit()

### Longitudinal annotation from Skin3D
bodytex_csv = '../skin3d/data/3dbodytex-1.1-highres/bodytex.csv'
bodytex_df = pd.read_csv(bodytex_csv, converters={'scan_id': lambda x: str(x)})
data_dir = args.input # path to the folder of 3dbodytex
bodytex = BodyTexDataset(df=bodytex_df, dir_textures=data_dir)
long_df = bodytex.annotated_samples_in_partition('long')
long_ids = [long_id.split('-') for long_id in long_df.subject_id.values] # get pairs of corresponding id

output_dir = args.output # path to output folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mesh_basename = "real_scale_in_mm.ply"
unit = "mm"
scale_factor = 1
if unit == "mm":
    scale_factor = 1000

for long_id in long_ids:
    for scan_id in long_id:
        scan_row = bodytex.scan_row(scan_id)
        scan_name = scan_row.scan_name.values[0]
        print(scan_name)
        
        # load data     
        obj_filename = os.path.join(data_dir, scan_name, "model_highres_0_normalized.obj")
        mesh_trimesh = trimesh.load(obj_filename, process=False) # still has re-ordering because of wedge texture
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(obj_filename)
        
        # apply transformation
        ms.compute_matrix_from_translation(axisx=-mesh_trimesh.centroid[0], axisy=-mesh_trimesh.centroid[1],\
                                           axisz=-mesh_trimesh.centroid[2], freeze=False)                
        ms.compute_matrix_from_scaling_or_normalization(axisx=scale_factor, axisy=scale_factor, axisz=scale_factor, freeze=False)
        ms.compute_matrix_from_rotation(rotaxis=2, angle=90, freeze=False)

        # output dir per scan
        output_dir_scan = os.path.join(output_dir, scan_id)
        if not os.path.exists(output_dir_scan):
            os.makedirs(output_dir_scan)

        # save transformation matrix
        transformation_matrix = ms.current_mesh().trasform_matrix()
        output_fname_tf = os.path.join(output_dir_scan, "transformation.txt")
        np.savetxt(output_fname_tf, transformation_matrix, fmt='%s', delimiter=',')
        
        # freeze transformation for the mesh
        ms.apply_matrix_freeze()
        
        # save ply
        output_fname_ply = os.path.join(output_dir_scan, mesh_basename)
        ms.save_current_mesh(output_fname_ply)