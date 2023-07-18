import os
import numpy as np
import pymeshlab
import argparse

# Script to get the texture signal as scalar per vertex by:
# 1. Load the mesh
# 2. Turn the mesh in grayscale
# 3. Get the scalar per vertex signal of texture

# argument set-up
parser = argparse.ArgumentParser(description="Get the texture signal as scalar per vertex")
parser.add_argument("-i", "--input", type=str, help="Input directory of the pre-processed 3dbodytex data")

# Parse the command line arguments to an object
args = parser.parse_args()
if not args.input:
    print("No input folder is provided.")
    print("For help type --help")
    exit()

base_dir = args.input 
root_dirs = os.listdir(base_dir)
root_dirs.sort()

for root_dir in root_dirs:
    print(root_dir)
    root_dir = os.path.join(base_dir, root_dir)
    
    mesh_path = os.path.join(root_dir, "real_scale_in_mm.ply")
    texture_path = os.path.join(root_dir, "model_highres_0_normalized.png")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    # ms.compute_color_from_texture_per_vertex() # same as ms.transfer_color_texture_to_vertex(), deprecated
    ms.transfer_texture_to_color_per_vertex() # upperbound=pymeshlab.Percentage(0.1)
    ms.apply_color_desaturation_per_vertex(method='Luminosity')
    output_fname_ply = mesh_path.replace(".ply", "_gray.ply")
    ms.save_current_mesh(output_fname_ply, binary=False, save_textures=False, save_wedge_texcoord=False)

    m = ms.current_mesh()
    vc = m.vertex_color_matrix()
    vc = np.delete(vc, 3, 1)

    scalar_per_vertex = vc[:,0] * 255
    scalar_per_vertex = scalar_per_vertex.astype(int)
    outfile_fname = os.path.join(root_dir, "texture_signal_per_vertex.txt")
    np.savetxt(outfile_fname, scalar_per_vertex, fmt="%d")