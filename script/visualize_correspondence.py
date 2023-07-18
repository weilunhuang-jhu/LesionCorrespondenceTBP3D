import sys, os
sys.path.append("../lib")
import pyvista as pv
import numpy as np
import argparse
import pickle
from PyQt5.QtWidgets import (QApplication, QFileDialog)

from MeshLoader import MeshData3DBodyTexTransformed

def key_callback_save_camera(*args):
    # print(plotter.camera_position)
    # print(plotter.camera)
    app = QApplication([])
    filename, filetype = QFileDialog.getSaveFileName(None, 'save file', os.getcwd(),
                                'All Files (*);;PKL Files (*.pkl)')
    cam_file = open(filename,'wb')
    cam_params = {}
    cam_params["position"] = plotter.camera.position
    cam_params["focal_point"] = plotter.camera.focal_point
    cam_params["view_up"] = plotter.camera.up
    cam_params["view_angle"] = plotter.camera.view_angle
    cam_params["clipping_range"] = plotter.camera.clipping_range

    print(cam_params)
    pickle.dump(cam_params, cam_file)
    print("cam parmas saved in " + filename)

def key_callback_load_camera(*args):
    app = QApplication([])
    filename, filetype = QFileDialog.getOpenFileName(None, 'open camera file', os.getcwd(),
                                'All Files (*);;PKL Files (*.pkl)')
    cam_file = open(filename,'rb')
    cam_params = pickle.load(cam_file)
    plotter.camera.position = cam_params["position"]
    plotter.camera.focal_point = cam_params["focal_point"] 
    plotter.camera.up = cam_params["view_up"]
    plotter.camera.view_angle = cam_params["view_angle"]
    plotter.camera.clipping_range = cam_params["clipping_range"]

    plotter.update()
    print(cam_params)
    print("cam parmas loaded from " + filename)

# argument set-up
parser = argparse.ArgumentParser(description="Find lesion correspodnence of from source to the target")
parser.add_argument("-s", "--source", type=str, help="Path to the source patient folder")
parser.add_argument("-t", "--target", type=str, help="Path to the target patient folder")
parser.add_argument("-r", "--result", type=str, help="Path to the parameter file")
parser.add_argument("-vs", "--visualize_source", action="store_true", help="Visualize source mesh and its LOIs only (store true)")
parser.add_argument("-vl", "--visualize_loi_labels", action="store_true", help="Visualize labels for LOIs (store true)")

# Parse the command line arguments to an object
args = parser.parse_args()
if not args.source or not args.target or not args.result:
    print("No input folder is provided.")
    print("For help type --help")
    exit()

np.random.seed(0) # fixed color

mesh_name ="real_scale_in_mm.ply"
texture_name = "model_highres_0_normalized.png"
landmark_name = "landmarks.txt"
loi_name = "lesion_of_interest.txt" 

root_dir_1 = args.source
ply_filename_1 = os.path.join(root_dir_1, mesh_name)
texture_filename_1 = os.path.join(root_dir_1, texture_name)
landmark_filename_1 = os.path.join(root_dir_1, landmark_name)
loi_filename_1 = os.path.join(root_dir_1, loi_name)

root_dir_2 = args.target
ply_filename_2 = os.path.join(root_dir_2, mesh_name)
texture_filename_2 = os.path.join(root_dir_2, texture_name)
landmark_filename_2 = os.path.join(root_dir_2, landmark_name)
loi_filename_2 = os.path.join(root_dir_2, loi_name)

mesh_data_1 = MeshData3DBodyTexTransformed(root_dir_1, mesh_name, texture_name, landmark_name, loi_name, use_trimesh=False, use_pvmesh=True, use_path_solver=True)
mesh_data_2 = MeshData3DBodyTexTransformed(root_dir_2, mesh_name, texture_name, landmark_name, loi_name, use_trimesh=False, use_pvmesh=True, use_path_solver=True)
colors = np.random.random(size=(len(mesh_data_1.loi_vertex_id_list), 3))

outfname = args.result
with open(outfname, 'rb') as outfile:
    output_data = pickle.load(outfile)

### Plot result
point_size = 7.5
radius_loi = 10
plotter = pv.Plotter(shape=(1, 2))
ADD_LOI_LABEL = args.visualize_loi_labels
VISUAL_SOURCE = args.visualize_source

if VISUAL_SOURCE:
    ### Clean mesh
    # Note that the (0, 0) location is active by default
    plotter.add_mesh(mesh_data_1.mesh_pv_processed, texture=mesh_data_1.texture_pv)

    # LOIs
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_data_1.mesh_pv_processed, texture=mesh_data_1.texture_pv)
    loi_points_source = mesh_data_1.V[mesh_data_1.loi_vertex_id_list]
    for loi_point_source, color in zip(loi_points_source, colors):
        s = pv.Sphere(radius=radius_loi, center=loi_point_source)
        plotter.add_mesh(s, color=color, opacity=0.4)
else:
    ### Clean mesh
    plotter.add_mesh(mesh_data_2.mesh_pv_processed, texture=mesh_data_2.texture_pv)

    # LOIs
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_data_2.mesh_pv_processed, texture=mesh_data_2.texture_pv)
    loi_points_target = mesh_data_2.V[mesh_data_2.loi_vertex_id_list]
    for loi_point_target, color in zip(loi_points_target, colors):
        s = pv.Sphere(radius=radius_loi, center=loi_point_target)
        plotter.add_mesh(s, color=color, opacity=0.4)

    if ADD_LOI_LABEL:
        # add LOI labels
        pts_loi_2 = pv.PolyData(loi_points_target)
        pts_loi_2["My Labels"] = [f"LOI {i}" for i in range(pts_loi_2.n_points)]
        plotter.add_point_labels(pts_loi_2, "My Labels", point_size=1,\
            shape_opacity=0.5, always_visible=False, font_size=20, name="loi_labels_2")

    # Localized LOI
    for loi_id, loi_vertex_id_localized in zip(output_data['loi_ids_localized'],\
                                            output_data['loi_vertex_ids_localized']):   
        loi_coord = mesh_data_2.V[loi_vertex_id_localized]
        if loi_vertex_id_localized != mesh_data_2.loi_vertex_id_list[loi_id]:
            # add geodesic path if not the same
            geodesic_path_pts = mesh_data_2.path_solver.find_geodesic_path(v_start=mesh_data_2.loi_vertex_id_list[loi_id],\
                    v_end=loi_vertex_id_localized)
            plotter.add_points(geodesic_path_pts, color=colors[loi_id], point_size=5)
        plotter.add_points(loi_coord, color=colors[loi_id], point_size=point_size)

# plotter setup
plotter.link_views()
plotter.add_key_event("s", key_callback_save_camera)
plotter.add_key_event("o", key_callback_load_camera)
plotter.show()