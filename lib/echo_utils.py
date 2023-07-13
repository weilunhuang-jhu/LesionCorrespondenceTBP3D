import os
import subprocess
import numpy as np
import multiprocessing

from utils import load_params

params_path = "params.yml"
params = load_params(params_path)
ECHO_BIN = params["echo"]["bin"]
ECHO_MULTI_VERT_NUM = params["echo"]["multi_vert_num"]

# Helper functions for ECHO descriptor

def get_hRadius(radius, avg_edge_length):

    # size of histogram bin (in mm) should be at least 2 times of the average edge length
    # radius / hRadius ~= 2 x avg_edge_length
    hRadius = max(2, int(radius / (2 * avg_edge_length)))

    return hRadius

def get_cached_ECHO_vert_ids(echo_dir):
    '''
        Get vertex ids of cached ECHO descriptor in a folder
    '''
    echo_vertex_ids = []

    if not os.path.exists(echo_dir):
        print("ECHO descriptors do not exist. The algorithm will acquire all ECHO descriptors during run-time.")
        return echo_vertex_ids

    print("Some ECHO descriptors exist. The algorithm will skip acquiring cached descriptors during run-time.")
    echo_descriptor_fnames = os.listdir(echo_dir)
    echo_descriptor_fnames.sort()
    # process to get vert_id
    for echo_descriptor_fname in echo_descriptor_fnames:
        vertex_id = echo_descriptor_fname.split("_")[-1].split(".")[0]
        echo_vertex_ids.append(int(vertex_id))
    
    return echo_vertex_ids

def get_echo_descriptors(data_root_dir, mesh_name, tau, hRadius, distance_type=0):
    '''
        Utility function to get ECHO descriptors of a list of vertices
    '''

    input_mesh = os.path.join(data_root_dir, mesh_name)
    input_texture_signal = os.path.join(data_root_dir, "texture_signal_per_vertex.txt")
    input_vertex_id = input_mesh.replace(".ply", "_vert_id_list.txt")
    out_dir = os.path.join(data_root_dir, "echo_descriptors" + "_tau_" + str(tau) + "_h_" + str(hRadius))
    os.makedirs(out_dir, exist_ok=True)
    pEcho = subprocess.Popen( [ECHO_BIN, "--in", input_mesh, "--texture", input_texture_signal,\
        "--source_vertices", input_vertex_id, "--vertex", str(-1), "--out", out_dir,\
        "--distance", str(distance_type),  "--tau", str(tau), "--hRadius", str(hRadius),\
        "--noDisk"] )
    pEcho.wait()

def worker_get_echo_descriptors(data_root_dir, mesh_name, input_vertex_id_txt, tau, hRadius, distance_type=0):
    '''
        Worker to get ECHO descriptors of a list of vertices
    '''
    
    input_mesh = os.path.join(data_root_dir, mesh_name)
    input_texture_signal = os.path.join(data_root_dir, "texture_signal_per_vertex.txt")
    
    out_dir = os.path.join(data_root_dir, "echo_descriptors" + "_tau_" + str(tau) + "_h_" + str(hRadius))
    os.makedirs(out_dir, exist_ok=True)
    
    pEcho = subprocess.Popen( [ECHO_BIN, "--in", input_mesh, "--texture", input_texture_signal,\
        "--source_vertices", input_vertex_id_txt, "--vertex", str(-1), "--out", out_dir,\
        "--distance", str(distance_type),  "--tau", str(tau), "--hRadius", str(hRadius),\
        "--noDisk"] )
    pEcho.wait()
    
def load_echo_descriptor(descriptor_fname):
    '''
        Load a single ECHO descriptor from a txt file
    '''
    echo_d = np.loadtxt(descriptor_fname, delimiter=" ", dtype=np.float64)
    return echo_d

def get_echo_descriptor_similarity(echo_d_1, echo_ds_2):
    '''
        Get similarity (cosine similarity and l2 distance) between two ECHO descriptors
        ASSUME a single echo_d_1 and multiple echo_d_2  (echo_ds_2)
        NOTE: Not using ssim for two reasons:
        1. SSIM is not suitable for comparing ECHO descriptors
        2. The echo bin size can be less than 7 when using the average edge length in the func get_hRadius()
    '''
    
    sim_cosine_list = []
    sim_l2_list = []
    for echo_d_2 in echo_ds_2:
        sim_cosine = np.dot(echo_d_1, echo_d_2) / (np.linalg.norm(echo_d_1) * np.linalg.norm(echo_d_2))
        sim_l2 = np.linalg.norm(echo_d_1 - echo_d_2)
        
        sim_cosine_list.append(sim_cosine)
        sim_l2_list.append(sim_l2)
    
    sim_cosine_list = np.array(sim_cosine_list)
    sim_l2_list = -np.array(sim_l2_list) # NOTE: use negative scores for sim_l2
    
    return sim_cosine_list, sim_l2_list

def get_echo_descriptors_multiprocess(tau, hRadius, root_dir, mesh_name, unique_vert_ids, distance_type=0):
    '''
        Utility funciton to acquire ECHO descriptors with multiprocessing.
    '''

    num_worker = os.cpu_count()
    # split verts 
    unique_vert_ids_splitted_list = np.array_split(unique_vert_ids, num_worker)
    for i, unique_vert_ids_list in enumerate(unique_vert_ids_splitted_list):
        out_fname = mesh_name.replace(".ply", "_vert_id_list_" + str(i).zfill(2) + ".txt")
        out_fname = os.path.join(root_dir, out_fname)
        np.savetxt(out_fname, unique_vert_ids_list, fmt="%d")

    pool = multiprocessing.Pool(num_worker)
    for i in range(num_worker):   
        input_vertex_id_txt = mesh_name.replace(".ply", "_vert_id_list_" + str(i).zfill(2) + ".txt")
        input_vertex_id_txt = os.path.join(root_dir, input_vertex_id_txt)
        pool.apply_async(worker_get_echo_descriptors, (root_dir, mesh_name, input_vertex_id_txt, tau, hRadius, distance_type))
    pool.close()
    pool.join()

    # remove splitted vert txt
    for i in range(num_worker):   
        input_vertex_id_txt = mesh_name.replace(".ply", "_vert_id_list_" + str(i).zfill(2) + ".txt")
        input_vertex_id_txt = os.path.join(root_dir, input_vertex_id_txt)
        os.remove(input_vertex_id_txt)

def get_echos_in_alg(iter_num, echo_vertex_ids_prev, corrs_candidates_shape,\
                     echo_taus, echo_support_radii, echo_distance_type,\
                     loi_vertex_id_list_1, root_dir_1, root_dir_2, mesh_name, avg_edge_length, debug=False):
    '''
        Utility funciton to acquire ECHO descriptors for the source and target meshes in the algorithm during run-time.
    '''

    ply_filename_1 = os.path.join(root_dir_1, mesh_name)
    ply_filename_2 = os.path.join(root_dir_2, mesh_name)

    # Get ECHO descriptors for source
    if iter_num == 0 or iter_num == "final": # Get ECHO for source (in final run, we may need to get ECHO descriptors for initially excluded LOIs)
        # save vert_id_list for source
        out_fname = ply_filename_1.replace(".ply", "_vert_id_list.txt")
        np.savetxt(out_fname, loi_vertex_id_list_1, fmt="%d")
        for tau, support_radius in zip(echo_taus, echo_support_radii):
            hRadius = get_hRadius(support_radius, avg_edge_length)
            get_echo_descriptors(root_dir_1, mesh_name, tau, hRadius, echo_distance_type)

    # Get ECHO descriptors for target (avoid acquring cached ECHO descriptors when possible)
    echo_vertex_ids = []
    for corrs_ind in corrs_candidates_shape:
        echo_vertex_ids.extend(corrs_ind)
    echo_vertex_ids = np.unique(np.array(echo_vertex_ids))

    if debug:
        print("Num of ECHO verts needed in alg: " + str(len(echo_vertex_ids)))

    # get and save new vertices only
    echo_vertex_ids = np.setdiff1d(echo_vertex_ids, echo_vertex_ids_prev)

    if len(echo_vertex_ids) > 0:
        # save vert_id_list for target
        out_fname = ply_filename_2.replace(".ply", "_vert_id_list.txt")
        np.savetxt(out_fname, echo_vertex_ids, fmt="%d")
        for tau, support_radius in zip(echo_taus, echo_support_radii):
            hRadius = get_hRadius(support_radius, avg_edge_length)
            if len(echo_vertex_ids) > ECHO_MULTI_VERT_NUM:
                get_echo_descriptors_multiprocess(tau, hRadius, root_dir_2, mesh_name, echo_vertex_ids)
            else:
                get_echo_descriptors(root_dir_2, mesh_name, tau, hRadius, echo_distance_type) #~ 3min for 15K vertices, support radius 10 mm

    # update ECHO variables
    echo_vertex_ids_prev = np.union1d(echo_vertex_ids_prev, echo_vertex_ids).astype(int)

    return (echo_vertex_ids, echo_vertex_ids_prev)