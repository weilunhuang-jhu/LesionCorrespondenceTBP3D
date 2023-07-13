import sys, os
sys.path.append("../")
import numpy as np

from echo_utils import get_hRadius, load_echo_descriptor, get_echo_descriptor_similarity

#############################################
# Helper functions for texture correspondence
#############################################

def find_loi_with_insignificant_signal(dir_source, loi_vertex_ids_source, tau, support_radius, avg_edge_length, threshold=100):
    '''
    Find LOIs with insignificant signal, with single FOV.
    Q: How to determine threshold?? empirically found
    '''
    loi_ids = []
    echo_ds = []
    hRadius = get_hRadius(support_radius, avg_edge_length)
    echo_dir_source = os.path.join(dir_source, "echo_descriptors" + "_tau_" + str(tau) + "_h_" + str(hRadius))
    
    for loi_vertex_id_source in loi_vertex_ids_source:   
        # Load ECHO descriptors # DEBUG: root_dir_1 is global var HERE!!!
        echo_descriptor_fname = "vert_" + str(loi_vertex_id_source) + ".txt"
        echo_descriptor_fname = os.path.join(echo_dir_source, echo_descriptor_fname)
        echo_d = load_echo_descriptor(echo_descriptor_fname)
        echo_ds.append(echo_d)
    for loi_id, echo_d in enumerate(echo_ds):
        echo_sum = np.sum(echo_d)
        
        if echo_sum < threshold:
            loi_ids.append(loi_id)
        
    return loi_ids

def filter_candidates_with_insignificant_signal(root_dir, corrs_ind_all, scores_shape,\
                                tau, support_radius, avg_edge_length,  threshold=100):
    '''
    Filter candidates with insignificant signal, with single FOV.
    NOTE: the echo descriptors returned is only at the single FOV.
    '''
    hRadius = get_hRadius(support_radius, avg_edge_length)
    echo_dir = os.path.join(root_dir, "echo_descriptors" + "_tau_" + str(tau) + "_h_" + str(hRadius))
    corrs_ind_filtered_all = []
    scores_shape_filtered_all = []

    for corrs_ind_per_loi, scores_shape_per_loi in zip(corrs_ind_all, scores_shape):   


        corrs_ind_filtered_per_loi = []
        scores_shape_filtered_per_loi = []

        for vertex_id, score_shape in zip(corrs_ind_per_loi, scores_shape_per_loi):
            echo_descriptor_fname = "vert_" + str(vertex_id) + ".txt"
            echo_descriptor_fname = os.path.join(echo_dir, echo_descriptor_fname)
            echo_descriptor = load_echo_descriptor(echo_descriptor_fname)
            echo_sum = np.sum(echo_descriptor)
            if echo_sum > threshold:
                corrs_ind_filtered_per_loi.append(vertex_id)
                scores_shape_filtered_per_loi.append(score_shape)

        # # debug
        # print("num: " + str(len(corrs_ind_per_loi)) + " , " + str(len(scores_shape_filtered_per_loi)))

        corrs_ind_filtered_all.append(corrs_ind_filtered_per_loi)
        scores_shape_filtered_all.append(scores_shape_filtered_per_loi)

    corrs_ind_filtered_all = np.array(corrs_ind_filtered_all, dtype='object')
    scores_shape_filtered_all = np.array(scores_shape_filtered_all, dtype='object')

    return (corrs_ind_filtered_all, scores_shape_filtered_all)
    
def get_texture_similarity_scores_multi_FOV(dir_source, dir_target, loi_vertex_ids_source, corrs_ind_all,\
                     avg_edge_length, echo_descriptors_target_all=None, score_type="cosine"):
    '''
    Get texture similarit scores for all candidate correspondence in the target
    Load ECHO descriptors in 3 FOV and compute the similarity scores between source and target candidate
    Performance: NOTE: slow, check the root cause (might be loading echo descriptors)
    '''
    taus = [0.01212, 0.02424, 0.04848] # 0.00606 => support radius ~5 mm
    support_radii = [10, 20, 40]
    corrs_scores_texture = [] # num_loi x num_FOV x num_corrs_per_loi

    for loi_id, loi_vertex_id_source in enumerate(loi_vertex_ids_source):   

        corrs_scores_per_loi = []

        for tau, support_radius in zip(taus, support_radii): # multi-FOV
            
            hRadius = get_hRadius(support_radius, avg_edge_length)
            
            # Load ECHO descriptors for LOI and target candidates 
            echo_dir_source = os.path.join(dir_source, "echo_descriptors" + "_tau_" + str(tau) + "_h_" + str(hRadius))
            echo_dir_target = os.path.join(dir_target, "echo_descriptors" + "_tau_" + str(tau) + "_h_" + str(hRadius))
            
            # Source
            echo_descriptor_fname = "vert_" + str(loi_vertex_id_source) + ".txt"
            echo_descriptor_fname = os.path.join(echo_dir_source, echo_descriptor_fname)
            echo_descriptor_source = load_echo_descriptor(echo_descriptor_fname)

            # Target (only load from disk if needed)
            if echo_descriptors_target_all is None:
                echo_descriptors_target_per_loi = []
                for vertex_id in corrs_ind_all[loi_id]:
                    echo_descriptor_fname = "vert_" + str(vertex_id) + ".txt"
                    echo_descriptor_fname = os.path.join(echo_dir_target, echo_descriptor_fname)
                    echo_descriptor = load_echo_descriptor(echo_descriptor_fname)
                    echo_descriptors_target_per_loi.append(echo_descriptor)
            else: # Use pre-loaded target echos
                echo_descriptors_target_per_loi = echo_descriptors_target_all[loi_id]
                
            # Get similarity scores of ECHO descriptors comparing to source LOI
            sim_cosine_list, sim_l2_list = get_echo_descriptor_similarity(echo_descriptor_source, echo_descriptors_target_per_loi)

            if score_type == "cosine":
                sim_list = sim_cosine_list.copy()
            elif score_type == "l2":
                sim_list = sim_l2_list.copy()
            else:
                print("ERROR: Score type unsupported")

            corrs_scores_per_loi.append(sim_list)
        corrs_scores_texture.append(corrs_scores_per_loi)
    corrs_scores_texture = np.array(corrs_scores_texture, dtype='object')
        
    return corrs_scores_texture

def get_scores_texture(dir_source, dir_target, loi_vertex_ids_source, corrs_ind_all,\
                            avg_edge_length, fov_weights=(0.33, 0.33, 0.33), echo_descriptors_target_all=None, score_type="cosine"):
    '''
    Get overall texture score for each (LOI). Score range: [0, 1] 
    NOTE: Currently assign equal weights to 3 FOVs, could be a learning step to better assign the weights 
    
    Args:
    Returns:
        scores: 
    '''
    
    corrs_scores_texture = get_texture_similarity_scores_multi_FOV(dir_source, dir_target,\
         loi_vertex_ids_source, corrs_ind_all, avg_edge_length, echo_descriptors_target_all, score_type=score_type)

    scores = []
    for corrs_scores_texture_per_loi in corrs_scores_texture:
        corrs_scores_per_fov_0 = corrs_scores_texture_per_loi[0]
        corrs_scores_per_fov_1 = corrs_scores_texture_per_loi[1]
        corrs_scores_per_fov_2 = corrs_scores_texture_per_loi[2]

        score = corrs_scores_per_fov_0 * fov_weights[0] + corrs_scores_per_fov_1 * fov_weights[1] + corrs_scores_per_fov_2 * fov_weights[2] # noramlize scores
        scores.append(score)
     
    scores = np.array(scores, dtype='object')

    return scores