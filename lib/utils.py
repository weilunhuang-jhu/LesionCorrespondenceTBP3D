import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import yaml
import pickle
from collections import OrderedDict

class OrderedVertices(object):
    """
    Class for storing ordered vertices, used for LOIs and landmarks
    """
    def __init__(self, ids=[], vertex_ids=[]):
        # ids: list of int
        # vertex_ids: list of int
        self.vertices = OrderedDict() # key: id, value: vertex_id
        for id, vertex_id in zip(ids, vertex_ids):
            self.vertices[id] = vertex_id
        self.vertices = OrderedDict(sorted(self.vertices.items())) # sort by id

    def merge(self, ordered_vertices):
        """
        Merge ordered_vertices with self.vertices. NOTE: The ids of ordered_vertices should be unique from self.vertices.
        TODO: allow overwriting of existing ids

        Args:
        Returns: None
        """
        for id, vertex_id in ordered_vertices.vertices.items():
            # sanity check
            if id in self.vertices:
                print("[WARNING]: id " + str(id) + " already exists and is not updated.")
                continue
            self.vertices[id] = vertex_id
        self.vertices = OrderedDict(sorted(self.vertices.items())) # sort by id

    def ids(self):
        return np.array(list(self.vertices.keys())).astype(int)

    def vertex_ids(self):
        return np.array(list(self.vertices.values())).astype(int)

    def __len__(self):
        return len(self.vertices)

#####################################
# Helper functions for debug and save
####################################
def compute_error_stats(mesh_data, output_data, metric="geodesic"):
    # Compute localization errors
    msg = "\n"
    errors = []
    loi_ids_localized = output_data['loi_ids_localized']
    loi_vertex_ids_localized = output_data['loi_vertex_ids_localized']

    if (len(mesh_data.loi_vertex_id_list) != len(loi_ids_localized)):
        print("[WARNING]: Missing LOIs to be localized below: ")
        print(set(range(len(mesh_data.loi_vertex_id_list))) - set(loi_ids_localized))

    for loi_id_localized, loi_vertex_id_localized in zip(loi_ids_localized, loi_vertex_ids_localized):   

        loi_gt_vert_id = mesh_data.loi_vertex_id_list[loi_id_localized]

        if metric == "geodesic":
            error = mesh_data.dist_solver.compute_distance(loi_gt_vert_id)[loi_vertex_id_localized]

        elif metric == "euclidean":
            loi_gt = mesh_data.V[loi_gt_vert_id]
            loi_coord = mesh_data.V[loi_vertex_id_localized]
            error = np.linalg.norm(loi_coord - loi_gt)
        
        else:
            print("Metric not supported.")
            return 
        
        msg += "LOI " + str(loi_id_localized) + " : " + str(error) + "\n"
        errors.append(error)
        
    errors = np.array(errors)
        
    msg += "===summary===\n"
    msg += str(errors.mean()) + "\n"
    msg += str(errors.std()) + "\n"

    return msg

def compute_CLE(mesh_data, loi_ids_localized, loi_vertex_ids_localized, metric="geodesic"):

    errors = []
    for loi_id_localized, loi_vertex_id_localized in zip(loi_ids_localized, loi_vertex_ids_localized):   

        loi_gt_vert_id = mesh_data.loi_vertex_id_list[loi_id_localized]

        if metric == "geodesic":
            error = mesh_data.dist_solver.compute_distance(loi_gt_vert_id)[loi_vertex_id_localized]

        elif metric == "euclidean":
            loi_gt = mesh_data.V[loi_gt_vert_id]
            loi_coord = mesh_data.V[loi_vertex_id_localized]
            error = np.linalg.norm(loi_coord - loi_gt)
        print("LOI " + str(loi_id_localized) + " : " + str(error))
        errors.append(error)
        
    errors = np.array(errors)
        
    print("===summary===")
    print(errors.mean())
    print(errors.std())

    return errors

def show_msg(msg, logger):
    print(msg)
    logger.info(msg)

def set_params(params, radius_geodesic, threshold_texture_score, threshold_shape_texture_consensus):
    params['update'] = {}
    params['update']['radius_geodesic'] = radius_geodesic # Start from min radius, increasing the radius iteratively
    params['update']['threshold_texture_score'] = threshold_texture_score
    params['update']['threshold_shape_texture_consensus'] = threshold_shape_texture_consensus
    return params

def update_params(params, radius_geodesic_multiplier=1., threshold_texture_multiplier=1., threshold_consensus_multiplier=1.):
    params['update']['radius_geodesic'] *= radius_geodesic_multiplier
    params['update']['threshold_texture_score'] *= threshold_texture_multiplier
    params['update']['threshold_shape_texture_consensus'] *= threshold_consensus_multiplier
    return params

def load_params(path_name):
    params = yaml.load(open(path_name, 'r'), Loader=yaml.FullLoader)
    return params

def save_output(subject_name, outflag, iter_num, lois, lois_localized,\
                corrs_shape_feature, corrs_texture, corrs_combined,\
                scores_shape, scores_texture, scores_combined,\
                corrs_candidates_shape_unfiltered, corrs_candidates_shape, corrs_candidates_texture_clusters, save_dir="output"):
    
    output_data = {}    
    output_data['loi_ids_remained'] = lois.ids() 
    output_data['loi_ids_localized'] = lois_localized.ids()
    output_data['loi_vertex_ids_localized'] = lois_localized.vertex_ids() 

    output_data['corrs_shape_feature'] = corrs_shape_feature
    output_data['corrs_texture'] = corrs_texture
    output_data['corrs_combined'] = corrs_combined

    output_data['scores_shape'] = scores_shape
    output_data['scores_texture'] = scores_texture
    output_data['scores_combined'] = scores_combined

    output_data['corrs_candidates_shape'] = corrs_candidates_shape
    output_data['corrs_candidates_shape_unfiltered'] = corrs_candidates_shape_unfiltered
    output_data['corrs_candidates_texture_clusters'] = corrs_candidates_texture_clusters


    outfname = "output_" + str(subject_name) + "_" + str(iter_num).zfill(2) + outflag + ".pkl"
    outfname = os.path.join(save_dir, outfname)
    os.makedirs(save_dir, exist_ok=True)
    with open(outfname, "wb") as outfile:
        pickle.dump(output_data, outfile, pickle.HIGHEST_PROTOCOL)
    print(outfname + " is saved.")

    return output_data

####################################
# Helper functions for correpondence
####################################
def get_correspondence_highest_score(scores):
    """
    Get correspondence in mesh 2 for each lesion of interest in mesh 1

    Args:
        scores: np array
    Returns:
    """
    
    if scores.ndim == 1: # unequal shape in the second dimension
        corrs_vertex_ids = []
        for score in scores:
            if len(score > 0):
                corrr_vertex_id = np.argmax(score)
                corrs_vertex_ids.append(corrr_vertex_id)
            else: # empty score list
                corrs_vertex_ids.append(-1)
        corrs_vertex_ids = np.array(corrs_vertex_ids)
    else:
        # get correspondence vertices
        corrs_vertex_ids = np.argmax(scores, axis=1)

    return corrs_vertex_ids

def combine_scores(scores_1, scores_2, method='add', weights=(0.5, 0.5)):
    scores = []
    for score_1, score_2 in zip(scores_1, scores_2):
        assert len(score_1) == len(score_2)

        if method == "product":
            score = np.array(score_1) * np.array(score_2)
        elif method == "add":
            score = np.array(score_1) * weights[0] + np.array(score_2) * weights[1]
        else:
            print("Unsupported method")

        scores.append(score)  

    scores = np.array(scores, dtype='object')

    return scores

def find_correspondence_hungarian(feature_vecs_1, feature_vecs_2):
    # find correspondence with Hungarian method
    # NOTE: assume square matrix here
    cost_matrix = distance_matrix(feature_vecs_1, feature_vecs_2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind].sum()
    
    return (cost, row_ind, col_ind)

###########################
# Helper functions for mesh
# #########################

# ref: https://github.com/pyvista/pyvista-support/issues/96
def find_faces_with_node(index, faces):
    """Pass the index of the node in question.
    Returns the face indices of the faces with that node."""
    return [i for i, face in enumerate(faces) if index in face]

# ref: https://github.com/pyvista/pyvista-support/issues/96
def find_connected_vertices(index, faces):
    """Pass the index of the node in question.
    Returns the vertex indices of the vertices connected with that node."""
    cids = find_faces_with_node(index, faces)
    connected = np.unique(faces[cids].ravel())
    return np.delete(connected, np.argwhere(connected == index))

def find_splitted_vertex(vertices_coord, index):
    """Pass the index of the node in question.
    Returns the vertex indices of the vertices splitted with that node."""
    vertex_coord = vertices_coord[index]
    diff = vertices_coord - vertex_coord
    identical_indices = np.where(np.linalg.norm(diff, axis=1) < 1E-6)[0]
    return identical_indices

def uv_to_polar(log_map_in_uv):
    r = np.sqrt(np.sum(log_map_in_uv * log_map_in_uv, axis=1))
    theta = np.arctan2(log_map_in_uv[:,1], log_map_in_uv[:,0])
    log_map_in_polar = np.hstack((r.reshape((-1, 1)), theta.reshape((-1,1))))

    return log_map_in_polar

def find_closest_vertex(pt, V):
    diff = V - np.array(pt)
    dist = np.linalg.norm(diff, axis=1)
    closest_vertex_id = np.argmin(dist)

    return closest_vertex_id