import sys
sys.path.append("../")
import numpy as np
import scipy

from utils import get_correspondence_highest_score

###########################################
# Helper functions for shape correspondence
###########################################

def construct_feature_vectors(landmarks_vert_ind, dist_solver, feature_type=1, to_normalized=True):
    """
    Construct feature vector per vertex based on the geodesic distance from the vertex to the landmarks.

    Args:
        landmarks_vert_ind: list of vertex index of landmarks
        dist_solver: geodesic distance solver
        feature_type:
            0: 1/sqrt(dist)
            1: 1/dist
            2: 1/(dist)^2
            3: max(dist) - dist
    Returns:
        feature_vecs: normalized feature vectors, in matrix form, F^{v x s}, for all the vertices. (v:num_vertices, s:num_landmarks)
    """
    # return feature vector based on geodesic distance to landmarks 
    feature_vecs = []
    for landmark_vert_ind in landmarks_vert_ind:
        feature_vec = dist_solver.compute_distance(landmark_vert_ind)
        # print("=======")
        # print("max :" + str(np.max(feature_vec)))
        # print("min :" + str(np.min(feature_vec)))
        # feature_vecs.append(feature_vec) # using distance, incorrect

        if feature_type == 0:
            feature_vecs.append(1/(np.sqrt(feature_vec) + 0.000001)) # using 1/dist^(1/2) to assign more weights for closer landmarks, add regularization to prevent dividing by zero
        elif feature_type == 1:
            feature_vecs.append(1/(feature_vec + 0.000001)) # using 1/dist to assign more weights for closer landmarks, add regularization to prevent dividing by zero
        elif feature_type == 2:
            feature_vecs.append(1/(np.square(feature_vec) + 0.000001)) # using 1/dist^2 to assign more weights for closer landmarks, add regularization to prevent dividing by zero
        else:
            feature_vecs.append(feature_vec) # using distance, for features require post-processing

    feature_vecs = np.array(feature_vecs) # (num_landmarks, num_vertices)
    feature_vecs = feature_vecs.T
    if feature_type == 3:
        feature_vecs = np.repeat(np.max(feature_vecs, axis=1).reshape(len(feature_vecs), 1), len(landmarks_vert_ind), axis=1)\
             - feature_vecs # using max(dist) - dist

    if to_normalized:
        feature_norm = np.linalg.norm(feature_vecs, axis=1).reshape((len(feature_vecs), 1))
        feature_norm = np.repeat(feature_norm, len(landmarks_vert_ind), axis=1)
        feature_vecs = feature_vecs / feature_norm # (num_vertices, num_landmarks)

    return feature_vecs

def get_score_shape_gaussian(dist_solver, corrs_ind_shape, center_vertex_ids):
    """
    Get shape score based on Gaussian distribution centered at the best shape correspondence.
    NOTE: Difficult to determine the STD. STD is currently determined by the max geodesic distance.

    Args:
        dist_solver
        corrs_ind_shape
        center_vertex_ids
    Returns:
        scores: list of numpy arrays
    """
    scores = []
    for center_vertex_id, corr_ind_shape in zip(center_vertex_ids, corrs_ind_shape):
        corr_ind_shape = corr_ind_shape.astype(int)
        geo_distances = dist_solver.compute_distance(center_vertex_id)
        geo_distances = geo_distances[corr_ind_shape]
        std = np.max(geo_distances) # DEBUG
        score = scipy.stats.norm(0, std).pdf(geo_distances)
        score = score / np.max(score, axis=0) # Normalize score to [0,1]
        scores.append(score)

    return scores     

def get_scores_shape_feature(feature_vecs_1_LOI, feature_vecs_2, metric="cosine", corrs_ind_shape=None):
    """
    Get shape score based on the similarity of feature vector between source and target.

    Args:
        feature_vecs_1_LOI
        feature_vecs_2
        corrs_ind_shape
    Returns:
        scores: np array
    """
    
    if metric == "cosine":
        if corrs_ind_shape is None:
            scores = np.matmul(feature_vecs_1_LOI, feature_vecs_2.T) # (num_LOI, num_vertices)
            return scores
        
        scores = []
        for feature_vec_1_LOI, corr_ind_shape in zip(feature_vecs_1_LOI, corrs_ind_shape):  
            feature_vec_corr = feature_vecs_2[corr_ind_shape] # num_corr x num_LM
            score = np.matmul(feature_vec_corr, feature_vec_1_LOI) # num_corr x 1
            scores.append(score) 
        scores = np.array(scores, dtype='object')

    else:
        print("Not implemented")
    
    return scores

def find_neighbors_in_geodesic_circle(dist_solver, center_vertex_id, radius=30):
    '''
    Find neighboring vertices within radius centered at the provided vertex

    Args:
        dist_solver
        center_vertex_id
        radius 
    Returns:
        neighbors_vertex_ids: np array

    '''    
    geo_distances = dist_solver.compute_distance(center_vertex_id)
    neighbors_vertex_ids = np.where(geo_distances <= radius)[0]
    
    return neighbors_vertex_ids
        
def find_corrs_candidates_in_geodesic_circle(dist_solver, center_vertex_ids, radius=30):
    '''
    Find candidate correspondence vertices within radius centered at the localized LOI, for all LOI in the target mesh
    NOTE: the distribution of top N correspondences is related to the distance to the closest landmark!!
    Performance: ~ 10 sec
    
    Args:
    Returns:
        neighbors_vertex_ids: list of numpy array
    '''    
    neighbors_vertex_ids = []
    for center_vertex_id in center_vertex_ids:
        neighbors_vertex_ids_per_loi = find_neighbors_in_geodesic_circle(dist_solver, center_vertex_id, radius=radius)
        neighbors_vertex_ids.append(neighbors_vertex_ids_per_loi)
    
    return neighbors_vertex_ids


def find_corrs_candidates_shape(lm_vert_id_list_1, lm_vert_id_list_2,\
                    loi_vertex_id_list_1, dist_solver_1, dist_solver_2,\
                    feature_type="1", metric="cosine", to_normalized=True,\
                    radius=30, std_factor=0.5, score_type="gaussian", return_scores=False):
    '''
    Find candidate corresponddence vertices for all LOI based on shape info
    NOTE: Using score from Gaussian distribution
    
    Args:
    Returns:
        corrs_candidates_shape: list of numpy array
    '''
    
    feature_vecs_1 = construct_feature_vectors(lm_vert_id_list_1,\
            dist_solver_1, feature_type=feature_type, to_normalized=to_normalized)
    feature_vecs_2 = construct_feature_vectors(lm_vert_id_list_2,\
        dist_solver_2, feature_type=feature_type, to_normalized=to_normalized)
    feature_vecs_1_loi = feature_vecs_1[loi_vertex_id_list_1] # feature vectors for lesion of interest in mesh 1

    # Get scores and correspondence from feature representation
    scores_shape_feature = get_scores_shape_feature(feature_vecs_1_loi, feature_vecs_2, metric=metric)
    corrs_shape_feature = get_correspondence_highest_score(scores_shape_feature) # correspondence from shape feature
    
    # Get candidates correspondence based on geodesic circle
    corrs_candidates_gedoesic_circle = find_corrs_candidates_in_geodesic_circle(dist_solver_2, corrs_shape_feature,\
                                                                               radius=radius)
    
    # Get candidate correspondence vertices using feature representation of landmark geodesic distance,
    # conditional on mean and std scores within the geodesic circle.
    # NOTE: The mean tends to be low and std tend to be high when the LOI is closer to landmarks.
    scores_candidates_geodesic_circle = get_scores_shape_feature(feature_vecs_1_loi, feature_vecs_2,\
                                        metric=metric, corrs_ind_shape=corrs_candidates_gedoesic_circle)
    corrs_candidates_shape_feature = []
    for score_1, score_2 in zip(scores_candidates_geodesic_circle, scores_shape_feature):
        thre = np.mean(score_1) + std_factor * np.std(score_1) # debug
        corrs_candidates_shape_feature.append(np.where(score_2 > thre)[0])

    # Merge with two kinds of candidate correspondence vertices
    corrs_candidates_shape = []
    for corr_1, corr_2 in zip(corrs_candidates_shape_feature, corrs_candidates_gedoesic_circle):
        arr1 = np.array(corr_1)
        arr2 = np.array(corr_2)
        corrs = np.union1d(arr1, arr2)
        corrs_candidates_shape.append(corrs)
        
    if not return_scores:
        return (corrs_candidates_shape, corrs_shape_feature, corrs_candidates_shape_feature, corrs_candidates_gedoesic_circle)
    else:
        if score_type=="gaussian":
            # Rescale scores based on spatial Gaussian distribution centered at corrs_shape_feature, score range: [0, 1]
            scores = get_score_shape_gaussian(dist_solver_2, corrs_candidates_shape, corrs_shape_feature)
        elif score_type=="feature": #TODO: Need to check this later and rescale scores
            scores = get_scores_shape_feature(feature_vecs_1_loi, feature_vecs_2, metric=metric, corrs_ind_shape=corrs_candidates_shape)

        return (corrs_candidates_shape, scores,\
                corrs_shape_feature, corrs_candidates_shape_feature, corrs_candidates_gedoesic_circle)