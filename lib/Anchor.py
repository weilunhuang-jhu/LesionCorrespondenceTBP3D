import numpy as np
from utils import OrderedVertices

##############################################
# Helper functions to anchor well-located LOIs
##############################################

def anchor_loi_unambiguous_texture(lois, corrs_scores, corrs_ind, V, avg_edge_length,\
                    texture_threshold=0.9, std_factor=1, dist_factor=2, return_clusters=False):
    '''
    Anchor good correspondence as landmarks by checking if the texture correspondence is unambiguous
    Find good correspondence by deviation of candidate locations (also characterize clusters)

    Args:
        lois
    Returns:

    '''    

    loi_ids = lois.ids()
    dist_threshold = dist_factor * avg_edge_length
    loi_vertex_ids_anchored = []
    loi_ids_anchored = []
    corrs_candidates_texture_clusters = []

    for loi_id, corr_scores, corr_ind in zip(loi_ids, corrs_scores, corrs_ind):

        assert len(corr_scores) == len(corr_ind)
        if len(corr_scores) == 0: # avoid emty scores
            corrs_candidates_texture_clusters.append([]) # append empty list
            continue

        if np.max(corr_scores) < texture_threshold: # check texture score of texture corr first
            corrs_candidates_texture_clusters.append([]) # append empty list
            continue

        corr_ind = np.array(corr_ind)
        corr_scores = np.array(corr_scores)
        
        # get threshold score
        score_threshold = np.max(corr_scores) - std_factor * np.std(corr_scores)
        corr_ind_over_threshold = corr_ind[corr_scores > score_threshold].astype(int) # debug
        corrs_candidates_texture_clusters.append(corr_ind_over_threshold)
        if len(corr_ind_over_threshold) == 0:
            continue
        
        # get the deviation of candidates over threhold 
        corr_coords = V[corr_ind_over_threshold]
        corr_center = corr_coords.mean(axis=0)
        avg_dist_to_center = np.mean(np.linalg.norm(corr_coords - corr_center, axis=1))
        
        # capture unambiguous correspondence
        if avg_dist_to_center < dist_threshold:
            loi_ids_anchored.append(loi_id)
#             loi_vertex_id = find_closest_vertex(corr_center, V) # NOTE: use corr_center 
            loi_vertex_id = corr_ind[np.argmax(corr_scores)] # NOTE: use texture corr 
            loi_vertex_ids_anchored.append(loi_vertex_id)

    lois_anchored = OrderedVertices(ids=loi_ids_anchored, vertex_ids=loi_vertex_ids_anchored)
        
    if not return_clusters:
        return lois_anchored

    return (lois_anchored, corrs_candidates_texture_clusters)

def anchor_loi_texture_thresholding(lois, corrs_scores, corrs_ind, factor_std=2, threshold_score_abs=0.9):
    '''
    Anchor good correspondence as landmarks by comparing texture scores within LOIs and using a threshold value.
    Currently the function will anchor no more than half of the remaining lesions
    '''    
    
    loi_ids = lois.ids()
    loi_vertex_ids_anchored = []
    loi_ids_anchored = []
    
    # set up threshold value
    threshold_score_rel = np.mean(corrs_scores) + factor_std * np.std(corrs_scores) # relative score by stats from LOIs
    threshold = max(threshold_score_rel, threshold_score_abs)

    # # debug
    # if threshold_score_rel > threshold_score_abs:
    #     msg = "!!!!!!Using relative texture threshold within LOIs!!!!!!"
    #     show_msg(msg)
    # msg = "Anchor LOI texture threshold: " +str(threshold)
    # show_msg(msg)

    for loi_id, corr_score, corr_ind in zip(loi_ids, corrs_scores, corrs_ind):
        if corr_score > threshold: # NOTE: it's probably dominated by the strict threshold_score_abs  
            loi_ids_anchored.append(loi_id)
            loi_vertex_ids_anchored.append(corr_ind)

    lois_anchored = OrderedVertices(ids=loi_ids_anchored, vertex_ids=loi_vertex_ids_anchored)
    
    return lois_anchored

def anchor_loi_shape_texture_consensus(lois, corrs_shape_feature, corrs_texture, V, threshold=10):
    '''
    Anchor good correspondence as landmarks by using the consensus of shape and texture correspondence.
    
    Args:
        threshold: in mm
    Returns:
    
    '''    
    
    loi_ids = lois.ids()
    loi_vertex_ids_anchored = []
    loi_ids_anchored = []
    
    for loi_id, corr_shape, corr_texture in zip(loi_ids, corrs_shape_feature, corrs_texture):
        if corrs_texture == -1: # avoid empty corrs
            continue
        dist = np.linalg.norm(V[corr_shape] - V[corr_texture])
        if dist < threshold:
            loi_ids_anchored.append(loi_id)
            loi_vertex_id = corr_texture # use ind from texture correspondence, should be more accurate than shape
            loi_vertex_ids_anchored.append(loi_vertex_id)
    lois_anchored = OrderedVertices(ids=loi_ids_anchored, vertex_ids=loi_vertex_ids_anchored)
            
    return lois_anchored

def anchor_lois_all(params, lois, corrs_shape_feature, corrs_texture, V, scores_texture, scores_texture_corrs,\
                     corrs_candidates_shape, avg_edge_length, return_cluster=True, debug=False):
    '''
    # NOTE: This step only helps with updating LOIs close to achored LOIs, far and single LOI cannot benefit from the update
    # Indicators for well-localized LOIs:
    # 1. high texture score (compared within LOIs and an absolute score)
    # 2. proximity of shape and texture correspondences <=> combined scores
    # 3. unambiguous texture (within search region)
    
    Args:
        threshold: in mm
    Returns:
    
    '''    

    # from params['anchor']
    std_factor_within_lois_texture = params['anchor']['std_factor_within_lois_texture']
    std_factor_unambiguous_texture = params['anchor']['std_factor_unambiguous_texture']
    factor_dist_unambiguous_texture = params['anchor']['factor_dist_unambiguous_texture']
    threshold_unambiguous_texture = params['anchor']['threshold_unambiguous_texture']

    threshold_texture_score =  params['update']['threshold_texture_score']
    threshold_shape_texture_consensus = params['update']['threshold_shape_texture_consensus']

    lois_anchored_1 = anchor_loi_texture_thresholding(lois, scores_texture_corrs, corrs_texture,\
                                                        factor_std=std_factor_within_lois_texture,\
                                                        threshold_score_abs=threshold_texture_score)    
    lois_anchored_2 = anchor_loi_shape_texture_consensus(lois, corrs_shape_feature, corrs_texture,\
                                                        V, threshold=threshold_shape_texture_consensus)    
    lois_anchored_3, corrs_candidates_texture_clusters = anchor_loi_unambiguous_texture(\
                                                        lois, scores_texture, corrs_candidates_shape, V, avg_edge_length=avg_edge_length,\
                                                        texture_threshold=threshold_unambiguous_texture, std_factor=std_factor_unambiguous_texture,\
                                                        dist_factor=factor_dist_unambiguous_texture, return_clusters=return_cluster)    

    if debug:
        msg = "new anchor from salient texture among LOIs: " + str(lois_anchored_1.ids())
        print(msg)
        # show_msg(msg)
        msg = "new anchor from consensus: " + str(lois_anchored_2.ids())
        print(msg)
        # show_msg(msg)
        msg = "new anchor from unambiguous texture within region: " + str(lois_anchored_3.ids())
        print(msg)

    lois_anchored_1.merge(lois_anchored_2)
    lois_anchored_1.merge(lois_anchored_3)
    if return_cluster:
        return lois_anchored_1, corrs_candidates_texture_clusters

    return lois_anchored_1