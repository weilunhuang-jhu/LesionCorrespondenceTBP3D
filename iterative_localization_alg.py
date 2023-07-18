import sys, os
sys.path.append("lib")
import math
import numpy as np
import time
import logging

from Anchor import anchor_lois_all
from Shape import find_corrs_candidates_shape
from Texture import get_scores_texture, find_loi_with_insignificant_signal, filter_candidates_with_insignificant_signal
from utils import load_params, save_output, show_msg, set_params, update_params, get_correspondence_highest_score, combine_scores, compute_error_stats, OrderedVertices
from echo_utils import get_hRadius, get_cached_ECHO_vert_ids, get_echo_descriptors, get_echos_in_alg
from MeshLoader import MeshData3DBodyTexTransformed

VERBAL = False
DEBUG = True

def setup_lm_loi(mesh_data_1, mesh_data_2, loi_excluded=OrderedVertices(), loi_localized=OrderedVertices()):
    """
    Set up landmarks and LOIs as input to the algorithm

    Args:
        mesh_data_1: 
        mesh_data_2: 
        loi_localized: NOTE: Defined in mesh 2, this should be empty at the inital iteration
        loi_excluded: NOTE: Defined in mesh 1, this should be empty at the final iteration
    Returns:
        landmarks_1:
        landmarks_2:
        lois_1:
    """
    num_landmarks_initial = len(mesh_data_1.landmarks_vertex_id_list)
    landmarks_1 = OrderedVertices(ids=np.arange(num_landmarks_initial), vertex_ids=mesh_data_1.landmarks_vertex_id_list)
    landmarks_2 = OrderedVertices(ids=np.arange(num_landmarks_initial), vertex_ids=mesh_data_2.landmarks_vertex_id_list)

    # NOTE: ids in landmarks 
    new_landmarks_1 = OrderedVertices(ids=np.arange(num_landmarks_initial, num_landmarks_initial + len(loi_localized)),\
                                      vertex_ids=mesh_data_1.loi_vertex_id_list[loi_localized.ids()])
    new_landmarks_2 = OrderedVertices(ids=np.arange(num_landmarks_initial, num_landmarks_initial + len(loi_localized)),\
                                      vertex_ids=loi_localized.vertex_ids())
    landmarks_1.merge(new_landmarks_1)
    landmarks_2.merge(new_landmarks_2)

    lois_1_ids = list(set(range(len(mesh_data_1.loi_vertex_id_list))) - set(loi_localized.ids()) - set(loi_excluded.ids()))
    lois_1_ids.sort()
    lois_1_vertex_ids = mesh_data_1.loi_vertex_id_list[lois_1_ids]
    lois_1 = OrderedVertices(ids=lois_1_ids, vertex_ids=lois_1_vertex_ids)

    # sanity check for new landmarks
    assert len(landmarks_1) == len(landmarks_2)

    return (landmarks_1 , landmarks_2, lois_1)

def exclude_loi(root_dir, mesh_name, loi_vertex_id_list, params, avg_edge_length, logger):
    """
    Exclude LOI with insignificant signal from the iterative algorithm

    Args:
        root_dir:  source
        mesh_name: 
        loi_vertex_id_list: NOTE: Defined in mesh 1, this should be empty list at the inital iteration
        -----------------------------------------------------------------------------------------------
        params:
        avg_edge_length:
        logger:
    Returns:
        lois_excluded:
    """
    # Get ECHO descriptors of source LOI first
    out_fname = os.path.join(root_dir, mesh_name).replace(".ply", "_vert_id_list.txt")
    np.savetxt(out_fname, loi_vertex_id_list, fmt="%d") # save vert_id_list for source
    hRadius = get_hRadius(params['texture']['echo_support_radii'][0], avg_edge_length)
    get_echo_descriptors(root_dir, mesh_name, params['texture']['echo_taus'][0], hRadius, params['texture']['echo_distance_type']) # use the smallest radius
    loi_ids_excluded = find_loi_with_insignificant_signal(root_dir, loi_vertex_id_list,\
                            tau=params['texture']['echo_taus'][0], support_radius=params['texture']['echo_support_radii'][0],\
                            avg_edge_length=avg_edge_length, threshold=params['texture']['threshold_texture_signal'])

    show_msg("LOIs excluded from the Alg: " + str(loi_ids_excluded), logger)

    # convert loi_ids_excluded to  lois_excluded TODO: move to somewhere else
    if len(loi_ids_excluded) == 0:
        return OrderedVertices()

    lois_excluded = OrderedVertices(ids=loi_ids_excluded, vertex_ids=loi_vertex_id_list[loi_ids_excluded])
    return lois_excluded

def run_single_iter_alg(iter_num, params, mesh_data_1, mesh_data_2, landmarks_1, landmarks_2, lois_1, lois_localized,\
                        echo_vertex_ids_prev, logger, use_anchor=True, verbal=False, debug=True):
    """
    Perform one iteration of the algorithm by:
        1. Find correspondence candidates
        2. Filter candidates

    Args:
        params
        mesh_data_1: 
        mesh_data_1: 
        landmarks_1:
        landmarks_2:
        lois_1:
        lois_localized:
        ----------------------------------------------------------------------------------------------- 
        echo_vertex_ids_prev: NOTE:
        logger
        use_anchor:
        verbal:
        debug:
    Returns:
        lois_localized
        lois_anchored
        corrs_shape_feature
        corrs_texture
        corrs_combined
        scores_shape
        scores_texture
        scores_combined
        corrs_candidates_shape_unfiltered
        corrs_candidates_shape
        corrs_candidates_texture_clusters
        echo_vertex_ids_prev
    """

    # Get params
    shape_feature_type = params['shape']['shape_feature_type']
    shape_metric_type = params['shape']['shape_metric_type']
    to_normalized = params['shape']['to_normalized']
    shape_score_type = params['shape']['shape_score_type']
    std_factor_shape = params['shape']['std_factor_shape']

    threshold_texture_signal = params['texture']['threshold_texture_signal']
    texture_metric_type = params['texture']['texture_metric_type']
    echo_taus = params['texture']['echo_taus']
    echo_support_radii = params['texture']['echo_support_radii']
    echo_distance_type = params['texture']['echo_distance_type']
    echo_fov_weights = params['texture']['echo_fov_weights']

    combined_method = params['combined_score']["combined_method"]
    score_weights =  params['combined_score']["score_weights"]

    radius_geodesic = params['update']['radius_geodesic']

    avg_edge_length = max(mesh_data_1.avg_edge_length, mesh_data_2.avg_edge_length)
    mesh_name = mesh_data_1.mesh_name

    # Correspondence from shape
    t0 = time.perf_counter()
    corrs_candidates_shape, scores_shape,\
    corrs_shape_feature, _, _= find_corrs_candidates_shape(landmarks_1.vertex_ids(), landmarks_2.vertex_ids(),\
                                        lois_1.vertex_ids(), mesh_data_1.dist_solver, mesh_data_2.dist_solver,\
                                        feature_type=shape_feature_type, metric=shape_metric_type, to_normalized=to_normalized,\
                                        radius=radius_geodesic, std_factor=std_factor_shape,\
                                        score_type=shape_score_type, return_scores=True)
    corrs_candidates_shape_unfiltered = corrs_candidates_shape.copy()
    t1 = time.perf_counter()
    if verbal:
        show_msg("time for shape corrs: " + str(t1-t0), logger)

    # Get echo descriptors
    t0 = time.perf_counter()
    echo_vertex_ids, echo_vertex_ids_prev = get_echos_in_alg(iter_num, echo_vertex_ids_prev, corrs_candidates_shape,\
                     echo_taus, echo_support_radii, echo_distance_type,\
                     lois_1.vertex_ids(), mesh_data_1.root_dir, mesh_data_2.root_dir, mesh_name, avg_edge_length, debug=verbal)

    if verbal:
        show_msg("Acquiring: " + str(len(echo_vertex_ids)) + " ECHO descritpors...", logger)
    t1 = time.perf_counter()
    if verbal:
        show_msg("time for ECHO descriptors: " + str(t1-t0), logger)

    # Incorporate with texture info, bounded by correspondence candidiates from shape
    t0 = time.perf_counter()

    # Update correspondence candidates from shape and associated scores by removing vertices with weak signals
    corrs_candidates_shape, scores_shape = filter_candidates_with_insignificant_signal(mesh_data_2.root_dir, corrs_candidates_shape, scores_shape,\
                                 echo_taus[0], echo_support_radii[0], avg_edge_length, threshold=threshold_texture_signal)

    # Get texture scores, [0, 1]
    scores_texture = get_scores_texture(mesh_data_1.root_dir, mesh_data_2.root_dir, lois_1.vertex_ids(), corrs_candidates_shape,\
         avg_edge_length, fov_weights=echo_fov_weights, echo_descriptors_target_all=None, score_type=texture_metric_type)

    # Correspondence from texture
    temp = get_correspondence_highest_score(scores_texture) # ind in corrs_candidates_shape
    corrs_texture = [(ele[ele_id] if ele_id >= 0 else -1) for ele, ele_id in zip(corrs_candidates_shape, temp)]
    scores_texture_corrs = [(ele[ele_id] if ele_id >= 0 else 0) for ele, ele_id in zip(scores_texture, temp)]

    # Correspondence from combined score
    scores_combined = combine_scores(scores_shape, scores_texture, method=combined_method, weights=score_weights)
    temp = get_correspondence_highest_score(scores_combined) # ind in corrs_candidates_shape
    corrs_combined = [(ele[ele_id] if ele_id >= 0 else -1) for ele, ele_id in zip(corrs_candidates_shape, temp)]
    scores_combined_corrs = [(ele[ele_id] if ele_id >= 0 else 0) for ele, ele_id in zip(scores_combined, temp)]

    t1 = time.perf_counter()
    if verbal:
        show_msg("time for texture corrs: " + str(t1-t0), logger)

    # Anchor LOIs
    t0 = time.perf_counter()
    if not use_anchor:
        corrs_candidates_texture_clusters = None # dummy
        loi_vertex_ids_anchored = [ele for ele in corrs_combined]
        lois_anchored = OrderedVertices(ids=lois_1.ids(), vertex_ids=loi_vertex_ids_anchored)
    else:
        # Anchor LOIs as new landmarks
        lois_anchored, corrs_candidates_texture_clusters = anchor_lois_all(params, lois_1, corrs_shape_feature, corrs_texture, mesh_data_2.V,\
                                        scores_texture, scores_texture_corrs, corrs_candidates_shape, mesh_data_2.avg_edge_length, debug=VERBAL)

    # Update localized LOIs
    lois_localized.merge(lois_anchored)

    t1 = time.perf_counter()
    if verbal:
        show_msg("time for anchors: " + str(t1-t0), logger)
    if DEBUG:
        show_msg("Num of localized LOI: " + str(len(lois_localized)), logger)
        show_msg("Lesion ids: {}".format(str(lois_localized.ids())), logger)
        show_msg("Lesion vertex ids: {}".format(str(lois_localized.vertex_ids())), logger)

    return (lois_localized, lois_anchored,\
            corrs_shape_feature, corrs_texture, corrs_combined,\
            scores_shape, scores_texture, scores_combined,\
            corrs_candidates_shape_unfiltered, corrs_candidates_shape, corrs_candidates_texture_clusters,\
            echo_vertex_ids_prev)

def main():


    import argparse
    # argument set-up
    parser = argparse.ArgumentParser(description="Find lesion correspodnence of from source to the target")
    parser.add_argument("-s", "--source", type=str, help="Path to the source patient folder")
    parser.add_argument("-t", "--target", type=str, help="Path to the target patient folder")
    parser.add_argument("-p", "--params", type=str, default="params.yml", help="Path to the parameter file")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode (store true)")
    parser.add_argument("-v", "--verbal", action="store_true", help="Verbal (store true)")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("-outflag", "--outflag", default="_iterative_alg", type=str, help="Flag for output file name")


    # Parse the command line arguments to an object
    args = parser.parse_args()
    if not args.source or not args.target:
        print("No input folder is provided.")
        print("For help type --help")
        exit()

    DEBUG = args.debug
    VERBAL = args.verbal
    OUTFLAG = args.outflag
    SAVE_DIR = args.output_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    # uniformed naming
    mesh_name ="real_scale_in_mm.ply"
    texture_name = "model_highres_0_normalized.png"
    landmark_name = "landmarks.txt"
    loi_name = "lesion_of_interest.txt" 

    # load data
    root_dir_1 = args.source
    root_dir_2 = args.target
    subject_name = os.path.basename(os.path.normpath(root_dir_2)) # use target as the subject
    mesh_data_1 = MeshData3DBodyTexTransformed(root_dir_1, mesh_name, texture_name, landmark_name, loi_name, use_trimesh=False, use_pvmesh=False)
    mesh_data_2 = MeshData3DBodyTexTransformed(root_dir_2, mesh_name, texture_name, landmark_name, loi_name, use_trimesh=False, use_pvmesh=False)
    avg_edge_length = max(mesh_data_1.avg_edge_length, mesh_data_2.avg_edge_length)

    # Logger
    logging.basicConfig(filename= os.path.join(SAVE_DIR, "log_" + subject_name + OUTFLAG + ".log"),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Variables for the algorithm ###
    params = load_params(args.params)
    num_iter = params["alg"]["num_iter"]
    radius_geodesic_multiplier = math.exp((math.log(params['shape']['radius_geodesic_max']) - \
                                            math.log(params['shape']['radius_geodesic_min']))/(num_iter - 1)) # < 1
    lois_excluded = OrderedVertices()
    lois_localized = OrderedVertices()

    # First step to remove LOIs with insignificant texture out of the workflow
    lois_excluded = exclude_loi(root_dir_1, mesh_name, mesh_data_1.loi_vertex_id_list, params, avg_edge_length, logger)

    if VERBAL:
        show_msg(str(params), logger)
        
    # TODO: add sanity check for the number of cached ECHO in echo_utils
    # NOTE: The number of echo descriptors at each support radius folder should be the same, otherwise use the least one to avoid error
    hRadius = get_hRadius(params['texture']['echo_support_radii'][0], avg_edge_length) 
    echo_dir = os.path.join(root_dir_2, "echo_descriptors" + "_tau_" + str(params['texture']['echo_taus'][0]) + "_h_" + str(hRadius))
    echo_vertex_ids_prev = get_cached_ECHO_vert_ids(echo_dir=echo_dir) # unique ids of vertices that have been queried for ECHO descriptors in the previous iteration

    params = set_params(params, params['shape']['radius_geodesic_min'], params['anchor']['threshold_texture_score'],\
                        params['anchor']['threshold_shape_texture_consensus'])

    # START ITERATION
    for iter_num in range(num_iter):

        if VERBAL or DEBUG:
            show_msg("=====ITER " + str(iter_num) + "=====", logger)
        if VERBAL:
            show_msg("Search region (radius geodesic): " + str(params['update']['radius_geodesic']), logger)
            show_msg("Anchor criteria: (thresh_texture, thresh_consensus) = " + str((params['update']['threshold_texture_score'], params['update']['threshold_shape_texture_consensus'])), logger)
        
        # Set up landmarks and LOIs
        landmarks_1, landmarks_2, lois_1 = setup_lm_loi(mesh_data_1, mesh_data_2, lois_excluded, lois_localized)
        show_msg("landmarks_1: {}; landmarks_2: {}; lois: {}".format(len(landmarks_1), len(landmarks_2), len(lois_1)), logger)

        # Early ending
        if len(lois_1) == 0:
            show_msg("Early ending at iteration " + str(iter_num), logger)
            break

        # main alg
        (lois_localized, lois_anchored,\
        corrs_shape_feature, corrs_texture, corrs_combined,\
        scores_shape, scores_texture, scores_combined,\
        corrs_candidates_shape_unfiltered, corrs_candidates_shape, corrs_candidates_texture_clusters,\
        echo_vertex_ids_prev) = run_single_iter_alg(iter_num, params, mesh_data_1, mesh_data_2,\
                                        landmarks_1, landmarks_2, lois_1, lois_localized,\
                                        echo_vertex_ids_prev, logger, use_anchor=True, verbal=VERBAL, debug=DEBUG)

        # Update parameters: increase num of candidates, relax criteria to anch LOIs
        params = update_params(params, radius_geodesic_multiplier, params['anchor']['threshold_texture_multiplier'],\
                               params['anchor']['threshold_consensus_multiplier'])

        output_data = save_output(subject_name, OUTFLAG, iter_num, lois_1, lois_localized,\
                                corrs_shape_feature, corrs_texture, corrs_combined,\
                                scores_shape, scores_texture, scores_combined,\
                                corrs_candidates_shape_unfiltered, corrs_candidates_shape, corrs_candidates_texture_clusters, save_dir=SAVE_DIR)

    # END ITERATION
    iter_num = "final"
    if len(lois_localized) == len(mesh_data_1.loi_vertex_id_list):
        print("All lesions are localized.")
    else:
        # Reset landmarks and lois, setting empty lois_excluded
        landmarks_1, landmarks_2, lois_1 = setup_lm_loi(mesh_data_1, mesh_data_2, OrderedVertices(), lois_localized)
        show_msg("landmarks_1: {}; landmarks_2: {}; lois: {}".format(len(landmarks_1), len(landmarks_2), len(lois_1)), logger)
        params = set_params(params, params['shape']['radius_geodesic_max'], params['update']['threshold_texture_score'],\
                            params['update']['threshold_shape_texture_consensus'])
        if VERBAL or DEBUG:
            show_msg("=====ITER " + str(iter_num) + "=====", logger)
        if VERBAL:
            show_msg("Search region (radius geodesic): " + str(params['update']['radius_geodesic']), logger)

        (lois_localized, lois_anchored,\
        corrs_shape_feature, corrs_texture, corrs_combined,\
        scores_shape, scores_texture, scores_combined,\
        corrs_candidates_shape_unfiltered, corrs_candidates_shape, corrs_candidates_texture_clusters,\
        echo_vertex_ids_prev) = run_single_iter_alg(iter_num, params, mesh_data_1, mesh_data_2,\
                                        landmarks_1, landmarks_2, lois_1, lois_localized,\
                                        echo_vertex_ids_prev, logger, use_anchor=False, verbal=VERBAL, debug=DEBUG)

        output_data = save_output(subject_name, OUTFLAG, iter_num, lois_1, lois_localized,\
                                corrs_shape_feature, corrs_texture, corrs_combined,\
                                scores_shape, scores_texture, scores_combined,\
                                corrs_candidates_shape_unfiltered, corrs_candidates_shape, corrs_candidates_texture_clusters, save_dir=SAVE_DIR)

    show_msg("========Result ==========", logger)
    show_msg(compute_error_stats(mesh_data_2, output_data), logger)

if __name__ == "__main__":
    main()