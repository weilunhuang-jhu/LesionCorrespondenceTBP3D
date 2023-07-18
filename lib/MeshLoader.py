from packaging.version import parse as parse_version
import os
import pyvista as pv
import numpy as np
import potpourri3d as pp3d
import trimesh
import igl

from utils import find_closest_vertex

### Custom mesh loader for Skin3D dataset
class MeshData3DBodyTexTransformed(object):
    def __init__(self, root_dir, ply_filename="real_scale_in_mm.ply", texture_filename="model_highres_0_normalized.png",\
                landmark_filename=None, loi_filename=None,\
                use_trimesh=True, use_pvmesh=True, use_prominent=False, use_path_solver=False, debug=True):
        '''
            Input:
                root_dir: root directory of the mesh
                ply_filename: filename of the mesh in .ply format
                texture_filename: filename of the texture in .png format
                landmark_filename: filename of the landmarks in .txt format
                loi_filename: filename of the LOI (lesion of interest) in .txt format
                use_trimesh: whether to load the mesh in trimesh format
                use_pvmesh: whether to load the mesh in pyvista format
                use_prominent: whether to use prominent landmarks (as defined in Skin3D)
                use_path_solver: whether to use path solver (from potpourri3d)
                debug: whether to print debug information

            Member variables:
                root_dir: root directory of the mesh
                mesh_name: name of the mesh (basename)
                V: vertices
                F: faces
                dist_solver: distance solver (from potpourri3d)
                avg_edge_length: average edge length of the mesh
                path_solver: path solver (from potpourri3d)
                mesh_trimesh: mesh in trimesh format
                mesh_pv_processed: mesh in pyvista format
                texture_pv: texture in pyvista format
                landmarks_vertex_id_list: list of vertex ids of landmarks
                landmarks_coord_list: list of coordinates of landmarks
                loi_vertex_id_list: list of vertex ids of LOI
                loi_coord_list: list of coordinates of LOI
        '''

        self.root_dir = root_dir
        self.mesh_name = ply_filename # basename
        ply_filename = os.path.join(root_dir, ply_filename)
        texture_filename = os.path.join(root_dir, texture_filename)
         
        if debug:
            print("loading " + root_dir)

        # basics and solvers
        self.V, self.F = pp3d.read_mesh(ply_filename)
        self.dist_solver = pp3d.MeshHeatMethodDistanceSolver(self.V, self.F)
        self.avg_edge_length = igl.avg_edge_length(self.V, self.F)
        if use_path_solver:
            self.path_solver = pp3d.EdgeFlipGeodesicSolver(self.V, self.F) # shares precomputation for repeated solves

       # meshes
        self.mesh_trimesh = None
        self.mesh_pv_processed = None
        if use_trimesh:
            self.mesh_trimesh = trimesh.load(ply_filename, process=False) # avoid re-ordering of vertices in trimesh
        if use_pvmesh:
            self.mesh_pv_processed = pv.read(ply_filename)
            self.texture_pv = pv.read_texture(texture_filename)
            if parse_version(pv.__version__) > parse_version("0.38"):
                assert self.mesh_pv_processed.is_all_triangles
            else:
                assert self.mesh_pv_processed.is_all_triangles()
      
        # landmarks and LOI
        self.landmarks_vertex_id_list = []
        self.landmarks_coord_list = []
        self.loi_vertex_id_list = []
        self.loi_coord_list = []

        if landmark_filename:
            landmark_filename = os.path.join(root_dir, landmark_filename)
            self.load_landmark(landmark_filename)

        if loi_filename:
            loi_filename = os.path.join(root_dir, loi_filename)
            self.load_loi(loi_filename, use_prominent)
            
        if debug:
            if use_trimesh:
                print("trimesh")
                print("V: " + str(self.mesh_trimesh.vertices.shape))
                print("F: " + str(self.mesh_trimesh.faces.shape))
            if use_pvmesh:
                print("pv processed")
                print("V: " + str(self.mesh_pv_processed.points.shape))
                faces_pv = self.mesh_pv_processed.faces.reshape((-1,4))[:, 1:4]
                print("F: " + str(faces_pv.shape))
            print("pp3d")
            print("V: " + str(self.V.shape))
            print("F: " + str(self.F.shape))

    def load_landmark(self, filename):
        """
        Load vertex index and coordinate for landmakrs, save in self.landmarks_vertex_id_list and self.landmarks_coord_list

        Args:
            filename
        """
        # For each landmark in mesh, get the vertex id for the nearest vertex
        # NOTE: Discretization error comes for using vertex rather than point, does not matter because landmarks come from rough estimation
        landmarks_vert_id = [] # index of vertex
        landmarks_coords = [] # coordinate of lanmark
        with open(filename, "r") as landmark_file:
            for landmark_info in landmark_file.readlines():
                x, y, z = landmark_info.split(" ")
                landmark = [float(ele) for ele in [x,y,z]]
                landmarks_coords.append(landmark)
                closest_vertex_id = find_closest_vertex(landmark, self.V)
                landmarks_vert_id.append(closest_vertex_id)

        self.landmarks_vertex_id_list = np.array(landmarks_vert_id)
        self.landmarks_coord_list = np.array(landmarks_coords)

    def load_loi(self, filename, use_prominent=False):
        """
        Load vertex index and coordinate for LOI (lesion of interest), save in self.loi_vertex_id_list and self.loi_coord_list

        Args:
            filename
            use_prominent: use only the first 10 LOI (prominent lesiobs are defined in Skin3D dataset), default False
        """
        # Note that the id is in vertex order of transformed mesh file, different from the raw mesh file
        loi_coord_list = []
        loi_id_list = [] 
        with open(filename, "r") as loi_file:
            for loi_info in loi_file.readlines():
                vert_id, x, y, z = loi_info.split(",") # NOTE: the first number in LOI file is vert id
                vert_id = int(vert_id)
                loi_id_list.append(vert_id)
                coord = [float(ele) for ele in [x,y,z]]
                loi_coord_list.append(coord)

        self.loi_vertex_id_list = np.array(loi_id_list)
        self.loi_coord_list = np.array(loi_coord_list)
        
        if use_prominent:
            self.loi_vertex_id_list = self.loi_vertex_id_list[:10]
            self.loi_coord_list = self.loi_coord_list[:10]