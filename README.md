# Lesion Correspondence TBP 3D 

This code-base implements the method presented in the paper [Skin Lesion Correspondence Localization in
Total Body Photography]().
Given a set of lesions of interest in a soruce textured 3D mesh, we would like to localize lesion correspondence in a target textured 3D mesh.

![Proposed Method](https://i.imgur.com/M7NaLER.png)

## Installation

* conda: conda-forge channel
* Essential: trimesh, vtk, pyqt5, pyvista, pyvistaqt, yaml, [potpourri3d](https://github.com/nmwsharp/potpourri3d)
* Preprocessing: pymeshlab, pandas
* submodule: [skin3d](https://github.com/jeremykawahara/skin3d), [ECHODescriptors](https://github.com/weilunhuang-jhu/ECHODescriptors.git)

```
conda create --name tbp_lesion_corr -c conda-forge python=3 pyvista pyvistaqt trimesh pyqt
conda activate tbp_lesion_corr
pip install potpourri3d pymeshlab pandas yaml
```

Please compile [ECHODescriptors](https://github.com/weilunhuang-jhu/ECHODescriptors.git) and modify the path under <b>echo/bin</b> in [params.yml](https://github.com/weilunhuang-jhu/LesionCorrepsondenceTBP3D/blob/main/params.yml) to the path of the <b>GetDescriptor</b> exe/bin file.

## Data

The textured 3D human mesh model can be downloaded: [3DBodyTex.v1](https://cvi2.uni.lu/datasets/).
The original skin lesion annotation on the 3DBodyTex data for longitudinal tracking can be found in [skin3d](https://github.com/jeremykawahara/skin3d).

**NOTE**: We use the <b>high</b> resolution meshes in the 3DBodyTex dataset. 

The directory structure for each subject should follow the hierarchy below. Please refer to [Preprocess data](#Preprocess-data). Annotation of landmarks and lesion of interest after <b>pre-processing</b> can be found [here](https://drive.google.com/file/d/1mRB62xgqOnT1BdjLQQ6CC2T0NrvKILy8/view?usp=sharing).

```
${Subject}
|-- echo_descriptors_tau_x.xxxxx_h_x (optional, will be generated during run-time if missing)
|   |-- vert_xxxxx.txt
|   `-- ...
|-- echo_descriptors_tau_y.yyyyy_h_y (optional, will be generated during run-time if missing)
|   |-- vert_xxxxx.txt
|   `-- ...
|-- echo_descriptors_tau_z.zzzzz_h_z (optional, will be generated during run-time if missing)
|   |-- vert_xxxxx.txt
|   `-- ...
|-- real_scale_in_mm.ply
|-- model_highres_0_normalized.png
|-- texture_signal_per_vertex.txt
|-- landmarks.txt
`-- lesion_of_interest.txt
```

## Preprocess data

In [script/](https://github.com/weilunhuang-jhu/LesionCorrepsondenceTBP3D/blob/main/script)

- Transform data: (convert to mm scale)
```
python transform_3dbodytex.py -i path_to_3dbodytex_data
```

- Get scalar per vertex texture signal:
```
python get_color_per_vertex.py -i path_to_output_from_previous_step
```

## Usage

### Run the proposed algorithm

- Localize lesion correspondence:

```
python iterative_localization_alg.py -s path_to_source_folder -t path_to_target_folder -o path_to_output_folder
```
The localized lesion correspondence in each iteration is saved in the <b>pkl</b> file (one <b>pkl</b> file per iteration), with a final iteration to complete the correspondence of the remaining LOIs using weighted average of geometric score and texture score.

### Visualization

In [script/](https://github.com/weilunhuang-jhu/LesionCorrepsondenceTBP3D/blob/main/script)

- Visualize landmarks and LOIs:

```
python visualize_landmarks_and_lois.py -s path_to_source_folder -t path_to_target_folder
```

- Visualize correspondence:

```
python visualize_correspondence.py -s path_to_source_folder -t path_to_target_folder -r path_to_output_pkl_file
```

![Visualization of correspondence](https://i.imgur.com/AA9nEQM.png)

## Acknowledgement

The repo is built on the excellent work and dataset: [Skin3D](https://github.com/jeremykawahara/skin3d), [potpourri3d](https://github.com/nmwsharp/potpourri3d), [ECHODescriptors](https://github.com/mkazhdan/EchoDescriptors), and [3DBodyTex.v1](https://cvi2.uni.lu/datasets/). Thanks for these great projects.