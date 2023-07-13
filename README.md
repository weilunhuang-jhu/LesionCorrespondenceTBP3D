# Lesion Correspondence TBP 3D 

This code-base implements the method presented in the paper [Skin Lesion Correspondence Localization in
Total Body Photography]()
Given a set of lesions of interest in a soruce textured 3D mesh, we would like to localize lesion correspondence in a target textured 3D mesh. 


## Installation

* conda: conda-forge channel
* Essential: trimesh, vtk, pyqt5, pyvista, pyvistaqt, [potpourri3d](https://github.com/nmwsharp/potpourri3d)
* Preprocessing: pymeshlab
* submodule: [skin3d](https://github.com/jeremykawahara/skin3d), [ECHODescriptors](https://github.com/weilunhuang-jhu/ECHODescriptors.git)



## Documentation

- [Test 1](#test-1)


## Data

## Note

* In skin3d, modify path for dir_annotate and dir_multi_annotate in BodyTexDataset (bodytex.py)

## Acknowledgement

* Skin3D
* 3DBodyTex

