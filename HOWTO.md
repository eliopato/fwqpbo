# HOW TO USE
First install required packages, see dependencies in `environment.yml`.

Example command-line usage:
```
./main.py -d params/data.yml -a params/algo_3D.yml -m params/model.yml
```

Example parameter files are available in the `params` folder. 

See below for details on each files' content.

# DATA PARAMETERS

Parameters used to select the input data (FOV, echoes, slices, etc).

## Required parameters

```
dirs: [/home/user/path/to/dicom/folder/phase/, /home/user/path/to/folder/magnitude/]
out_dir: /home/user/path/to/output/folder/
```

`dirs`: list of input directories containing the DICOM. You can only process one acquisition at the time.

`out_dir`: directory to store output files in. Will be created if it doesn't exist.

## Optional parameters

```
rescale: 0.0001
slice_list: [0, 1, 2, 3]
echoes: [0, 1, 2]
temperature: 37
crop_fov: [64, 128, 64, 92] 
slabs_size: 2
clockwise_precession: False
offres_center: 0.
```

`rescale`: rescale the value of the pixels in the input images. Default: 1. Type: float.

`slice_list`: if set, will only process the slices specified in the list. Slice indexes start from 0. Default: all slices. Type: list of int.

`echoes`: if set, will only user the echoes specified in the list. Use the index of the echoes, not the time of echo, to specify echoes. `echoes: [0, 1, 2]` will select the first three echoes. Default: all slices. Type: list of int.

`temperature`: Temperature in Celsius degrees to use as a parameter to calculate the water chemical shift model parameter (Hernando 2014). The resulting formula is : `water_cs = 1.3 + 3.748 -.01085 * temperature`. Default: 32. Type: float

`crop_fov`: reduce the image field of view to use for WF reconstruction. The cropped pixels will be set to 0 in the output maps, so that the output images have the same dimensions as the input. Default: no crop. Type: list of int ([x0, x1, y0, y1])

`slabs_size`: process the volume in slabs of `slabs_size` slices. The resulting image will have the same number of slices as the input. Default: 1. Type: int

`clockwise_precession`: if set to true, the chemical shift model parameter is multiplied by -1. Default: False. Type: boolean

`offres_center`: Default: 0. Type: int.


# MODEL PARAMETERS

All values are optional.

```
fat_cs: [5.3, 4.31, 2.76, 2.1, 1.3, 0.9]
rel_amps: [0.048, 0.039, 0.004, 0.128, 0.693, 0.087]
wat_cs: 4.7
n_fac: 0
```
`fat_cs`: Default: [1.3]. Type: list of floats.

`rel_amps`: Only used if `n_fac` = 0. Type: list of floats.

`wat_cs`: Type: float.

`n_fac`: Default: 0. Type: float



# ALGORITHM PARAMETERS

All values are optional.

```
n_r2: 145
r2_max: 144.0
r2_cand: [40.]
mu: 0.1
n_b0: 100
n_icm_iter: 10
graphcut: True
multiscale: True
use_3D: False
magnitude_discrimination: False
```

`n_r2`: Default: 1. Type: int

`r2_max`: Default: 100. Type: float

`r2_cand`: Default: [0.]. Type: list of float

`mu`: Default: 1. Type: float

`n_b0`: Default: 100. Type: int

`n_icm_iter`: Default: 0. Type: int

`multiscale`: Default: False. Type: Boolean


`use_3D`: Default: False. Type: Boolean

`magnitude_discrimination`: Default: True. Type: Boolean

`offres_penalty`: Default: 0. Type: Float