# QUICKSTART

Install dependencies by running `conda env create -f environment.yml` from a terminal located in the fwqpbo folder, with Conda installed.

Once the required packages are installed, update the parameter files as needed and run the following command:
```
./main.py -d params/data_batch.yml -a params/algo_batch.yml -m params/model_batch.yml
```

Each parameter file allows for testing several configurations. The format for defining different configurations is:
```
default:
    param1: default_value1
    param2: default_value2
    param3: default_value3

config_name1:
    param1: value1

config_name2:
    param2: value2
    param3: value3
```

If a configuration named "default" exists, it will be used to set parameters that are not redefined in the other configurations (hence overiding the default paramers defined in the code). The default configuration is also ran as a configuration itself.

The config names are used to create output folders. They can contain slashes to organize outputs in subfolders, for example a data parameter file like 
```
default:
    out_dir: /home/user/output/FWQPBO/

3t/brain:
    dirs: [/home/user/input/brain/phase, /home/user/input/brain/magnitude]
    slice_list: [12, 13]

3t/liver:
    dirs: [/home/user/input/liver/phase, /home/user/input/liver/magnitude]
    slice_list: [24, 25]
```
will create the following subfolders:
```
/home/user/output/FWQPBO/default/
/home/user/output/FWQPBO/3t/brain/
/home/user/output/FWQPBO/3t/liver/
```
# DATA PARAMETERS

Parameters used to select the input data (FOV, echoes, slices, etc).

## Required parameters

`dirs`: list of input directories containing the DICOM. You can only process one acquisition at the time.

`out_dir`: directory to store output files in. Will be created if it doesn't exist.

`update_existing_outputs`: set to True to run all configs and overide any existing output, or False to skip configs that already have all the existing outputs.

## Optional parameters

`rescale`: rescale the value of the pixels in the input images. Default: 1 (no rescaling). Type: float.

`slice_list`: if set, will only process the slices specified in the list. Slice indexes start from 0. Default: all slices. Type: list of int.

`echoes`: if set, will only user the echoes specified in the list. Use the index of the echoes, not the time of echo, to specify echoes. `echoes: [0, 1, 2]` will select the first three echoes. Default: all slices. Type: list of int.

`temperature`: Temperature in Celsius degrees to use as a parameter to calculate the water chemical shift model parameter (Hernando 2014). The resulting formula is : `water_cs = 1.3 + 3.748 -.01085 * temperature`. Default: 32. Type: float

`crop_fov`: reduce the image field of view to use for WF reconstruction. The cropped pixels will be set to 0 in the output maps, so that the output images have the same dimensions as the input. Default: no crop. Type: list of int ([x0, x1, y0, y1])

`slabs_size`: process the volume in slabs of `slabs_size` slices. The resulting image will have the same number of slices as the input. Default: 1. Type: int

`clockwise_precession`: if set to true, the chemical shift model parameter is multiplied by -1. Default: False. Type: boolean

`offres_center`: Off-resonnance center. Default: 0. Type: int.

`seg` : subcategory to define a segmentation file and expected values to compare output maps with.
- `file`: path to the segmentation file, must be a nifti file
- `expected_values`: subcategory holding for each map type (e.g. ff, r2_map, etc) the expected values of each label
    - `[map name]`
        - `label1: expected value1`
        - `label2: expected value2`
        - ...

## Example file content
```
phantom: 
    dirs: [/home/user/path/to/dicom/folder/phase/, /home/user/path/to/folder/magnitude/]
    out_dir: /home/user/path/to/output/folder/
    seg: 
      file: /home/user/path/to/segmentation.nii.gz
      expected_values:
        ff:
          1: 0
          2: 25
          3: 100
    rescale: 0.0001
    slice_list: [0, 1, 2, 3]
    echoes: [0, 1, 2]
    temperature: 37
    crop_fov: [64, 128, 64, 92] 
    slabs_size: 2
    clockwise_precession: False
    offres_center: 0.
```


# MODEL PARAMETERS

This file defines the parameters that define the signal, depending on water and fat chemical shift. 

All values are optional.

`wat_cs`: Water Chemical Shift, in ppm. Default: 4.7. Type: float.

`fat_cs`: List of Fat Chemical Shift, in ppm. Default: [1.3]. Type: list of floats.

`rel_amps`: Relative amplitudes at the fat chemical shifts listed in `fat_cs`, normalized so that their sum=1. Only used if `n_fac` = 0. Default: all peaks have the same amplitude. Type: list of floats.

`n_fac`: Number of fatty acid composition parameters. Possible values: 0, 1, 2, 3. Default: 0. Type: int.

`CL`: Fatty acid carbon chain length. Only used if n_fac > 0. Default: 17.4. Type: float.

`UD`: Fatty acid carbon unsaturation degree. Only used if n_fac > 0.Default: 2.6. Type: float.

`P2U`: Fatty acid carbon polyunsaturation degree per unsaturation degree (PUD per UD). Only used if n_fac > 0. Default: 0.2. Type: float.


## Example file content:
```
ISMRM2012:
  fat_cs:   [ 5.3,    4.31,   2.76,   2.1,    1.3,    0.9]
  rel_amps: [ 0.048,  0.039,  0.004,  0.128,  0.693,  0.087]
  wat_cs: 4.7
  n_fac: 0
```

# ALGORITHM PARAMETERS

All values are optional.

`n_r2`: Number of R<sub>2</sub> steps. Default: 1. Type: int

`r2_max`: Max R<sub>2</sub>, in s<sup>-1</sup>. Default: 100. Type: float

`r2_cand`: R<sub>2</sub><sup>*</sup> candidates in s<sup>-1</sup>. Default: [0.]. Type: list of float

`mu`: Regularization parameter to balance data smoothness in B<sub>0</sub> map correction. Default: 1. Type: float

`n_b0`: Number of steps per period. Default: 100. Type: int

`n_icm_iter`: Number of Iterated Conditional Modes iterations. Default: 0. Type: int

`multiscale`: Whether or not to use the multiscale frameword. Default: False. Type: Boolean

`use_3D`: Whether or not to process the volume slice by slice (2D), or all slices together (3D). Default: False. Type: Boolean

`magnitude_discrimination`: Default: True. Type: Boolean

`offres_penalty`: Off-resonance penalty. Default: 0. Type: Float

## Example file content:

```
default:
  n_r2: 145
  r2_max: 144.0
  r2_cand: [40.]
  mu: 0.1
  n_b0: 100
  n_icm_iter: 10
  graphcut: True
  multiscale: True
  use_3D: True
  magnitude_discrimination: False
```