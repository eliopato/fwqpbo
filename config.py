# import configparser
import DICOM
import MATLAB
import numpy as np
from pathlib import Path
import yaml


# extract data parameter object representing a single slice
def getSliceDataParams(data_param, slice, z):
    slice_data_params = dict(data_param)
    slice_data_params['sliceList'] = [slice]
    slice_data_params['img'] = data_param['img'][:, [z], :, :]
    slice_data_params['nz'] = 1
    return slice_data_params


# extract data_param object representing a slab of contiguous slices starting at z
def getSlabDataParams(data_param, slices, z):
    slab_data_params = dict(data_param)
    slab_data_params['sliceList'] = slices
    slab_size = len(slices)
    slab_data_params['img'] = data_param['img'][:, z:z+slab_size, :, :]
    slab_data_params['nz'] = slab_size
    return slab_data_params


# Update algorithm parameter object algo_param and set default parameters
def setupAlgoParams(algo_param, N, nFAC=0):
    defaults = [
        ('nR2', 1),
        ('R2max', 100.),
        ('R2cand', [0.]),
        ('mu', 1.),
        ('nB0', 100),
        ('nICMiter', 0),
        ('multiScale', False),
        ('use3D', False),
        ('magnitudeDiscrimination', True),
        ('offresPenalty', 0.)

    ]

    for param, defval in defaults:
        if param not in algo_param:
            algo_param[param] = defval

    if 'graphcut' not in algo_param:
        algo_param['graphcut'] = 'graphcutlevel' in algo_param
    
    if algo_param['graphcut']:
        if 'graphcutlevel' not in algo_param:
            algo_param['graphcutLevel'] = 0
    else:
        algo_param['graphcutLevel'] = None

    if 'realEstimates' in algo_param:
        if not algo_param['realEstimates'] and N==2:
            raise Exception('Real-valued estimates needed for two-point Dixon')
    elif N==2:
        algo_param['realEstimates'] = True
    else:
        algo_param['realEstimates'] = False

    if algo_param['nR2'] > 1:
        algo_param['R2step'] = algo_param['R2max']/(algo_param['nR2']-1)  # [sec-1]
    else:
        algo_param['R2step'] = 1.0  # [sec-1]
    
    algo_param['iR2cand'] = np.array(list(set([min(algo_param['nR2']-1, int(R2/algo_param['R2step']))
                            for R2 in algo_param['R2cand']])))  # [msec]

    algo_param['maxICMupdate'] = round(algo_param['nB0']/10)

    # For Fatty Acid Composition, create algorithmParams for two passes: algo_param and algo_param['pass2']
    # First pass: use standard fat-water separation to determine B0 and R2*
    # Second pass: use B0- and R2*-maps from first pass
    if nFAC > 0:
        algo_param['pass2'] = dict(algo_param)  # modify algoParams for pass 2:
        algo_param['pass2']['nICMiter'] = 0  # to omit ICM
        algo_param['pass2']['graphcutLevel'] = None  # to omit the graphcut
        algo_param['pass2']['graphcut'] = False
    
    algo_param['output'] = ['wat', 'fat', 'ff', 'B0map']
    if algo_param['realEstimates']:
        algo_param['output'].append('phi')
    if (algo_param['nR2'] > 1):
        algo_param['output'].append('R2map')
    if (nFAC > 2):
        algo_param['output'].append('CL')
    if (nFAC > 1):
        algo_param['output'].append('PUD')
    if (nFAC > 0):
        algo_param['output'].append('UD')


# Get relative weights alpha of fat resonances based on CL, UD, and PUD per UD
def getFACalphas(CL=None, P2U=None, UD=None):
    P = 11  # Expects one water and ten triglyceride resonances
    M = [CL, UD, P2U].count(None)+2
    alpha = np.zeros([M, P], dtype=np.float32)
    alpha[0, 0] = 1.  # Water component
    if M == 2:
        # F = 9A+(6(CL-4)+UD(2P2U-8))B+6C+4UD(1-P2U)D+6E+2UDP2UF+2G+2H+I+2UDJ
        alpha[1, 1:] = [9, 6*(CL-4)+UD*(2*P2U-8), 6, 4*UD*(1-P2U), 6, 2*UD*P2U,
                        2, 2, 1, UD*2]
    elif M == 3:
        # F1 = 9A+6(CL-4)B+6C+6E+2G+2H+I
        # F2 = (2P2U-8)B+4(1-P2U)D+2P2UF+2J
        alpha[1, 1:] = [9, 6*(CL-4), 6, 0, 6, 0, 2, 2, 1, 0]
        alpha[2, 1:] = [0, 2*P2U-8, 0, 4*(1-P2U), 0, 2*P2U, 0, 0, 0, 2]
    elif M == 4:
        # F1 = 9A+6(CL-4)B+6C+6E+2G+2H+I
        # F2 = -8B+4D+2J
        # F3 = 2B-4D+2F
        alpha[1, 1:] = [9, 6*(CL-4), 6, 0, 6, 0, 2, 2, 1, 0]
        alpha[2, 1:] = [0, -8, 0, 4, 0, 0, 0, 0, 0, 2]
        alpha[3, 1:] = [0, 2, 0, -4, 0, 2, 0, 0, 0, 0]
    elif M == 5:
        # F1 = 9A-24B+6C+6E+2G+2H+I
        # F2 = -8B+4D+2J
        # F3 = 2B-4D+2F
        # F4 = 6B
        alpha[1, 1:] = [9, -24, 6, 0, 6, 0, 2, 2, 1, 0]
        alpha[2, 1:] = [0, -8, 0, 4, 0, 0, 0, 0, 0, 2]
        alpha[3, 1:] = [0, 2, 0, -4, 0, 2, 0, 0, 0, 0]
        alpha[4, 1:] = [0, 6, 0, 0, 0, 0, 0, 0, 0, 0]
    return alpha


# Update model parameter object model_param and set default parameters
def setupModelParams(model_param, clockwisePrecession=False, temperature=None):

    defaults = [
        ('fatCS', [1.3]),
        ('nFAC', 0),
        ('CL', 17.4), # Derived from Lundbom 2010
        ('P2U', 0.2),  # Derived from Lundbom 2010
        ('UD', 2.6),  # Derived from Lundbom 2010
    ]

    for param, defval in defaults:
        if param not in model_param:
            model_param[param] = defval

    if 'watCS' not in model_param:
        if temperature: # Temperature dependence according to Hernando 2014
            model_param['watCS'] = 1.3 + 3.748 -.01085 * temperature # Temp in [°C]
        else:
            model_param['watCS'] = 4.7
    
    model_param['CS'] = np.array([model_param['watCS']] + model_param['fatCS'], dtype=np.float32)
    
    if clockwisePrecession:
        model_param['CS'] *= -1
    
    model_param['P'] = len(model_param['CS'])

    if model_param['nFAC'] > 0 and model_param['P'] != 11:
        raise Exception(
            'FAC excpects exactly one water and ten triglyceride resonances')
    
    model_param['M'] = 2+model_param['nFAC']

    if model_param['nFAC'] == 0:
        model_param['alpha'] = np.zeros([model_param['M'], model_param['P']], dtype=np.float32)
        model_param['alpha'][0, 0] = 1.
        if 'relAmps' in model_param:
            for (p, a) in enumerate(model_param['relAmps']):
                model_param['alpha'][1, p+1] = float(a)
        else:
            for p in range(1, model_param['P']):
                model_param['alpha'][1, p] = float(1/len(model_param['fatCS']))
    elif model_param['nFAC'] == 1:
        model_param['alpha'] = getFACalphas(model_param['CL'], model_param['P2U'])
    elif model_param['nFAC'] == 2:
        model_param['alpha'] = getFACalphas(model_param['CL'])
    elif model_param['nFAC'] == 3:
        model_param['alpha'] = getFACalphas()
    else:
        raise Exception('Unknown number of FAC parameters: {}'
                        .format(model_param['nFAC']))

    # For Fatty Acid Composition, create modelParams for two passes: model_param and model_param['pass2']
    # First pass: use standard fat-water separation to determine B0 and R2*
    # Second pass: do the Fatty Acid Composition
    if model_param['nFAC'] > 0: 
        model_param['pass2'] = dict(model_param) # copy model_param into pass 2, then modify pass 1
        model_param['alpha'] = getFACalphas(model_param['CL'], model_param['P2U'], model_param['UD'])
        model_param['M'] = model_param['alpha'].shape[0]


# group slices in sliceList in slabs of reconSlab contiguous slices
def getSlabs(slice_list, reconSlab):
    slabs = []
    slices = []
    pos = 0
    for z, slice in enumerate(slice_list):
        # start a new slab
        if slices and (len(slices) == reconSlab or not slice == slices[-1]+1):
            slabs.append((slices, pos))
            slices = [slice]
            pos = z
        else:
            slices.append(slice)
    slabs.append((slices, pos))
    return slabs

    
# Update data param object, set default parameters and read data from files
def setupDataParams(data_param: dict, out_dir:str|None=None) -> None:
    if out_dir:
        data_param['outDir'] = Path(out_dir)
    elif 'outDir' in data_param:
        data_param['outDir'] = Path(data_param['outDir'])
    else:
        raise Exception('No outDir defined')

    defaults = [
        ('reScale', 1.0),
        ('temperature', None),
        ('clockwisePrecession', False),
        ('offresCenter', 0.),
        ('files', []),
        ('isEnhanced', False)
    ]

    for param, defval in defaults:
        if param not in data_param:
            data_param[param] = defval

    if 'files' in data_param:
        data_param['files'] = [data_param['configPath'] / file for file in list(data_param['files']) if Path(data_param['configPath'] / file).is_file()]
    
    if 'dirs' in data_param:
        data_param['dirs'] = [data_param['configPath'] / dir for dir in list(data_param['dirs']) if Path(data_param['configPath'] / dir).is_dir()]
        for path in data_param['dirs']:
            data_param['files'] += [obj for obj in path.iterdir() if obj.is_file()]
    
    validFiles = DICOM.getValidFiles(data_param['files'])
    
    if validFiles:
        DICOM.updateDataParams(data_param, validFiles)
    else:
        if len(data_param['files']) == 1 and data_param['files'][0].suffix == '.mat':
            MATLAB.updateDataParams(data_param, data_param['files'][0])
        else:
            raise Exception('No valid files found')
    
    if 'reconSlab' in data_param:
        data_param['slabs'] = getSlabs(data_param['sliceList'], data_param['reconSlab'])


def readConfig(file: str) -> dict:
    """ Read .yaml configuration file located at file and return a dictionnary"""
    file = Path(file)
    with open(file, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            raise Exception(f'Error reading config file {file}') from exc
    config['configPath'] = file.parent
    return config