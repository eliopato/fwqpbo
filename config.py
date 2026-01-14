import numpy as np
from pathlib import Path
import yaml
import dicom_tools
from dicom_processing import FrameCollection
import copy

# extract data parameter object representing a single slice
def get_slice_data_params(frame_coll: FrameCollection, slice: int):
    new_frame_coll = copy(frame_coll)
    new_frame_coll.user_params['slice_list'] = [slice]
    new_frame_coll.img = new_frame_coll.img[:, [slice], :, :]
    return frame_coll


# extract data_param object representing a slab of contiguous slices starting at z
def get_slab_data_params(frame_coll: FrameCollection, slices: list[int]):
    new_frame_coll = copy(frame_coll)
    new_frame_coll.user_params['slice_list'] = slices
    slab_size = len(slices)
    new_frame_coll.img = new_frame_coll.img[:, slice:slice + slab_size, :, :]
    return frame_coll


# Update algorithm parameter object algo_param and set default parameters
def setup_algo_params(algo_param, N, n_fac=0):

    defaults = [
        ('n_r2', 1),
        ('r2_max', 100.),
        ('r2_cand', [0.]),
        ('mu', 1.),
        ('n_b0', 100),
        ('n_icm_iter', 0),
        ('multiscale', False),
        ('use_3D', False),
        ('magnitude_discrimination', True),
        ('offres_penalty', 0.)
    ]

    for param, defval in defaults:
        if param not in algo_param:
            algo_param[param] = defval

    if 'graphcut' not in algo_param:
        algo_param['graphcut'] = 'graphcutlevel' in algo_param
    
    if algo_param['graphcut']:
        if 'graphcutlevel' not in algo_param:
            algo_param['graph_cut_level'] = 0
    else:
        algo_param['graph_cut_level'] = None

    if 'real_estimates' in algo_param:
        if not algo_param['real_estimates'] and N==2:
            raise Exception('Real-valued estimates needed for two-point Dixon')
    elif N == 2:
        algo_param['real_estimates'] = True
    else:
        algo_param['real_estimates'] = False

    if algo_param['n_r2'] > 1:
        algo_param['r2_step'] = algo_param['r2_max']/(algo_param['n_r2']-1)  # [sec-1]
    else:
        algo_param['r2_step'] = 1.0  # [sec-1]
    
    ir2 = [min(algo_param['n_r2']-1, int(R2/algo_param['r2_step'])) for R2 in algo_param['r2_cand']]
    algo_param['i_r2_cand'] = np.array(list(set(ir2)))  # [msec]

    algo_param['max_icm_update'] = round(algo_param['n_b0']/10)

    # For Fatty Acid Composition, create algorithmParams for two passes: algo_param and algo_param['pass2']
    # First pass: use standard fat-water separation to determine B0 and R2*
    # Second pass: use B0- and R2*-maps from first pass
    if n_fac > 0:
        algo_param['pass2'] = dict(algo_param)  # modify algoParams for pass 2:
        algo_param['pass2']['n_icm_iter'] = 0  # to omit icm
        algo_param['pass2']['graph_cut_level'] = None  # to omit the graphcut
        algo_param['pass2']['graphcut'] = False
    
    algo_param['output'] = ['wat', 'fat', 'ff', 'b0_map']
    if algo_param['real_estimates']:
        algo_param['output'].append('phi')
    if (algo_param['n_r2'] > 1):
        algo_param['output'].append('r2_map')
    if (n_fac > 2):
        algo_param['output'].append('CL')
    if (n_fac > 1):
        algo_param['output'].append('PUD')
    if (n_fac > 0):
        algo_param['output'].append('UD')


# Get relative weights alpha of fat resonances based on CL, UD, and PUD per UD
def get_fac_alphas(CL = None, P2U = None, UD = None):
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
def setup_model_params(model_param, clockwise_precession=False, temperature=None):

    defaults = [
        ('fat_cs', [1.3]),
        ('n_fac', 0),
        ('CL', 17.4), # Derived from Lundbom 2010
        ('P2U', 0.2),  # Derived from Lundbom 2010
        ('UD', 2.6),  # Derived from Lundbom 2010
    ]

    for param, defval in defaults:
        if param not in model_param:
            model_param[param] = defval

    if 'wat_cs' not in model_param:
        if temperature: # Temperature dependence according to Hernando 2014
            model_param['wat_cs'] = 1.3 + 3.748 -.01085 * temperature # Temp in [°C]
        else:
            model_param['wat_cs'] = 4.7
    
    model_param['CS'] = np.array([model_param['wat_cs']] + model_param['fat_cs'], dtype=np.float32)
    
    if clockwise_precession:
        model_param['CS'] *= -1
    
    model_param['P'] = len(model_param['CS'])

    if model_param['n_fac'] > 0 and model_param['P'] != 11:
        raise Exception('FAC excpects exactly one water and ten triglyceride resonances')
    
    model_param['M'] = 2 + model_param['n_fac']

    if model_param['n_fac'] == 0:
        model_param['alpha'] = np.zeros([model_param['M'], model_param['P']], dtype=np.float32)
        model_param['alpha'][0, 0] = 1.
        if 'rel_amps' in model_param:
            for (p, a) in enumerate(model_param['rel_amps']):
                model_param['alpha'][1, p+1] = float(a)
        else:
            for p in range(1, model_param['P']):
                model_param['alpha'][1, p] = float(1/len(model_param['fat_cs']))
    elif model_param['n_fac'] == 1:
        model_param['alpha'] = get_fac_alphas(model_param['CL'], model_param['P2U'])
    elif model_param['n_fac'] == 2:
        model_param['alpha'] = get_fac_alphas(model_param['CL'])
    elif model_param['n_fac'] == 3:
        model_param['alpha'] = get_fac_alphas()
    else:
        raise Exception(f"Unknown number of FAC parameters: {model_param['n_fac']}")

    # For Fatty Acid Composition, create modelParams for two passes: model_param and model_param['pass2']
    # First pass: use standard fat-water separation to determine B0 and R2*
    # Second pass: do the Fatty Acid Composition
    if model_param['n_fac'] > 0: 
        model_param['pass2'] = dict(model_param) # copy model_param into pass 2, then modify pass 1
        model_param['alpha'] = get_fac_alphas(model_param['CL'], model_param['P2U'], model_param['UD'])
        model_param['M'] = model_param['alpha'].shape[0]

    
# Update data param object, set default parameters and read data from files
def setup_data_params(data_param: dict, out_dir:str|None=None):
    if out_dir:
        data_param['out_dir'] = Path(out_dir)
    elif 'out_dir' in data_param:
        data_param['out_dir'] = Path(data_param['out_dir'])
    else:
        raise Exception('No out_dir defined')

    defaults = [
        ('rescale', 1.0),
        ('temperature', None),
        ('clockwise_precession', False),
        ('offres_center', 0.),
        ('files', [])
    ]

    for param, defval in defaults:
        if param not in data_param:
            data_param[param] = defval

    if 'files' in data_param:
        data_param['files'] = [data_param['config_path'] / file for file in list(data_param['files']) if Path(data_param['config_path'] / file).is_file()]
    
    if 'dirs' in data_param:
        data_param['dirs'] = [data_param['config_path'] / dir for dir in list(data_param['dirs']) if Path(data_param['config_path'] / dir).is_dir()]
        for path in data_param['dirs']:
            data_param['files'] += [obj for obj in path.iterdir() if obj.is_file()]
    
    valid_files = dicom_tools.get_valid_files(data_param['files'])
    
    if not valid_files:
        raise Exception('No valid files found')
    
    data_param['files'] = valid_files



def read_configfile(file: str) -> dict:
    """ Read .yaml configuration file located at file and return a dictionnary"""
    file = Path(file)
    with open(file, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            raise Exception(f'Error reading config file {file}') from exc
    config['config_path'] = file.parent
    return config