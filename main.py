#!/usr/bin/env python3

import numpy as np
import datetime
import sys
import optparse
import config
import fat_water_separation
import dicom_processing
from dicom_tools import get_slabs, print_dt


def merged_output_slices(output_list: list) -> dict:
    """Merge output for slices/slabs reconstructed separately
    Return a dict with the numpy array for each image type (ff, water, etc)"""
    merged_output = output_list[0]
    for output in output_list[1:]:
        for series_type in output:
            merged_output[series_type] = np.concatenate((merged_output[series_type], output[series_type]))
    return merged_output


def get_fatty_acid_composition(rho: list) -> tuple[float, float, float]:
    """calculate UD, UD and PUD or UD, PUD and CL dependant on the number of fatty acid components (length(rho) -2)"""
    n_fac = len(rho) - 2 # Number of Fatty Acid Composition Parameters
    eps = sys.float_info.epsilon
    CL, UD, PUD = None, None, None

    if n_fac == 1:
        # UD = F2/F1
        UD = np.abs(rho[2] / (rho[1] + eps))
    elif n_fac == 2:
        # UD = F2/F1
        # PUD = F3/F1
        UD = np.abs(rho[2] / (rho[1] + eps))
        PUD = np.abs(rho[3] / (rho[1] + eps))
    elif n_fac == 3:
        # UD = F2/F1
        # PUD = F3/F1
        # CL = F4/F1
        UD = np.abs(rho[2] / (rho[1] + eps))
        PUD = np.abs(rho[3] / (rho[1] + eps))
        CL = np.abs(rho[4] / (rho[1] + eps))
    else:
        raise Exception('Unknown number of Fatty Acid Composition parameters: {}'.format(n_fac))

    return CL, UD, PUD


def get_fat(rho, alpha) -> np.array:
    """Get total fat component (for Fatty Acid Composition; trivial otherwise)"""
    fat = np.zeros(rho.shape[1:], dtype=complex)
    for m in range(1, alpha.shape[0]):
        fat += sum(alpha[m, 1:])*rho[m]
    return fat


def reconstruct(frame_coll: dicom_processing.FrameCollection, algo_param: dict, model_param: dict, selected_slices:list[int]|None=None) -> dict:
    """Perform fat/water separation and return prescribed output.
    The output is a dict where keys are map names, and values are the numpy array images."""

    if selected_slices is None:
        print_dt('Start reconstruction')
    else:
        print_dt(f'Start reconstruction of slices {selected_slices}')

    # Do the fat/water separation
    rho, b0_map, r2_map = fat_water_separation.reconstruct(frame_coll, algo_param, model_param, selected_slices=selected_slices)
    wat = rho[0]
    fat = get_fat(rho, model_param['alpha'])

    # Prepare prescribed output
    output = {}
    if 'wat' in algo_param['output']:
        output['wat'] = np.abs(wat)
    if 'fat' in algo_param['output']:
        output['fat'] = np.abs(fat)
    if 'phi' in algo_param['output']:
        output['phi'] = np.angle(wat, deg=True) + 180
    if 'ip' in algo_param['output']: # Calculate synthetic in-phase
        output['ip'] = np.abs(wat + fat)
    if 'op' in algo_param['output']: # Calculate synthetic opposed-phase
        output['op'] = np.abs(wat - fat)
    if 'ff' in algo_param['output']: # Calculate the fat fraction
        if algo_param['magnitude_discrimination']:  # to avoid bias from noise
            output['ff'] = 100 * np.real(fat / (wat + fat + sys.float_info.epsilon))
        else:
            output['ff'] = 100 * np.abs(fat)/(np.abs(wat) + np.abs(fat) + sys.float_info.epsilon)
    if 'b0_map' in algo_param['output']:
        output['b0_map'] = b0_map
    if 'r2_map' in algo_param['output']:
        output['r2_map'] = r2_map

    # Do any Fatty Acid Composition in a second pass
    if model_param['n_fac'] > 0:
        print_dt('Start Fatty Acid Composition ')
        rho = fat_water_separation.reconstruct(frame_coll, algo_param['pass2'], model_param['pass2'], b0_map, r2_map, selected_slices=selected_slices)[0]
        CL, UD, PUD = get_fatty_acid_composition(rho)
    
        if 'CL' in algo_param['output']:
            output['CL'] = CL
        if 'UD' in algo_param['output']:
            output['UD'] = UD
        if 'PUD' in algo_param['output']:
            output['PUD'] = PUD

    return output


def main(data_param_filepath: str, algo_param_filepath: str, model_param_filepath: str):
    start_time = datetime.datetime.now()
    print_dt('Reading config files')
    data_param = config.read_configfile(data_param_filepath)
    config.setup_data_params(data_param)
    algo_param = config.read_configfile(algo_param_filepath)
    model_param = config.read_configfile(model_param_filepath)

    # setup data params and read input images
    print_dt('Reading input images')
    
    frame_coll = dicom_processing.read_input_images(data_param)    
    if 'slabs_size' in frame_coll.user_params:
        frame_coll.user_params['slabs'] = get_slabs(frame_coll.slice_indexes, data_param['slabs_size'])

    # setsup model and algo parameters
    config.setup_model_params(model_param, data_param['clockwise_precession'], data_param['temperature'])
    config.setup_algo_params(algo_param, frame_coll.n_echo, model_param['n_fac'])

    print_dt(f'B0 = {round(frame_coll.b0, 2)}')
    print_dt(f'N echoes = {frame_coll.n_echo} ({frame_coll.echo_times})')
    print_dt(f't1/dt = {round(frame_coll.t1*1000, 2)}/{round(frame_coll.dt*1000, 2)} msec')
    print_dt(f'nx,ny,nz = {frame_coll.nx}, {frame_coll.ny}, {frame_coll.n_slice_indexes}')
    print_dt(f'dx,dy,dz = {round(frame_coll.dx, 2)}, {round(frame_coll.dy, 2)}, {round(frame_coll.dz, 2)}')

    # Run fat/water processing and save output
    
    if not algo_param['use_3D'] or 'slabs' in frame_coll.user_params:
        output = []
        
        if 'slabs' in frame_coll.user_params: 
            for n_slab, (slices, _) in enumerate(frame_coll.user_params['slabs']):
                print_dt(f'Processing slab {n_slab+1}/{len(frame_coll.user_params['slabs'])} (slices {slices[0]+1}-{slices[-1]+1})...')
                output.append(reconstruct(frame_coll, algo_param, model_param, selected_slices=slices))
        elif not algo_param['use_3D']:
            for z_slice in frame_coll.slice_indexes:
                print_dt(f'Processing slice {z_slice+1}/{frame_coll.n_slice_indexes}...')
                output.append(reconstruct(frame_coll, algo_param, model_param, selected_slices=[z_slice]))
        else:
            raise Exception('Error: cant do slab processing if use_3D is set to False, please update the data_params.yml file')
        output = merged_output_slices(output)
        
    elif algo_param['use_3D']:
        output = reconstruct(frame_coll, algo_param, model_param)

    dicom_processing.save(output, frame_coll)
    print(f'Total run time: {datetime.datetime.now() - start_time}')
        


if __name__ == '__main__':
    # Initiate command line parser
    p = optparse.OptionParser()
    p.add_option('--data_param_filepath', '-d', default='',  type='string', help='File path of data parameter configuration file')
    p.add_option('--algo_param_filepath', '-a', default='',  type='string', help='File path of algorithm parameter configuration file')
    p.add_option('--model_param_filepath', '-m', default='',  type='string', help='File path of model parameter configuration file')

    # Parse command line
    options, arguments = p.parse_args()

    main(options.data_param_filepath, options.algo_param_filepath, options.model_param_filepath)