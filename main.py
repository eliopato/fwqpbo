#!/usr/bin/env python3

import numpy as np
import sys
import optparse
import config
import fat_water_separation

gyro = 42.58  # 1H gyromagnetic ratio

s
# Merge output for slices reconstructed separately
def merged_output_slices(output_list):
    merged_output = output_list[0]
    for output in output_list[1:]:
        for series_type in output:
            merged_output[series_type] = np.concatenate((merged_output[series_type], output[series_type]))
    return merged_output


def get_fatty_acid_composition(rho):
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


# Get total fat component (for Fatty Acid Composition; trivial otherwise)
def get_fat(rho, alpha):
    fat = np.zeros(rho.shape[1:], dtype=complex)
    for m in range(1, alpha.shape[0]):
        fat += sum(alpha[m, 1:])*rho[m]
    return fat


def reconstruct(data_param: dict, algo_param: dict, model_param: dict) -> dict:
    """Perform fat/water separation and return prescribed output.
    The output is a dict where keys are map names, and values are the numpy array images."""

    # Do the fat/water separation
    rho, b0_map, r2_map = fat_water_separation.reconstruct(data_param, algo_param, model_param)
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
        rho = fat_water_separation.reconstruct(data_param, algo_param['pass2'], model_param['pass2'], b0_map, r2_map)[0]
        CL, UD, PUD = get_fatty_acid_composition(rho)
    
        if 'CL' in algo_param['output']:
            output['CL'] = CL
        if 'UD' in algo_param['output']:
            output['UD'] = UD
        if 'PUD' in algo_param['output']:
            output['PUD'] = PUD

    return output


def main(data_param_filepath: str, algo_param_filepath: str, model_param_filepath: str, out_dir=None):
    # Read configuration files
    data_param = config.read_configfile(data_param_filepath)
    algo_param = config.read_configfile(algo_param_filepath)
    model_param = config.read_configfile(model_param_filepath)

    # Setup configuration objects
    frame_list = config.setup_data_params(data_param, out_dir)
    config.setup_model_params(model_param, data_param['clockwise_precession'], data_param['temperature'])
    config.setup_algo_params(algo_param, data_param['nb_echoes'], model_param['n_fac'])

    print(f'B0 = {round(frame_list.b0, 2)}')
    print(f'N echoes = {frame_list.nb_echoes}')
    print(f't1/dt = {round(frame_list.t1*1000, 2)}/{round(frame_list.dt*1000, 2)} msec')
    print(f'nx,ny,nz = {frame_list.nx},{frame_list.ny},{frame_list.nz}')
    print(f'dx,dy,dz = {round(frame_list.dx, 2)},{round(frame_list.dy, 2)},{round(frame_list.dz, 2)}')

    # Run fat/water processing and save output
    if algo_param['use_3D'] or frame_list.n_frame_indexes == 1:
        if 'slabs' in data_param:
            for i_slab, (slices, z) in enumerate(data_param['slabs']):
                print(f'Processing slab {i_slab+1}/{len(data_param['slabs'])} (slices {slices[0]+1}-{slices[-1]+1})...')
                slab_data_params = config.get_slab_data_params(data_param, slices, z)
                output = reconstruct(slab_data_params, algo_param, model_param)
                save(output, slab_data_params) # save data slab-wise to save memory
        else:
            output = reconstruct(frame_list, data_param, algo_param, model_param)
            save(output, frame_list, data_param)
    else:
        output = []
        for z, slice in enumerate(data_param['slice_list']):
            print(f'Processing slice {slice+1} ({z+1}/{len(data_param['slice_list'])})...')
            slice_data_params = config.get_slice_data_params(data_param, slice, z)
            output.append(reconstruct(slice_data_params, algo_param, model_param))
        save(merged_output_slices(output), frame_list, data_param)


if __name__ == '__main__':
    # Initiate command line parser
    p = optparse.OptionParser()
    p.add_option('--data_param_filepath', '-d', default='',  type='string', help='File path of data parameter configuration file')
    p.add_option('--algo_param_filepath', '-a', default='',  type='string', help='File path of algorithm parameter configuration file')
    p.add_option('--model_param_filepath', '-m', default='',  type='string', help='File path of model parameter configuration file')

    # Parse command line
    options, arguments = p.parse_args()

    main(options.data_param_filepath, options.algo_param_filepath, options.model_param_filepath)