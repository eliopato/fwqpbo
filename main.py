#!/usr/bin/env python3

import datetime
import optparse
import config
import os
import numpy as np

from fat_water_separation import reconstruct_main
from dicom_processing import read_input_images, save
from tools import get_slabs, print_dt, merged_output_slices, load_nifti_as_array

if __name__ == '__main__':

    start_time_main = datetime.datetime.now()
        
    # Initiate command line parser
    p = optparse.OptionParser()
    p.add_option('--data_param_filepath', '-d', default='',  type='string', help='File path of data parameter configuration file')
    p.add_option('--algo_param_filepath', '-a', default='',  type='string', help='File path of algorithm parameter configuration file')
    p.add_option('--model_param_filepath', '-m', default='',  type='string', help='File path of model parameter configuration file')

    # Parse command line
    options, arguments = p.parse_args()

    data_params = config.read_configfile(options.data_param_filepath)
    algo_params = config.read_configfile(options.algo_param_filepath)
    model_params = config.read_configfile(options.model_param_filepath)

    data_defaults = data_params['default'] if 'default' in data_params else dict()
    algo_defaults = algo_params['default'] if 'default' in algo_params else dict()
    model_defaults = data_params['default'] if 'default' in algo_params else dict()
    
    config_stats = dict()
    
    for algo_configname in algo_params:
        
        config_stats[algo_configname] = dict()

        for model_configname in model_params:
            
            config_stats[algo_configname][model_configname] = dict()

            for data_configname in data_params:
                
                start_time = datetime.datetime.now()
                print('--------------------------------------------------------------------------------------')
                print_dt(f'Processing data {data_configname} with algo {algo_configname} and model {model_configname}')
                
                # set default params
                algo_dict = config.set_default(algo_defaults, algo_params[algo_configname])
                data_dict = config.set_default(data_defaults, data_params[data_configname])
                model_dict = config.set_default(model_defaults, model_params[model_configname])

                if 'dirs' not in data_dict and 'files' not in data_dict:
                    print_dt('skip data config, no input files defined')
                    continue

                root_out_dir = data_dict['out_dir']
                config_out_dir = f'{root_out_dir}/{data_configname}/{algo_configname}_{model_configname}/'
                os.makedirs(config_out_dir, exist_ok=True)
                data_dict['out_dir'] = config_out_dir

                print_dt(f'Detecting valid files for data {data_configname}')
                data_dict = config.detect_valid_files(data_dict)
                if data_dict is None: 
                    continue
                
                # setup data params and read input images
                print_dt('Reading input images')                
                frame_coll = read_input_images(data_dict)    
                if 'slabs_size' in data_dict:
                    frame_coll.user_params['slabs'] = get_slabs(frame_coll.slice_indexes, data_dict['slabs_size'])

                # setup model and algo parameters
                config.setup_model_params(model_dict, data_dict['clockwise_precession'], data_dict['temperature'])
                config.setup_algo_params(algo_dict, frame_coll.n_echo, model_dict['n_fac'])

                # don't run the same config if output files already exist and config is not set to rerun configs
                missing_output = False
                if not data_dict['update_existing_outputs']:
                    for out_folder in algo_dict['output']:
                        if not os.path.exists(f'{config_out_dir}/{out_folder}'):
                            missing_output = True
                            break
                else:
                    missing_output = True
                
                if missing_output:
                    # save parameters to a file (same name as the config out dir minus /)
                    with open(f'{config_out_dir.rstrip("/")}_params.txt', mode='w') as f:
                        f.write(str({'data': data_dict, 'algo': algo_dict, 'model': model_dict}))

                    print_dt(f'B0 = {round(frame_coll.b0, 2)}')
                    print_dt(f'N echoes = {frame_coll.n_echo} ({frame_coll.echo_times})')
                    print_dt(f't1/dt = {round(frame_coll.t1*1000, 2)}/{round(frame_coll.dt*1000, 2)} msec')
                    print_dt(f'nx,ny,nz = {frame_coll.nx}, {frame_coll.ny}, {frame_coll.n_slice_indexes}')
                    print_dt(f'dx,dy,dz = {round(frame_coll.dx, 2)}, {round(frame_coll.dy, 2)}, {round(frame_coll.dz, 2)}')

                    # Run fat/water processing and save output
                    
                    if not algo_dict['use_3D'] or 'slabs' in frame_coll.user_params:
                        output = []
                        
                        if 'slabs' in frame_coll.user_params: 
                            for n_slab, (slices, _) in enumerate(frame_coll.user_params['slabs']):
                                print_dt(f'Processing slab {n_slab+1}/{len(frame_coll.user_params['slabs'])} (slices {slices[0]+1}-{slices[-1]+1})...')
                                output.append(reconstruct_main(frame_coll, algo_dict, model_dict, selected_slices=slices))
                        elif not algo_dict['use_3D']:
                            for z_slice in range(frame_coll.n_slice_indexes):
                                print_dt(f'Processing slice {z_slice+1}/{frame_coll.n_slice_indexes}...')
                                output.append(reconstruct_main(frame_coll, algo_dict, model_dict, selected_slices=[z_slice]))
                        else:
                            raise Exception('Error: cant do slab processing if use_3D is set to False, please update the data_params.yml file')
                        output = merged_output_slices(output)
                        
                    elif algo_dict['use_3D']:
                        output = reconstruct_main(frame_coll, algo_dict, model_dict)

                    save(output, frame_coll)

                if frame_coll.seg is not None:
                    print_dt('Computing stats with expected labels per segmented region')
                    current_stats_dict = dict()
                    for map_type in frame_coll.seg['expected_values']:
                        map_filepath = f"{frame_coll.user_params['out_dir']}/{map_type}.nii.gz"
                        if not os.path.exists(map_filepath):
                            print_dt(f'Expected output map file for {map_type} doesnt exist in {map_filepath}')
                            continue
                        map_vol_map = load_nifti_as_array(map_filepath)
                        seg_vol_map = frame_coll.seg['vol'].copy()
                        current_stats_dict[map_type] = dict()
                        for (label, expected_value) in frame_coll.seg['expected_values'][map_type].items():
                            roi = map_vol_map[seg_vol_map == label]
                            current_stats_dict[map_type][str(expected_value)] = {'mean': np.nan if len(roi) == 0 else roi.mean(),
                                                                                 'std': np.nan if len(roi) == 0 else roi.std()}
                    config_stats[algo_configname][model_configname][data_configname] = current_stats_dict
                
                with open(f'{root_out_dir}/stats.txt', mode='w') as f:
                    f.write(str(config_stats))

                print(f'Config run time: {datetime.datetime.now() - start_time}')
                
    print(f'Full script run time: {datetime.datetime.now() - start_time_main}')