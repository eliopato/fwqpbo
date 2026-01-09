import scipy.io
import numpy as np


# update data_param with information retrieved from MATLAB file
# (arranged according to ISMRM fat-water toolbox)
def updateDataParams(data_param, file):
    data_param['fileType'] = 'MATLAB'
    try:
        mat = scipy.io.loadmat(file)
    except:
        raise Exception('Could not read MATLAB file {}'.format(file))
    data = mat['imDataParams'][0, 0]

    for i in range(0, 4):
        if len(data[i].shape) == 5:
            img = data[i]  # Image data (row,col,slice,coil,echo)
        elif data[i].shape[1] > 2:
            echo_times = data[i][0]  # TEs [sec]
        else:
            if data[i][0, 0] > 1:
                data_param['B0'] = data[i][0, 0]  # Fieldstrength [T]
            else:
                clockwise = data[i][0, 0]  # Clockwiseprecession?

    if clockwise != 1:
        raise Exception('Warning: Not clockwise precession. ' +
                        'Need to write code to handle this case!')

    data_param['ny'], data_param['nx'], data_param['nz'], nCoils, data_param['nb_echoes'] = img.shape
    if nCoils > 1:
        raise Exception('Warning: more than one coil. ' +
                        'Need to write code to coil combine!')

    # Get only slices in data_param['sliceList']
    if 'sliceList' not in data_param:
        data_param['sliceList'] = range(data_param['nz'])
    else:
        img = img[:, :, data_param['sliceList'], :, :]
        data_param['nz'] = len(data_param['sliceList'])
    # Get only echoes in data_param['echoes']
    data_param['nb_echoes'] = data_param['nb_echoes']
    if not 'echoes' in data_param:
        data_param['echoes'] = range(data_param['nb_echoes'])
    else:
        img = img[:, :, :, :, data_param['echoes']]
        echo_times = echo_times[data_param['echoes']]
        data_param['nb_echoes'] = len(data_param['echoes'])
    if 'cropFOV' in data_param:
        x0, x1 = data_param['cropFOV'][0], data_param['cropFOV'][1]
        y0, y1 = data_param['cropFOV'][2], data_param['cropFOV'][3]
        data_param['Nx'], data_param['nx'] = data_param['nx'], x1-x0
        data_param['Ny'], data_param['ny'] = data_param['ny'], y1-y0
        img = img[y0:y1, x0:x1, :, :, :]
    if data_param['nb_echoes'] < 2:
        raise Exception(
            'At least 2 echoes required, only {} given'.format(data_param['nb_echoes']))
    data_param['t1'] = echo_times[0]
    data_param['dt'] = np.mean(np.diff(echo_times))
    if np.max(np.diff(echo_times))/data_param['dt'] > 1.05 or np.min(
      np.diff(echo_times))/data_param['dt'] < .95:
        raise Exception('Warning: echo inter-spacing varies more than 5%')

    data_param['frameList'] = []

    data_param['dx'], data_param['dy'], data_param['dz'] = 1.5, 1.5, 5  # Ad hoc assumption on voxelsize

    # To get data as: (echo,slice,row,col)
    img.shape = (data_param['ny'], data_param['nx'], data_param['nz'], data_param['nb_echoes'])
    img = np.transpose(img)
    img = np.swapaxes(img, 2, 3)

    data_param['img'] = img*data_param['reScale']


# Save output as MATLAB arrays
def save(output, data_param):
    data_param['outDir'].mkdir(parents=True, exist_ok=True)
    filename = data_param['outDir'] / './{}.mat'.format(data_param['sliceList'][0])
    print(r'Writing images to "{}"'.format(filename))
    scipy.io.savemat(filename, output)
