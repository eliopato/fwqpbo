#!/usr/bin/env python3

import numpy as np
import sys
import optparse
import config
import fatWaterSeparation
import DICOM
import MATLAB


gyro = 42.58  # 1H gyromagnetic ratio


# Zero pad back any cropped FOV
def padCropped(croppedImage, data_param):
    if 'cropFOV' in data_param:
        image = np.zeros((data_param['nz'], data_param['Ny'], data_param['Nx']))
        x1, x2 = data_param['cropFOV'][0], data_param['cropFOV'][1]
        y1, y2 = data_param['cropFOV'][2], data_param['cropFOV'][3]
        image[:, y1:y2, x1:x2] = croppedImage
        return image
    else:
        return croppedImage


def save(output, data_param):
    for seriesType in output: # zero pad if was cropped and reshape to row,col,slice
        output[seriesType] = np.moveaxis(padCropped(output[seriesType].reshape((data_param['nz'], data_param['ny'], data_param['nx'])), data_param), 0, -1)
    
    if data_param['fileType'] == 'DICOM':
        DICOM.save(output, data_param)
    elif data_param['fileType'] == 'MATLAB':
        MATLAB.save(output, data_param)
    else:
        raise Exception('Unknown filetype: {}'.format(data_param['fileType']))


# Merge output for slices reconstructed separately
def mergeOutputSlices(outputList):
    mergedOutput = outputList[0]
    for output in outputList[1:]:
        for seriesType in output:
            mergedOutput[seriesType] = np.concatenate((mergedOutput[seriesType], output[seriesType]))
    return mergedOutput


def getFattyAcidComposition(rho):
    nFAC = len(rho) - 2 # Number of Fatty Acid Composition Parameters
    eps = sys.float_info.epsilon
    CL, UD, PUD = None, None, None

    if nFAC == 1:
        # UD = F2/F1
        UD = np.abs(rho[2] / (rho[1] + eps))
    elif nFAC == 2:
        # UD = F2/F1
        # PUD = F3/F1
        UD = np.abs(rho[2] / (rho[1] + eps))
        PUD = np.abs(rho[3] / (rho[1] + eps))
    elif nFAC == 3:
        # UD = F2/F1
        # PUD = F3/F1
        # CL = F4/F1
        UD = np.abs(rho[2] / (rho[1] + eps))
        PUD = np.abs(rho[3] / (rho[1] + eps))
        CL = np.abs(rho[4] / (rho[1] + eps))
    else:
        raise Exception('Unknown number of Fatty Acid Composition parameters: {}'.format(nFAC))

    return CL, UD, PUD


# Get total fat component (for Fatty Acid Composition; trivial otherwise)
def getFat(rho, alpha):
    fat = np.zeros(rho.shape[1:], dtype=complex)
    for m in range(1, alpha.shape[0]):
        fat += sum(alpha[m, 1:])*rho[m]
    return fat


# Perform fat/water separation and return prescribed output
def reconstruct(data_param, algo_param, model_param):

    # Do the fat/water separation
    rho, B0map, R2map = fatWaterSeparation.reconstruct(data_param, algo_param, model_param)
    wat = rho[0]
    fat = getFat(rho, model_param['alpha'])

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
        if algo_param['magnitudeDiscrimination']:  # to avoid bias from noise
            output['ff'] = 100 * np.real(fat / (wat + fat + sys.float_info.epsilon))
        else:
            output['ff'] = 100 * np.abs(fat)/(np.abs(wat) + np.abs(fat) + sys.float_info.epsilon)
    if 'B0map' in algo_param['output']:
        output['B0map'] = B0map
    if 'R2map' in algo_param['output']:
        output['R2map'] = R2map

    # Do any Fatty Acid Composition in a second pass
    if model_param['nFAC'] > 0:
        rho = fatWaterSeparation.reconstruct(data_param, algo_param['pass2'], model_param['pass2'], B0map, R2map)[0]
        CL, UD, PUD = getFattyAcidComposition(rho)
    
        if 'CL' in algo_param['output']:
            output['CL'] = CL
        if 'UD' in algo_param['output']:
            output['UD'] = UD
        if 'PUD' in algo_param['output']:
            output['PUD'] = PUD

    return output


def main(dataParamFile, algoParamFile, modelParamFile, outDir=None):
    # Read configuration files
    data_param = config.readConfig(dataParamFile)
    algo_param = config.readConfig(algoParamFile)
    model_param = config.readConfig(modelParamFile)

    # Setup configuration objects
    config.setupDataParams(data_param, outDir)
    config.setupModelParams(model_param, data_param['clockwisePrecession'], data_param['temperature'])
    config.setupAlgoParams(algo_param, data_param['nb_echoes'], model_param['nFAC'])

    print(f'B0 = {round(data_param["B0"], 2)}')
    print(f'N echoes = {data_param["nb_echoes"]}')
    print(f't1/dt = {round(data_param["t1"]*1000, 2)}/{round(data_param["dt"]*1000, 2)} msec')
    print(f'nx,ny,nz = {data_param["nx"]},{data_param["ny"]},{data_param["nz"]}')
    print(f'dx,dy,dz = {round(data_param["dx"], 2)},{round(data_param["dy"], 2)},{round(data_param["dz"], 2)}')

    # Run fat/water processing and save output
    if algo_param['use3D'] or len(data_param['sliceList']) == 1:
        if 'slabs' in data_param:
            for iSlab, (slices, z) in enumerate(data_param['slabs']):
                print(f'Processing slab {iSlab+1}/{len(data_param['slabs'])} (slices {slices[0]+1}-{slices[-1]+1})...')
                slab_data_params = config.getSlabDataParams(data_param, slices, z)
                output = reconstruct(slab_data_params, algo_param, model_param)
                save(output, slab_data_params) # save data slab-wise to save memory
        else:
            output = reconstruct(data_param, algo_param, model_param)
            save(output, data_param)
    else:
        output = []
        for z, slice in enumerate(data_param['sliceList']):
            print(f'Processing slice {slice+1} ({z+1}/{len(data_param['sliceList'])})...')
            slice_data_params = config.getSliceDataParams(data_param, slice, z)
            output.append(reconstruct(slice_data_params, algo_param, model_param))
        save(mergeOutputSlices(output), data_param)


if __name__ == '__main__':
    # Initiate command line parser
    p = optparse.OptionParser()
    p.add_option('--dataParamFile', '-d', default='',  type='string', help='File path of data parameter configuration file')
    p.add_option('--algoParamFile', '-a', default='',  type='string', help='File path of algorithm parameter configuration file')
    p.add_option('--modelParamFile', '-m', default='',  type='string', help='File path of model parameter configuration file')

    # Parse command line
    options, arguments = p.parse_args()

    main(options.dataParamFile, options.algoParamFile, options.modelParamFile)