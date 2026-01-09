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
def reconstruct(data_param, aPar, mPar):

    # Do the fat/water separation
    rho, B0map, R2map = fatWaterSeparation.reconstruct(data_param, aPar, mPar)
    wat = rho[0]
    fat = getFat(rho, mPar['alpha'])

    # Prepare prescribed output
    output = {}
    if 'wat' in aPar['output']:
        output['wat'] = np.abs(wat)
    if 'fat' in aPar['output']:
        output['fat'] = np.abs(fat)
    if 'phi' in aPar['output']:
        output['phi'] = np.angle(wat, deg=True) + 180
    if 'ip' in aPar['output']: # Calculate synthetic in-phase
        output['ip'] = np.abs(wat+fat)
    if 'op' in aPar['output']: # Calculate synthetic opposed-phase
        output['op'] = np.abs(wat-fat)
    if 'ff' in aPar['output']: # Calculate the fat fraction
        if aPar['magnitudeDiscrimination']:  # to avoid bias from noise
            output['ff'] = 100 * np.real(fat / (wat + fat + sys.float_info.epsilon))
        else:
            output['ff'] = 100 * np.abs(fat)/(np.abs(wat) + np.abs(fat) + sys.float_info.epsilon)
    if 'B0map' in aPar['output']:
        output['B0map'] = B0map
    if 'R2map' in aPar['output']:
        output['R2map'] = R2map

    # Do any Fatty Acid Composition in a second pass
    if mPar['nFAC'] > 0:
        rho = fatWaterSeparation.reconstruct(data_param, aPar['pass2'], mPar['pass2'], B0map, R2map)[0]
        CL, UD, PUD = getFattyAcidComposition(rho)
    
        if 'CL' in aPar['output']:
            output['CL'] = CL
        if 'UD' in aPar['output']:
            output['UD'] = UD
        if 'PUD' in aPar['output']:
            output['PUD'] = PUD

    return output


def main(dataParamFile, algoParamFile, modelParamFile, outDir=None):
    # Read configuration files
    data_param = config.readConfig(dataParamFile)
    aPar = config.readConfig(algoParamFile)
    mPar = config.readConfig(modelParamFile)

    # Setup configuration objects
    config.setupDataParams(data_param, outDir)
    config.setupModelParams(mPar, data_param['clockwisePrecession'], data_param['temperature'])
    config.setupAlgoParams(aPar, data_param['nb_echoes'], mPar['nFAC'])

    print(f'B0 = {round(data_param["B0"], 2)}')
    print(f'N = {data_param["nb_echoes"]}')
    print(f't1/dt = {round(data_param["t1"]*1000, 2)}/{round(data_param["dt"]*1000, 2)} msec')
    print(f'nx,ny,nz = {data_param["nx"]},{data_param["ny"]},{data_param["nz"]}')
    print(f'dx,dy,dz = {round(data_param["dx"], 2)},{round(data_param["dy"], 2)},{round(data_param["dz"], 2)}')

    # Run fat/water processing and save output
    if aPar['use3D'] or len(data_param['sliceList']) == 1:
        if 'slabs' in data_param:
            for iSlab, (slices, z) in enumerate(data_param['slabs']):
                print(f'Processing slab {iSlab+1}/{len(data_param['slabs'])} (slices {slices[0]+1}-{slices[-1]+1})...')
                slabDataParams = config.getSlabDataParams(data_param, slices, z)
                output = reconstruct(slabDataParams, aPar, mPar)
                save(output, slabDataParams) # save data slab-wise to save memory
        else:
            output = reconstruct(data_param, aPar, mPar)
            save(output, data_param)
    else:
        output = []
        for z, slice in enumerate(data_param['sliceList']):
            print('Processing slice {} ({}/{})...'
                  .format(slice+1, z+1, len(data_param['sliceList'])))
            sliceDataParams = config.getSliceDataParams(data_param, slice, z)
            output.append(reconstruct(sliceDataParams, aPar, mPar))
        save(mergeOutputSlices(output), data_param)


if __name__ == '__main__':
    # Initiate command line parser
    p = optparse.OptionParser()
    p.add_option('--dataParamFile', '-d', default='',  type="string",
                 help="File path of data parameter configuration file")
    p.add_option('--algoParamFile', '-a', default='',  type="string",
                 help="File path of algorithm parameter configuration file")
    p.add_option('--modelParamFile', '-m', default='',  type="string",
                 help="File path of model parameter configuration file")

    # Parse command line
    options, arguments = p.parse_args()

    main(options.dataParamFile, options.algoParamFile, options.modelParamFile)