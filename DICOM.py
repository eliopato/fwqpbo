import pydicom
import datetime
import numpy as np


gyro = 42.58  # 1H gyromagnetic ratio

# from https://github.com/rordenlab/dcm2niix/blob/master/Philips/README.md
#  WS = RealWorldValue slope (0040,9225) "PhilipsRWVSlope"
#  WI = RealWorldValue intercept (0040,9224) "PhilipsRWVIntercept"
#  RS = rescale slope (0028,1053) "PhilipsRescaleSlope"
#  RI = rescale intercept (0028,1052) "PhilipsRescaleIntercept"
#  SS = scale slope (2005,100E) "PhilipsScaleSlope"


# Dictionary of DICOM tags
tag_dict = {
    'Image Type': 0x00080008,
    'SOP Class UID': 0x00080016,
    'SOP Instance UID': 0x00080018,
    'Series Description': 0x0008103E,
    'Slice Thickness': 0x00180050,
    'Spacing Between Slices': 0x00180088,
    'Echo Time': 0x00180081,
    'Imaging Frequency': 0x00180084,
    'Protocol Name': 0x00181030,
    'Study Instance UID': 0x0020000D,
    'Series Instance UID': 0x0020000E,
    'Series Number': 0x00200011,
    'Slice Location': 0x00201041,
    'Image Position Patient': 0x00200032,
    'Rows': 0x00280010,
    'Columns': 0x00280011,
    'Pixel Spacing': 0x00280030,
    'Pixel Aspect Ratio': 0x00280034,
    'Smallest Pixel Value': 0x00280106,
    'Largest Pixel Value': 0x00280107,
    'Window Center': 0x00281050,
    'Window Width': 0x00281051,
    'Rescale Intercept': 0x00281052,
    'Rescale Slope': 0x00281053,
    'Number of frames': 0x00280008,
    'Frame sequence': 0x52009230}  # Per-frame Functional Groups Sequence

# DICOM tags for output from fat/water separation
imgTypes = {
    'phi': {'descr': 'Initial phase (degrees)', 'seriesNumber': 100},
    'wat': {'descr': 'Water-only', 'seriesNumber': 101},
    'fat': {'descr': 'Fat-only', 'seriesNumber': 102},
    'ip': {'descr': 'In-phase', 'seriesNumber': 103},
    'op': {'descr': 'Opposed-phase', 'seriesNumber': 104},    
    'ff': {'descr': 'Fat Fraction (%)', 'seriesNumber': 105, 'RescaleIntercept': -100, 'RescaleSlope': 0.1},
    'R2map': {'descr': 'R2* (msec-1)', 'seriesNumber': 106},
    'B0map': {'descr': 'B0 inhomogeneity (ppm)', 'seriesNumber': 107, 'RescaleSlope': 0.001},
    'CL': {'descr': 'FAC Chain length', 'seriesNumber': 108, 'RescaleSlope': 0.01},
    'UD': {'descr': 'FAC Unsaturation degree', 'seriesNumber': 109, 'RescaleSlope': 0.01},
    'PUD': {'descr': 'FAC Polyunsaturation degree', 'seriesNumber': 110, 'RescaleSlope': 0.01}
}


# List of DICOM attributes required for the water-fat separation
# don't chang the order of the attributes as it would affect the whole processing
reqAttributes = ['Image Type', 
                 'Echo Time', 
                 'Slice Location',
                 'Imaging Frequency', 
                 'Columns', 
                 'Rows',
                 'Pixel Spacing',
                 'Spacing Between Slices']


# Translates series description tag to M/P/R/I for magn/phase/real/imaginary
def seriesDescription2type(series_description):
    for type, description in [('R', 'Real Image'), ('I', 'Imag Image')]:
        if description in series_description:
            return type
    return None


# Translates image type tag to M/P/R/I for magnitude/phase/real/imaginary
def typeTag2type(tagValue):
    for type in ['M', 'P', 'R', 'I']:
        if type in tagValue:
            return type
    return None


# Retrieves DICOM element value from dataset ds at tag=key.
# Use frame for multiframe DICOM files
def getTagValue(ds: pydicom.Dataset, key: str, frame=None):
    
    key_id = tag_dict[key]
    if key_id in ds:
        return ds[key_id].value
    
    # multiframe images (enhanced dicom)
    if frame is not None:

        frame_ds = ds[tag_dict['Frame sequence']].value[frame]

        # Philips(?) private tag containing frame tags
        # 0x2005140f = Image Patient Position
        if 0x2005140f in frame_ds:
            frame_ds = frame_ds[0x2005140f][0]
        
        if key_id in frame_ds:
            return key_id.value
    
        if key == 'Echo Time':
            return frame_ds.MREchoSequence[0].EffectiveEchoTime
        
        if key == 'Image Type':
            return frame_ds.MRImageFrameTypeSequence[0].FrameType[2]
        
        if key == 'Slice Location':
            return frame_ds.PlanePositionSequence[0].ImagePositionPatient[2]

        if key == 'Imaging Frequency':
            frame_ds = ds.SharedFunctionalGroupsSequence[0]
            return frame_ds.MRImagingModifierSequence[0].TransmitterFrequency ## semble être en str() en comparaison mais bloque plus tard

        if key == 'Pixel Spacing':
            return frame_ds.PixelMeasuresSequence[0].PixelSpacing 

        if key == 'Spacing Between Slices':
            return frame_ds.PixelMeasuresSequence[0].SliceThickness
        
        if key == 'Rescale Intercept':
            return frame_ds.PixelValueTransformationSequence[0].RescaleIntercept

        if key == 'Rescale Slope':
            return frame_ds.PixelValueTransformationSequence[0].RescaleSlope
        
        if key == 'Window Center':
            return frame_ds.FrameVOILUTSequence[0].WindowCenter
        
        if key == 'Window Width':
            return frame_ds.FrameVOILUTSequence[0].WindowWidth
            
    return None


# Retrieve attribute from dataset ds. Use frame for multiframe DICOM files.
# Must attributes are read directly from their corresponding DICOM tag
def getAttribute(ds: pydicom.Dataset, attr: str, frame=None):

    value = getTagValue(ds, attr, frame)

    if value is None:

        if attr == 'Slice Location':

            value = getTagValue(ds, 'Image Position Patient', frame)
            if value:
                value = value[2]

        elif attr == 'Spacing Between Slices':
            value = getTagValue(ds, 'Slice Thickness', frame)

    elif attr == 'Image Type':  # special handling of image type
        value = typeTag2type(value)
        if value is None:
            value = seriesDescription2type(getTagValue(ds, 'Series Description', frame))
            
    return value


# Check if attribute is in DICOM dataset ds
def attrInDataset(ds, attr, multiframe):
    if getAttribute(ds, attr) is not None:
        return True
    elif multiframe:
        for frame in range(len(ds[tag_dict['Frame sequence']].value)):
            if not getAttribute(ds, attr, frame):
                return False  # Attribute must be in all frames!
        return True
    return False


# Check if ds is a multiframe DICOM object
def isMultiFrame(ds):
    if tag_dict['Number of frames'] in ds:
        if tag_dict['Frame sequence'] in ds:
            if int(ds[tag_dict['Number of frames']].value) > 1:
                return True
    return False        


# Extract files that are readable and have all required DICOM tags
def getValidFiles(files):
    validFiles = []
    for file in files:
        try:
            ds = pydicom.read_file(str(file), stop_before_pixels=True)
        except:
            print(f'Could not read file: {file}')
            continue

        multiframe = isMultiFrame(ds)

        hasRequiredAttrs = [attrInDataset(ds, attr, multiframe) for attr in reqAttributes]
        if not all(hasRequiredAttrs):
            print(f'File {file} is missing required DICOM tags:')
            for i, hasAttr in enumerate(hasRequiredAttrs):
                if not hasAttr:
                    print(reqAttributes[i])
            continue
        else:
            validFiles.append(file)
    return validFiles


# get combination of image types for DICOM frames in frame_list
def getType(frame_list, printType=False):

    typeTags = [tags[2] for tags in frame_list]
    
    numR = typeTags.count('R')
    numI = typeTags.count('I')
    numM = typeTags.count('M')
    numP = typeTags.count('P')

    if numM + numP == 0 and numR + numI > 0 and numR == numI:
        if printType:
            print('Real/Imaginary images')
        return 'RI'
    
    elif numM + numP > 0 and numR + numI == 0 and numM == numP:
        if printType:
            print('Magnitude/Phase images')
        return 'MP'
    
    elif numP == 0 and numM + numR + numI > 0 and numM == numR == numI:
        if printType:
            print('Magnitude/Real/Imaginary images')
        return 'MRI'
    
    elif numM + numP + numR + numI > 0 and numM == numP == numR == numI:
        if printType:
            print('Magnitude/Phase/Real/Imaginary images')
        return 'MPRI'
    
    else:
        raise Exception(f'Unknown combination of image types: {numR} real, {numI} imag, {numM} magn, {numP} phase')


def getSOPInstanceUID():
    t = datetime.datetime.now()
    datestr = f'{t.year:04d}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}{t.microsecond//1000:03d}'
    randstr = str(np.random.randint(1000, 1000000000))
    uidstr = "1.3.12.2.1107.5.2.32.35356." + datestr + randstr
    return uidstr


def getSeriesInstanceUID(data_param, series_description):
    if not 'seriesInstanceUIDs' in data_param:
        data_param['seriesInstanceUIDs'] = {}
    if not series_description in data_param['seriesInstanceUIDs']:
        data_param['seriesInstanceUIDs'][series_description] = getSOPInstanceUID() + ".0.0.0"
    return data_param['seriesInstanceUIDs'][series_description]


# update data_param with info retrieved from the DICOM files including image data
def updateDataParams(data_param, files):
    data_param['fileType'] = 'DICOM'
    frame_list = []

    # read file headers to get meta data
    for file in files:
        ds = pydicom.read_file(str(file), stop_before_pixels=True)
        multiframe = isMultiFrame(ds)
        # enhanced
        if multiframe:
            for frame in range(len(ds[tag_dict['Frame sequence']].value)):
                attr_list = [getAttribute(ds, attr, frame) for attr in reqAttributes]
                frame_list.append([file] + [frame] + attr_list)
        # standard
        else: 
            frame_list.append([file]+[None]+[getAttribute(ds, attr) for attr in reqAttributes])
    
    frame_list.sort(key=lambda tags: tags[2])  # First, sort on type (M/P/R/I)
    frame_list.sort(key=lambda tags: tags[3])  # Second, sort on echo time
    frame_list.sort(key=lambda tags: tags[4])  # Third, sort on slice location

    type = getType(frame_list, True)
    data_param['dx'] = float(frame_list[0][8][1])
    data_param['dy'] = float(frame_list[0][8][0])
    data_param['dz'] = float(frame_list[0][9])
    data_param['B0'] = frame_list[0][5]/gyro

    # [msec]->[sec]
    echo_times = sorted(set([float(tags[3])/1000. for tags in frame_list]))
    data_param['nb_echoes'] = len(echo_times)

    # echoes not specified in parameter file, select all of them
    if 'echoes' not in data_param:
        data_param['echoes'] = range(data_param['nb_echoes'])
    # otherwise select the ones that are listed in parameter 'echoes'
    else:
        echo_times = [echo_times[echo] for echo in data_param['echoes']] 
        data_param['nb_echoes'] = len(echo_times)

    if data_param['nb_echoes'] < 2:
        raise Exception(f'At least 2 echoes required, only {data_param["nb_echoes"]} given')
    
    data_param['t1'] = echo_times[0]
    data_param['dt'] = np.mean(np.diff(echo_times))
    if not 0.95 < np.max(np.diff(echo_times))/data_param['dt'] < 1.05:
        print('Warning: echo inter-spacing varies more than 5%')
        print(echo_times)

    # take all the slices if nothing is specified in the params
    if 'sliceList' not in data_param:
        n_slices = len(set([tags[4] for tags in frame_list]))
        data_param['sliceList'] = [n for n in range(n_slices)]
    else:
        n_slices = len(data_param['sliceList'])

    data_param['nx'] = frame_list[0][6]
    data_param['ny'] = frame_list[0][7]
    data_param['nz'] = len(data_param['sliceList'])

    if 'cropFOV' in data_param:
        x1, x2 = data_param['cropFOV'][0], data_param['cropFOV'][1]
        y1, y2 = data_param['cropFOV'][2], data_param['cropFOV'][3]
        data_param['Nx'], data_param['nx'] = data_param['nx'], x2-x1
        data_param['Ny'], data_param['ny'] = data_param['ny'], y2-y1
    else:
        x1, x2 = 0, data_param['nx']
        y1, y2 = 0, data_param['ny']

    img = []

    if multiframe:
        file = frame_list[0][0]
        dcm = pydicom.read_file(str(file))
    
    for n in data_param['echoes']:

        for slice in data_param['sliceList']:

            i = (data_param['nb_echoes'] * slice + n) * len(type)
            
            if type == 'MP':  # Magnitude/phase images
                
                i_magn_frame = i
                magn_file = frame_list[i_magn_frame][0]
                magn_img = pydicom.read_file(str(magn_file)).pixel_array

                i_phase_frame = i + 1
                phase_file = frame_list[i_phase_frame][0]
                phase_ds = pydicom.read_file(str(phase_file))
                phase_img = phase_ds.pixel_array
                
                # read the current slice, depending on the array shape (enhanced images have the slice as the 1st dimension)
                if len(phase_img.shape) == 3:
                    magn = magn_img[slice, y1:y2, x1:x2].flatten()
                    phase = phase_img[slice, y1:y2, x1:x2].flatten()
                elif len(phase_img.shape) == 2:
                    magn = magn_img[y1:y2, x1:x2].flatten()
                    phase = phase_img[y1:y2, x1:x2].flatten()
                else:
                    raise Exception(f'the image shape should have 2 or 3 dimensions, not {len(phase_img.shape)}')
                
                # get rescale intercept
                rescale_intercept = np.abs(getAttribute(dcm, 'Rescale Intercept', frame_list[i_phase_frame][1]))
                if rescale_intercept is None or rescale_intercept == 0:
                    print('No Rescale Intercept DICOM tag found. Using 4096.')
                    rescale_intercept = 4096
                # Abs val needed for Siemens data to get correct phase sign
                rescale_intercept = np.abs(rescale_intercept)
                
                # For some reason, intercept is used as slope (Siemens only?)
                c = magn * np.exp(phase/float(rescale_intercept) * 2 * np.pi * 1j)
            
            # Real/imaginary images and Magnitude/real/imaginary images
            elif type in ['RI', 'MRI', 'MPRI']:
                
                i_real_frame = i + type.index('R') + 1
                real_file = frame_list[i_real_frame][0]
                real_ds = pydicom.read_file(str(real_file))
                real_img = real_ds.pixel_array

                i_imag_frame = i
                imag_file = frame_list[i_imag_frame][0]
                imag_img = pydicom.read_file(str(imag_file)).pixel_array
                
                dcm_frame = frame_list[i_real_frame][1]

                # read the current slice, depending on the array shape (enhanced images have the slice as the 1st dimension)
                if len(phase_img.shape) == 3:
                    real_part = real_img[slice, y1:y2, x1:x2].flatten()
                    imag_part = imag_img[slice, y1:y2, x1:x2].flatten()
                else:
                    real_part = real_img[y1:y2, x1:x2].flatten()
                    imag_part = imag_img[y1:y2, x1:x2].flatten()

                # Assumes real and imaginary slope/intercept are equal
                rescale_intercept = getAttribute(dcm, 'Rescale Intercept', dcm_frame)
                rescale_slope = getAttribute(dcm, 'Rescale Slope', dcm_frame)
                if rescale_intercept and rescale_slope:
                    offset = rescale_intercept/rescale_slope
                else:
                    offset = -2047.5

                c = (real_part + offset) + 1.0 * 1j * (imag_part + offset)

            else:
                raise Exception('Unknown image types')
            
            img.append(c)

    data_param['frame_list'] = frame_list
    img = np.array(img) * data_param['reScale']
    # todo check that the dims are in the right order for enhanced image 
    img = np.reshape(img, shape=(data_param['nb_echoes'], data_param['nz'], data_param['ny'], data_param['nx']))
    data_param['img'] = img


# Set window so that percentile % of pixels are inside
def getPercentileWindow(im, intercept, slope, percentile=95):
    lims = np.percentile(im, [(100-percentile)/2, percentile + (100-percentile)/2])
    width = lims[1]-lims[0]
    center = width/2.+lims[0]
    return center*slope+intercept, width*slope


# Sets DICOM element value in dataset ds at tag=key. Use frame for multiframe
# DICOM files. If missing tag, a new one is created if value representation VR
# is provided
def setTagValue(ds: pydicom.Dataset, key: str, val, frame=None, VR=None):

    key_id = tag_dict[key]

    # existing tag, update it
    if getTagValue(ds, key, frame) is not None:

        if key_id in ds:
            ds[key_id].value = val
            return True
        elif frame is not None:
            if key_id in ds:
                ds[key_id].value = val
            else:
                frame_ds = ds[tag_dict['Frame sequence']].value[frame]
                
                # Philips(?) private tag containing frame tags
                if 0x2005140f in frame_ds:   
                    frame_ds = frame_ds[0x2005140f][0]

                if key_id in frame_ds:
                    frame_ds[key_id].value = val
                elif key == 'Echo Time':
                    frame_ds.MREchoSequence[0].EffectiveEchoTime = val
                elif key == 'Image Type':
                    frame_ds.MRImageFrameTypeSequence[0].FrameType[2] = val
                elif key == 'Slice Location':
                    frame_ds.PlanePositionSequence[0].ImagePositionPatient[2] = val
                elif key == 'Imaging Frequency':
                    frame_ds = ds.SharedFunctionalGroupsSequence[0]
                    frame_ds.MRImagingModifierSequence[0].TransmitterFrequency = val 
                elif key == 'Pixel Spacing':
                    frame_ds.PixelMeasuresSequence[0].PixelSpacing  = val
                elif key == 'Spacing Between Slices':
                    frame_ds.PixelMeasuresSequence[0].SliceThickness = val
                elif key == 'Rescale Intercept':
                    frame_ds.PixelValueTransformationSequence[0].RescaleIntercept = val
                elif key == 'Rescale Slope':
                    frame_ds.PixelValueTransformationSequence[0].RescaleSlope = val
                elif key == 'Window Center':
                    frame_ds.FrameVOILUTSequence[0].WindowCenter = val
                elif key == 'Window Width':
                    frame_ds.FrameVOILUTSequence[0].WindowWidth = val
                else:
                    print(f'WARNING: Tag {key} with id {key_id} has no set method, please define it')
                    return False
            return True
        else:
            return False
                
    # Else, add as new DICOM element:
    if VR:
        
        first_level_keys = ['SOP Instance UID', 'Series Instance UID', 'Protocol Name',
                            'Series Description', 'Smallest Pixel Value', 'Largest Pixel Value']
        if frame is None or key in first_level_keys:
            ds.add_new(key_id, val, VR)
            return True
        else:
            frame_ds = ds[tag_dict['Frame sequence']].value[frame]
            
            # # Philips(?) private tag containing frame tags
            # if 0x2005140f in frame_ds:   
            #     frame_ds = frame_ds[0x2005140f][0]

            # if key == 'Echo Time':
            #     frame_ds = frame_ds.MREchoSequence[0]   
            #     frame_ds.add_new('EffectiveEchoTime', val, VR)
            # elif key == 'Image Type':
            #     frame_ds = frame_ds.MRImageFrameTypeSequence[0]
            #     frame_ds.add_new('FrameType', val, VR)
            # elif key == 'Slice Location':
            #     print(key, len(frame_ds.MREchoSequence))
            #     frame_ds.PlanePositionSequence[0].ImagePositionPatient[2] = val
            # elif key == 'Imaging Frequency':
            #     print(key, len(frame_ds.MREchoSequence))
            #     frame_ds = ds.SharedFunctionalGroupsSequence[0]
            #     frame_ds.MRImagingModifierSequence[0].TransmitterFrequency = val 
            # elif key == 'Pixel Spacing':
            #     print(key, len(frame_ds.MREchoSequence))
            #     frame_ds.PixelMeasuresSequence[0].PixelSpacing  = val
            # elif key == 'Spacing Between Slices':
            #     print(key, len(frame_ds.MREchoSequence))
            #     frame_ds.PixelMeasuresSequence[0].SliceThickness = val
            # elif key == 'Rescale Intercept':
            #     print(key, len(frame_ds.MREchoSequence))
            #     frame_ds.PixelValueTransformationSequence[0].RescaleIntercept = val
            # elif key == 'Rescale Slope':
            #     print(key, len(frame_ds.MREchoSequence))
            #     frame_ds.PixelValueTransformationSequence[0].RescaleSlope = val
            # elif key == 'Window Center':
            #     frame_ds.FrameVOILUTSequence[0].WindowCenter = val
            # elif key == 'Window Width':
            #     frame_ds.FrameVOILUTSequence[0].WindowWidth = val
            # else:
            #     return False
            # return True
        
    print(f'Warning: DICOM tag {key} was not set')
    return False


# Save numpy array to DICOM image.
# Based on input DICOM image if exists, else create from scratch
def saveSeries(out_dir: str, img_type:str, img: np.array, data_param: dict):
    print('> Saving DICOM series')
    print(f'Image type: {img_type}')
    print(f'Output directory: {out_dir}')

    series_description = imgTypes[img_type]['descr']
    series_number = imgTypes[img_type]['seriesNumber']
    if 'Rescale Intercept' in imgTypes[img_type]:
        rescale_intercept = imgTypes[img_type]['Rescale Intercept']
    else:
        rescale_intercept = 0.
    if 'Rescale Slope' in imgTypes[img_type]:
        rescale_slope = imgTypes[img_type]['Rescale Slope']
    else:
        rescale_slope = 1.
    
    seriesInstanceUID = getSeriesInstanceUID(data_param, series_description)
    # Single file is interpreted as multi-frame
    multiframe = data_param['frame_list'] and len(set([frame[0] for frame in data_param['frame_list']])) == 1
    if multiframe:
        ds = pydicom.read_file(str(data_param['frame_list'][0][0]))
        imVol = np.empty([data_param['nz'], data_param['ny']*data_param['nx']], dtype='uint16')
        frames = []
    if data_param['frame_list']:
        DICOMimgType = getType(data_param['frame_list'])
        
    for z, slice in enumerate(data_param['sliceList']):
        
        filename = out_dir / './{}.dcm'.format(slice)
        # Extract slice, scale and type cast pixel data
        pixelData = np.array([max(0, (val-rescale_intercept)/rescale_slope) for val in img[:, :, z].flatten()])
        pixelData = pixelData.astype('uint16')
        # Set window so that 95% of pixels are inside
        windowCenter, windowWidth = getPercentileWindow(pixelData, rescale_intercept, rescale_slope, 95)
        if data_param['frame_list']:
            # Get frame
            frame = data_param['frame_list'][data_param['nb_echoes']*slice*len(DICOMimgType)]
            n_frame = frame[1]
            if not multiframe:
                ds = pydicom.read_file(str(frame[0]))
        else:
            n_frame = None
            file_meta = pydicom.dataset.Dataset()
            file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
            file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.' + '1.1.111165684411017669021768385720736873780'
            file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.' + '1.0.100.4.0'
            file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            ds = pydicom.dataset.FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0"*128)
            # Add DICOM tags:
            ds.Modality = 'WSD'
            ds.ContentDate = str(datetime.date.today()).replace('-', '')
            ds.ContentTime = str(datetime.time())  # millisecs since the epoch
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 0
            ds.HighBit = 15
            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.SmallestImagePixelValue = '\\x00\\x00'
            ds.LargestImagePixelValue = '\\xff\\xff'
            ds.Columns = img.shape[1]
            ds.Rows = img.shape[0]
            setTagValue(ds, 'Study Instance UID', getSOPInstanceUID(), n_frame, 'UI')
            setTagValue(ds, 'Pixel Spacing', str([data_param['dx'], data_param['dy']]), n_frame, 'DS')
            setTagValue(ds, 'Pixel Aspect Ratio', str([int(data_param['dx']*100), int(data_param['dy']*100)]), n_frame, 'IS')

        # Change/add DICOM tags:
        setTagValue(ds, 'SOP Instance UID', getSOPInstanceUID(), n_frame, 'UI') 
        setTagValue(ds, 'Series Instance UID', seriesInstanceUID, n_frame, 'UI')
        setTagValue(ds, 'Protocol Name', 'Derived Image', n_frame, 'LO')
        setTagValue(ds, 'Series Description', series_description, n_frame, 'LO')
        setTagValue(ds, 'Smallest Pixel Value', np.min(pixelData), n_frame)
        setTagValue(ds, 'Largest Pixel Value', np.max(pixelData), n_frame)
        setTagValue(ds, 'Window Center', str(windowCenter), n_frame, 'DS')
        setTagValue(ds, 'Window Width', str(windowWidth), n_frame, 'DS')
        setTagValue(ds, 'Rescale Intercept', str(rescale_intercept), n_frame, 'DS')
        setTagValue(ds, 'Rescale Slope', str(rescale_slope), n_frame, 'DS')
        setTagValue(ds, 'Series Number', str(series_number), n_frame, 'IS')
        if isMultiFrame:
            setTagValue(ds, 'Echo Time', 0., n_frame, 'FD')
        else:
            setTagValue(ds, 'Echo Time', "0", n_frame, 'DS')

        if multiframe:
            imVol[z] = pixelData
            frames.append(n_frame)
        else:
            ds.PixelData = pixelData.tobytes()
            ds.save_as(str(filename))

    if multiframe:
        setTagValue(ds, 'SOP Instance UID', getSOPInstanceUID())
        setTagValue(ds, 'Number of Frames', len(frames))
        val = [ds[tag_dict['Frame Sequence']][frame] for frame in frames]
        ds[tag_dict['Frame sequence']].value = val
        ds.PixelData = imVol.tobytes()
        filename = out_dir / './0.dcm'
        ds.save_as(str(filename))


# Save all data in output as DICOM images
def save(output, data_param):
    for seriesType in output:
        out_dir = data_param['out_dir'] / seriesType
        out_dir.mkdir(parents=True, exist_ok=True)
        print(r'Writing image{} to "{}"'.format('s'*(data_param['nz'] > 1), out_dir))
        saveSeries(out_dir, seriesType, output[seriesType], data_param)
