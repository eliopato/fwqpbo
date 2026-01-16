import pydicom
import datetime
import numpy as np

# from https://github.com/rordenlab/dcm2niix/blob/master/Philips/README.md
#  WS = RealWorldValue slope (0040,9225) "PhilipsRWVSlope"
#  WI = RealWorldValue intercept (0040,9224) "PhilipsRWVIntercept"
#  RS = rescale slope (0028,1053) "PhilipsRescaleSlope"
#  RI = rescale intercept (0028,1052) "PhilipsRescaleIntercept"
#  SS = scale slope (2005,100E) "PhilipsScaleSlope"


# List of DICOM attributes required for the water-fat separation
# don't chang the order of the attributes as it would affect the whole processing
req_attributes = ['Image Type', 
                 'Echo Time', 
                 'Slice Location',
                 'Imaging Frequency', 
                 'Columns', 
                 'Rows',
                 'Pixel Spacing',
                 'Slice Thickness']

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
    'Frame sequence': 0x52009230,  # Per-frame Functional Groups Sequence
    'Frame Number': 0x00081160}  # slice number


def get_tag_value(ds: pydicom.Dataset, key: str, frame=None):
    """ Retrieves DICOM element value from dataset ds at tag=key.
    Use frame for multiframe DICOM files"""
    key_id = tag_dict[key]
    if key_id in ds:
        val = ds[key_id].value
        if key == 'Image Type':
            for t in ['M', 'P', 'R', 'I']:
                if t in val:
                    return t
        elif key == 'Echo Time':
            return float(val)
        return val
    
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
            type_list = frame_ds.MRImageFrameTypeSequence[0].FrameType[2]
            for t in ['M', 'P', 'R', 'I']:
                if t in type_list:
                    return t
        
        if key == 'Slice Location':
            return frame_ds.PlanePositionSequence[0].ImagePositionPatient[2]

        if key == 'Imaging Frequency':
            frame_ds = ds.SharedFunctionalGroupsSequence[0]
            return frame_ds.MRImagingModifierSequence[0].TransmitterFrequency

        if key == 'Pixel Spacing':
            return frame_ds.PixelMeasuresSequence[0].PixelSpacing 

        if key == 'Spacing Between Slices':
            return frame_ds.PixelMeasuresSequence[0].SpacingBetweenSlices
        
        if key == 'Slice Thickness':
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


def attr_in_dataset(ds, attr, is_enhanced):
    """Check if attribute is in DICOM dataset ds"""
    if get_tag_value(ds, attr) is not None:
        return True
    elif is_enhanced:
        for frame in range(len(ds[tag_dict['Frame sequence']].value)):
            if not get_tag_value(ds, attr, frame):
                return False  # Attribute must be in all frames!
        return True
    return False


# Check if ds is a multiframe DICOM object
def is_enhanced(ds):
    if tag_dict['Number of frames'] in ds:
        if tag_dict['Frame sequence'] in ds:
            if int(ds[tag_dict['Number of frames']].value) > 1:
                return True
    return False        


def get_valid_files(files):
    """Extract files that are readable and have all required DICOM tags"""

    valid_files = []
    
    for file in files:
    
        try:
            ds = pydicom.dcmread(str(file), stop_before_pixels=True)
        except:
            print(f'Could not read file: {file}')
            continue

        multiframe = is_enhanced(ds)

        has_required_attrs = [attr_in_dataset(ds, attr, multiframe) for attr in req_attributes]
        
        if not all(has_required_attrs):
            print(f'File {file} is missing required DICOM tags:')
            for i, has_attr in enumerate(has_required_attrs):
                if not has_attr:
                    print(req_attributes[i])
            continue
        else:
            valid_files.append(file)
    return valid_files


def get_sop_instance_uid():
    t = datetime.datetime.now()
    datestr = f'{t.year:04d}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}{t.microsecond//1000:03d}'
    randstr = str(np.random.randint(1000, 1000000000))
    uidstr = "1.3.12.2.1107.5.2.32.35356." + datestr + randstr
    return uidstr


def get_series_instance_uid(data_param, series_description):
    if not 'seriesInstanceUIDs' in data_param:
        data_param['seriesInstanceUIDs'] = {}
    if not series_description in data_param['seriesInstanceUIDs']:
        data_param['seriesInstanceUIDs'][series_description] = get_sop_instance_uid() + ".0.0.0"
    return data_param['seriesInstanceUIDs'][series_description]



# Sets DICOM element value in dataset ds at tag=key. Use frame for multiframe
# DICOM files. If missing tag, a new one is created if value representation VR
# is provided
def set_tag_value(ds: pydicom.Dataset, key: str, val, frame=None, VR=None):

    key_id = tag_dict[key]

    # existing tag, update it
    if get_tag_value(ds, key, frame) is not None:

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
            #     frame_ds.PlanePositionSequence[0].ImagePositionPatient[2] = val
            # elif key == 'Imaging Frequency':
            #     frame_ds = ds.SharedFunctionalGroupsSequence[0]
            #     frame_ds.MRImagingModifierSequence[0].TransmitterFrequency = val 
            # elif key == 'Pixel Spacing':
            #     frame_ds.PixelMeasuresSequence[0].PixelSpacing  = val
            # elif key == 'Spacing Between Slices':
            #     frame_ds.PixelMeasuresSequence[0].SliceThickness = val
            # elif key == 'Rescale Intercept':
            #     frame_ds.PixelValueTransformationSequence[0].RescaleIntercept = val
            # elif key == 'Rescale Slope':
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


# group slices in slice_list in slabs of slabs_size contiguous slices
def get_slabs(slice_list: list[int], slabs_size: int):
    slabs = []
    slices = []
    pos = 0
    for z, slice in enumerate(slice_list):
        # start a new slab
        if slices and (len(slices) == slabs_size or not slice == slices[-1]+1):
            slabs.append((slices, pos))
            slices = [slice]
            pos = z
        else:
            slices.append(slice)
    slabs.append((slices, pos))
    return slabs
