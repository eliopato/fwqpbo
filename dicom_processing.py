import pydicom
import datetime
import numpy as np
import dicom_tools

gyro = 42.58  # 1H gyromagnetic ratio

# DICOM tags for output from fat/water separation
img_types_dict = {
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


class FrameCollection():

    def __init__(self, is_enhanced: bool):
        self.frame_list = []
        self.is_enhanced = is_enhanced
        self.dx = None
        self.dy = None
        self.dz = None
        self.nx = None
        self.ny = None
        self.Nx = None
        self.Ny = None
        self.nz = None
        self.b0 = None
        self.t1 = None
        self.dt = None
        self.echo_times = None

    @property
    def frame_indexes(self) -> list:
        return list(set([s.frame_index for s in self.frame_list]))
    
    @property
    def n_frames_indexes(self) -> int:
        return len(self.frame_indexes)
    
    @property
    def n_frames(self) -> int:
        return len(self.frame_list)
        
    @property
    def n_echo(self) -> int:
        return 0 if self.echo_times is None else len(self.echo_times)
    
    def add_frame(self, frame: Frame) -> None:
        self.frame_list.append(frame)

    def get_frames(self, frame_idx: int) -> list[Frame]:
        return [s for s in self.frame_list if s.frame_idx == frame_idx] 
    
    def __getitem__(self, index: int) -> Frame:
        return self.frame_list[index]
    
    def get_all_attr_values(self, attr_name: str) -> list:
        return [s.get_attr(attr_name) for s in self.frame_list]

    def select_slices(self, selected_indexes:list[int]) -> None:
        """Drop slices that are not in the provided list"""
        self.frame_list = [s for s in self.frame_list if s.frame_index in selected_indexes]

    def get_image_types(self, print_type=False) -> str:
        """Get combination of image types for DICOM frames in frame_list. 
        Check that the number of frame per identified type is equal.
        Returns 'RI', 'MP', 'MRI', or 'MPRI'"""
        type_tags = self.get_all_attr_values('Image Type')
        
        num_tags = dict()
        for image_type in ['R', 'I', 'M', 'P']:
            num_tags[image_type] = type_tags.count(image_type)

        if num_tags['M'] == 0 and num_tags['P'] == 0:
            if num_tags['R'] > 0 and num_tags['R'] == num_tags['I']:
                if print_type:
                    print('Real/Imaginary images')
                return 'RI'
        else:    
            if num_tags['R'] == 0 and num_tags['I'] == 0 and num_tags['M'] == num_tags['P']:
                if print_type:
                    print('Magnitude/Phase images')
                return 'MP'
            elif num_tags['M'] > 0 and num_tags['M'] == num_tags['R'] == num_tags['I']:
                if num_tags['P'] == 0:
                    if print_type:
                        print('Magnitude/Real/Imaginary images')
                    return 'MRI'
                elif num_tags['M'] == num_tags['P']:
                    if print_type:
                        print('Magnitude/Phase/Real/Imaginary images')
                    return 'MPRI'
        
        raise Exception(f'Unknown combination of image types: {num_tags}')

class Frame():

    def __init__(self, file_path: str, frame_idx: int):
        self.frame_idx = frame_idx
        self.path = file_path
    
    def add_attr(self, attr_name, attr_value):
        self.__setattr__(attr_name, attr_value)

    def get_attr(self, attr_name):
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        else:
            return None


def seriesDescription2type(series_description):
    """Translates series description tag to M/P/R/I for magn/phase/real/imaginary"""
    for type, description in [('R', 'Real Image'), ('I', 'Imag Image')]:
        if description in series_description:
            return type
    return None


def typeTag2type(tagValue):
    """Translates image type tag to M/P/R/I for magnitude/phase/real/imaginary"""
    for type in ['M', 'P', 'R', 'I']:
        if type in tagValue:
            return type
    return None


# update data_param with info retrieved from the DICOM files including image data
def update_data_params(data_param: dict, files: list):

    # define if multiframe
    ds = pydicom.read_file(str(files[0]), stop_before_pixels=True)
    frame_list = FrameCollection(dicom_tools.is_enhanced(ds))

    # read file headers to get meta data
    for file in files:
        ds = pydicom.read_file(str(file), stop_before_pixels=True)
        if frame_list.is_enhanced:
            frames = [(f, f) for f in range(len(dicom_tools.getAttribute(ds, 'Frame sequence')))]
        else: 
            frames = [(None, dicom_tools.get_tag_value(ds, 'Frame Number'))]
        for n_frame, n_slice in frames:
            new_slice = Frame(str(file), n_slice)
            for attr in dicom_tools.req_attributes:
                new_slice.add_attr(attr, dicom_tools.get_tag_value(ds, attr, n_frame))
            frame_list.add_slice(new_slice)
    
    # frame_list.sort(key=lambda tags: tags[2])  # First, sort on type (M/P/R/I)
    # frame_list.sort(key=lambda tags: tags[3])  # Second, sort on echo time
    # frame_list.sort(key=lambda tags: tags[4])  # Third, sort on slice location

    type = frame_list.get_image_types(True)
    first_slice = frame_list[0]
    frame_list.dx = float(first_slice.get_attr('Pixel Spacing')[1])
    frame_list.dy = float(first_slice.get_attr('Pixel Spacing')[0])
    frame_list.dz = float(first_slice.get_attr('Spacing Between Slices'))
    frame_list.b0 = first_slice.get_attr('Imaging Frequency')/gyro

    # get the sorted list of echo times in seconds
    frame_list.echo_times = sorted(set([et/1000. for et in frame_list.get_all_attr_values('Echo Time')]))
    
    # select specified echoes from parameter file
    if 'echoes' in data_param:
        selected_echoes = []
        for et in data_param['echoes']:
            if et not in frame_list.echo_times:
                print(f'Echo time selected in parameter file {et} not found in images')
            else:
                selected_echoes.append(et)
        frame_list.echo_times = selected_echoes
    
    # check the number of echoes
    if frame_list.nb_echoes < 2:
        raise Exception(f'At least 2 echoes required, only {frame_list.nb_echoes} found')
    
    frame_list.t1 = frame_list.echo_times[0]
    frame_list.dt = np.mean(np.diff(frame_list.echo_times))
    if not 0.95 < np.max(np.diff(frame_list.echo_times))/frame_list.dt < 1.05:
        print('Warning: echo inter-spacing varies more than 5%')
        print(frame_list.echo_times)

    # take all the slices if nothing is specified in the params
    if 'slice_list' in data_param:
        frame_list.select_slices(data_param['slice_list'])

    frame_list.nx = frame_list[0].get_attr('Columns')
    frame_list.ny = frame_list[0].get_attr('Rows')

    if 'crop_fov' in data_param:
        x1, x2 = data_param['crop_fov'][0], data_param['crop_fov'][1]
        y1, y2 = data_param['crop_fov'][2], data_param['crop_fov'][3]
        frame_list.Nx, frame_list.nx = frame_list.nx, x2 - x1
        frame_list.Ny, frame_list.ny = frame_list.ny, y2 - y1
    else:
        x1, x2 = 0, frame_list.nx
        y1, y2 = 0, frame_list.ny

    img = []
    
    for n in frame_list.echo_times:

        for slice_idx in frame_list.slice_list:

            i = (data_param['nb_echoes'] * slice_idx + n) * len(type)
            
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
                    magn = magn_img[slice_idx, y1:y2, x1:x2].flatten()
                    phase = phase_img[slice_idx, y1:y2, x1:x2].flatten()
                elif len(phase_img.shape) == 2:
                    magn = magn_img[y1:y2, x1:x2].flatten()
                    phase = phase_img[y1:y2, x1:x2].flatten()
                else:
                    raise Exception(f'the image shape should have 2 or 3 dimensions, not {len(phase_img.shape)}')
                
                # get rescale intercept
                rescale_intercept = np.abs(dicom_tools.get_tag_value(phase_ds, 'Rescale Intercept', frame_list[i_phase_frame][1]))
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
                    real_part = real_img[slice_idx, y1:y2, x1:x2].flatten()
                    imag_part = imag_img[slice_idx, y1:y2, x1:x2].flatten()
                else:
                    real_part = real_img[y1:y2, x1:x2].flatten()
                    imag_part = imag_img[y1:y2, x1:x2].flatten()

                # Assumes real and imaginary slope/intercept are equal
                rescale_intercept = dicom_tools.get_tag_value(real_ds, 'Rescale Intercept', dcm_frame)
                rescale_slope = dicom_tools.get_tag_value(real_ds, 'Rescale Slope', dcm_frame)
                if rescale_intercept and rescale_slope:
                    offset = rescale_intercept/rescale_slope
                else:
                    offset = -2047.5

                c = (real_part + offset) + 1.0 * 1j * (imag_part + offset)

            else:
                raise Exception('Unknown image types')
            
            img.append(c)

    img = np.array(img) * data_param['rescale']
    new_shape = (frame_list.nb_echoes, frame_list.nz, frame_list.ny, frame_list.nx)
    frame_list.img = np.reshape(img, shape=new_shape)

    return frame_list

# Set window so that percentile % of pixels are inside
def get_percentile_window(im, intercept, slope, percentile=95):
    lims = np.percentile(im, [(100-percentile)/2, percentile + (100-percentile)/2])
    width = lims[1]-lims[0]
    center = width/2.+lims[0]
    return center * slope + intercept, width * slope

def save_series(out_dir: str, frame_list: FrameCollection, img_type:str, img: np.array, data_param: dict) -> None:
    """ Save numpy array to DICOM image.
    Based on input DICOM image if exists, else create from scratch """

    print('> Saving DICOM series')
    print(f'Image type: {img_type}')
    print(f'Output directory: {out_dir}')

    series_description = img_types_dict[img_type]['descr']
    series_number = img_types_dict[img_type]['seriesNumber']
    series_instance_uid = dicom_tools.get_series_instance_uid(data_param, series_description)

    rescale_intercept = 0. # default value
    rescale_slope = 1. # default value
    if 'Rescale Intercept' in img_types_dict[img_type]:
        rescale_intercept = img_types_dict[img_type]['Rescale Intercept']
    if 'Rescale Slope' in img_types_dict[img_type]:
        rescale_slope = img_types_dict[img_type]['Rescale Slope']
        
    # enhanced dicom have all slices in one file
    if frame_list.is_enhanced:
        ds = pydicom.read_file(frame_list[0].path)
        img_vol = np.empty([data_param['nz'], data_param['ny'] * data_param['nx']], dtype='uint16')
        frames = []

    img_types = frame_list.get_image_types()
    
    print(img_types)
    print(data_param['slice_list'])
    for z, slice in enumerate(data_param['slice_list']):

        output_filename = f'{out_dir}/0.dcm' if data_param['is_enhanced'] else f'{out_dir}/{slice}.dcm'

        # Extract slice, scale and type cast pixel data
        # pixel_data = np.array([max(0, (val-rescale_intercept)/rescale_slope) for val in img[:, :, z].flatten()])
        pixel_data = (np.array(img[:, :, z].flatten()) - rescale_intercept) / rescale_slope
        pixel_data[pixel_data < 0] = 0
        pixel_data = pixel_data.astype('uint16')
        # Set window so that 95% of pixels are inside
        window_center, window_width = get_percentile_window(pixel_data, rescale_intercept, rescale_slope, 95)
        if data_param['frame_list']:
            frame_info = data_param['frame_list'][data_param['nb_echoes'] * slice * len(img_types)]
            n_frame = frame_info[1]
            # for standard dicom, read the current slice file
            if not data_param['is_enhanced']:
                ds = pydicom.read_file(str(frame_info[0]))
        else:
            n_frame = None
            file_meta = pydicom.dataset.Dataset()
            file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
            file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.' + '1.1.111165684411017669021768385720736873780'
            file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.' + '1.0.100.4.0'
            file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            ds = pydicom.dataset.FileDataset(output_filename, {}, file_meta=file_meta, preamble=b"\0"*128)
            # Add DICOM tags:
            ds.Modality = 'WSD'
            ds.ContentDate = str(datetime.date.today()).replace('-', '')
            ds.ContentTime = str(datetime.time())  # millisecs since the epoch
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.PixelRepresentation = 0
            ds.HighBit = 15
            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.SmallestImagePixelValue = '\\x00\\x00'
            ds.LargestImagePixelValue = '\\xff\\xff'
            ds.Columns = img.shape[1]
            ds.Rows = img.shape[0]
            dicom_tools.set_tag_value(ds, 'Study Instance UID', dicom_tools.get_sop_instance_uid(), n_frame, 'UI')
            dicom_tools.set_tag_value(ds, 'Pixel Spacing', str([data_param['dx'], data_param['dy']]), n_frame, 'DS')
            dicom_tools.set_tag_value(ds, 'Pixel Aspect Ratio', str([int(data_param['dx']*100), int(data_param['dy']*100)]), n_frame, 'IS')

        # Change/add DICOM tags:
        dicom_tools.set_tag_value(ds, 'SOP Instance UID', dicom_tools.get_sop_instance_uid(), n_frame, 'UI') 
        dicom_tools.set_tag_value(ds, 'Series Instance UID', series_instance_uid, n_frame, 'UI')
        dicom_tools.set_tag_value(ds, 'Protocol Name', 'Derived Image', n_frame, 'LO')
        dicom_tools.set_tag_value(ds, 'Series Description', series_description, n_frame, 'LO')
        dicom_tools.set_tag_value(ds, 'Smallest Pixel Value', np.min(pixel_data), n_frame)
        dicom_tools.set_tag_value(ds, 'Largest Pixel Value', np.max(pixel_data), n_frame)
        dicom_tools.set_tag_value(ds, 'Window Center', str(window_center), n_frame, 'DS')
        dicom_tools.set_tag_value(ds, 'Window Width', str(window_width), n_frame, 'DS')
        dicom_tools.set_tag_value(ds, 'Rescale Intercept', str(rescale_intercept), n_frame, 'DS')
        dicom_tools.set_tag_value(ds, 'Rescale Slope', str(rescale_slope), n_frame, 'DS')
        dicom_tools.set_tag_value(ds, 'Series Number', str(series_number), n_frame, 'IS')

        if data_param['is_enhanced']:
            dicom_tools.set_tag_value(ds, 'Echo Time', 0., n_frame, 'FD')
            img_vol[z] = pixel_data
            frames.append(n_frame)
        else:
            dicom_tools.set_tag_value(ds, 'Echo Time', '0', n_frame, 'DS')
            ds.PixelData = pixel_data.tobytes()
            ds.save_as(str(output_filename))

    if data_param['is_enhanced']:
        dicom_tools.set_tag_value(ds, 'Number of frames', str(len(frames)))
        print([ds[dicom_tools.tag_dict['Frame sequence']][frame] for frame in frames])
        ds[dicom_tools.tag_dict['Frame sequence']].value = [ds[dicom_tools.tag_dict['Frame sequence']][frame] for frame in frames]
        ds.PixelData = img_vol.tobytes()
        ds.save_as(output_filename)


def pad_cropped(cropped_image: np.array, data_param: dict):
    """Zero pad back any cropped FOV"""
    if 'crop_fov' in data_param:
        image = np.zeros((data_param['nz'], data_param['Ny'], data_param['Nx']))
        x1, x2 = data_param['crop_fov'][0], data_param['crop_fov'][1]
        y1, y2 = data_param['crop_fov'][2], data_param['crop_fov'][3]
        image[:, y1:y2, x1:x2] = cropped_image
        return image
    else:
        return cropped_image

# Save all data in output as DICOM images
def save(output: dict, frame_list: FrameCollection, data_param: dict) -> None:
    
    for series_type in output:
        # zero pad if was cropped and reshape to row,col,slice
        new_shape = (frame_list.nz, frame_list.ny, frame_list.nx)
        output[series_type] = np.moveaxis(pad_cropped(output[series_type].reshape(new_shape), data_param), 0, -1)
        out_dir = data_param['out_dir'] / series_type
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'Writing images to {out_dir}')
        save_series(out_dir, series_type, output[series_type], data_param)
