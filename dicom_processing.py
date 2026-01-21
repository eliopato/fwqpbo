import pydicom
import numpy as np
import dicom_tools
import re

gyro = 42.576  # 1H gyromagnetic ratio

# DICOM tags for output from fat/water separation
map_params_dict = {
    'phi': {'descr': 'Initial phase (degrees)', 'seriesNumber': 100},
    'wat': {'descr': 'Water-only', 'seriesNumber': 101},
    'fat': {'descr': 'Fat-only', 'seriesNumber': 102},
    'ip': {'descr': 'In-phase', 'seriesNumber': 103},
    'op': {'descr': 'Opposed-phase', 'seriesNumber': 104},    
    'ff': {'descr': 'Fat Fraction (%)', 'seriesNumber': 105, 'RescaleIntercept': -100, 'RescaleSlope': 0.1},
    'r2_map': {'descr': 'R2* (msec-1)', 'seriesNumber': 106},
    'b0_map': {'descr': 'B0 inhomogeneity (ppm)', 'seriesNumber': 107, 'RescaleSlope': 0.001},
    'CL': {'descr': 'FAC Chain length', 'seriesNumber': 108, 'RescaleSlope': 0.01},
    'UD': {'descr': 'FAC Unsaturation degree', 'seriesNumber': 109, 'RescaleSlope': 0.01},
    'PUD': {'descr': 'FAC Polyunsaturation degree', 'seriesNumber': 110, 'RescaleSlope': 0.01}
}


class FrameCollection():

    def __init__(self, is_enhanced: bool, user_params: dict):
        self.frame_list = []
        self.user_params = user_params
        self.is_enhanced = is_enhanced
        self.dx = None
        self.dy = None
        self.dz = None
        self.nx = None
        self.ny = None
        self.Nx = None
        self.Ny = None
        self.b0 = None
        self.t1 = None
        self.dt = None
        self.echo_times = None

    @property
    def slice_indexes(self) -> list:
        return list(set([s.z_slice for s in self.frame_list]))
    
    @property
    def n_slice_indexes(self) -> int:
        return len(self.slice_indexes)
    
    @property
    def n_frames(self) -> int:
        return len(self.frame_list)
        
    @property
    def n_echo(self) -> int:
        return 0 if self.echo_times is None else len(self.echo_times)
    
    def add_frame(self, frame: Frame) -> None:
        self.frame_list.append(frame)

    def get_frames(self, z_idx: int | None, echo_time: float | None) -> list[Frame]:
        if z_idx is None and echo_time is None:
            return self.frame_list
        if echo_time is None:
            return [s for s in self.frame_list if s.z_slice == z_idx]
        if z_idx is None:
            return [s for s in self.frame_list if s.get_attr('Echo Time') == echo_time]
        return [s for s in self.frame_list if s.get_attr('Echo Time') == echo_time and s.z_slice == z_idx]
        
    
    def __getitem__(self, index: int) -> Frame:
        return self.frame_list[index]
    
    def get_all_attr_values(self, attr_name: str) -> list:
        return [s.get_attr(attr_name) for s in self.frame_list]

    def select_frames(self, selected_z_indexes: list[int] | None) -> None:
        """Drop slices that are not in the provided list"""
        self.frame_list = [s for s in self.frame_list if s.z_slice in selected_z_indexes]

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

    def __init__(self, file_path: str, frame_idx: int | None, z_slice: int):
        self.frame_idx = frame_idx
        self.z_slice = z_slice
        self.path = file_path
    
    def add_attr(self, attr_name, attr_value):
        self.__setattr__(attr_name, attr_value)

    def get_attr(self, attr_name):
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        else:
            return None


def series_description_to_type(series_description):
    """Translates series description tag to M/P/R/I for magn/phase/real/imaginary"""
    for type, description in [('R', 'Real Image'), ('I', 'Imag Image')]:
        if description in series_description:
            return type
    return None


def type_tag_to_type(tag_value):
    """Translates image type tag to M/P/R/I for magnitude/phase/real/imaginary"""
    for type in ['M', 'P', 'R', 'I']:
        if type in tag_value:
            return type
    return None


# update data_param with info retrieved from the DICOM files including image data
def read_input_images(data_param: dict):

    # define if multiframe
    ds = pydicom.dcmread(str(data_param['files'][0]), stop_before_pixels=True)
    frame_coll = FrameCollection(dicom_tools.is_enhanced(ds), data_param)

    # read file headers to get meta data
    for file in data_param['files']:
        ds = pydicom.dcmread(str(file), stop_before_pixels=True)
        # try to get Z index from DICOM tags
        if frame_coll.is_enhanced:
            frames = [(f, f) for f in range(len(dicom_tools.get_tag_value(ds, 'Frame sequence')))]
        else:
            frames = [(None, dicom_tools.get_tag_value(ds, 'Frame Number'))]

        for n_frame, z_slice in frames:
            new_slice = Frame(str(file), n_frame, z_slice)
            for attr in dicom_tools.req_attributes:
                new_slice.add_attr(attr, dicom_tools.get_tag_value(ds, attr, n_frame))
            frame_coll.add_frame(new_slice)
    
    img_types = frame_coll.get_image_types(True)
    first_slice = frame_coll[0]
    frame_coll.dx = float(first_slice.get_attr('Pixel Spacing')[1])
    frame_coll.dy = float(first_slice.get_attr('Pixel Spacing')[0])
    frame_coll.dz = float(first_slice.get_attr('Slice Thickness'))
    frame_coll.b0 = first_slice.get_attr('Imaging Frequency')/gyro

    # get the sorted list of echo times
    frame_coll.echo_times = sorted(set([et for et in frame_coll.get_all_attr_values('Echo Time')]))
    
    # standard dicom, try to recover the frame number from filename if it wasnt recovered from attributes
    if not frame_coll.is_enhanced and frame_coll.slice_indexes[0] is None:
        for i, frame in enumerate(frame_coll):
            z_slice = re.findall(r'.*_e0*[0-9]+_0*([0-9]+)\.dcm$', str(frame.path)) 
            if len(z_slice) != 1:
                print('Error, couldnt determine slice index from dicom header nor file name. Please adapt the regex to your file names. Use file index.')
                frame.z_slice = (i % (n_slices))
            else:
                n_slices = len(frame_coll.frame_list) // (frame_coll.n_echo * len(frame_coll.get_image_types()))
                frame.z_slice = (int(z_slice[0]) % (n_slices))

    # select specified echoes from parameter file
    if 'echoes' in data_param:
        print('dropping echoes...')
        frame_coll.echo_times = [frame_coll.echo_times[i] for i in data_param['echoes']]
    
    # check the number of echoes
    if frame_coll.n_echo < 2:
        raise Exception(f'At least 2 echoes required, only {frame_coll.n_echo} found')
    
    ms_echo_times = [et/1000 for et in frame_coll.echo_times]
    frame_coll.t1 = ms_echo_times[0] 
    frame_coll.dt = np.mean(np.diff(ms_echo_times)) 
    if not 0.95 < np.max(np.diff(ms_echo_times))/frame_coll.dt < 1.05:
        print('Warning: echo inter-spacing varies more than 5%')
        print(frame_coll.echo_times)

    # take all the slices if nothing is specified in the params
    if 'slice_list' in frame_coll.user_params:
        print('selecting specified slices...')
        frame_coll.select_frames(frame_coll.user_params['slice_list'])

    frame_coll.nx = frame_coll[0].get_attr('Columns')
    frame_coll.ny = frame_coll[0].get_attr('Rows')

    if 'crop_fov' in frame_coll.user_params:
        x1, x2 = frame_coll.user_params['crop_fov'][0], frame_coll.user_params['crop_fov'][1]
        y1, y2 = frame_coll.user_params['crop_fov'][2], frame_coll.user_params['crop_fov'][3]
        frame_coll.Nx, frame_coll.nx = frame_coll.nx, x2 - x1
        frame_coll.Ny, frame_coll.ny = frame_coll.ny, y2 - y1
    else:
        x1, x2 = 0, frame_coll.nx
        y1, y2 = 0, frame_coll.ny

    img = []
    
    for echo_time in frame_coll.echo_times:

        for z_index in frame_coll.slice_indexes:

            frames = frame_coll.get_frames(z_index, echo_time)
            if len(frames) != len(img_types):
                print(f'Error: You should have {len(img_types)} or frames for a given echo time/z_index for {img_types} images')
                continue

            n_frame = None
            frames_data = {}
        
            for f in frames:

                # read frames data
                n_frame = f.frame_idx
                ds = pydicom.dcmread(str(f.path))
                frame_img = ds.pixel_array
                # read the current slice, depending on the array shape (enhanced images have the slice as the 1st dimension)
                if len(frame_img.shape) == 3:
                    frame_img = frame_img[z_index, y1:y2, x1:x2].flatten()
                elif len(frame_img.shape) == 2:
                    frame_img = frame_img[y1:y2, x1:x2].flatten()
                else:
                    raise Exception(f'the image shape should have 2 or 3 dimensions, not {len(img.shape)}')

                rescale_intercept = dicom_tools.get_tag_value(ds, 'Rescale Intercept', n_frame)
                rescale_slope = dicom_tools.get_tag_value(ds, 'Rescale Slope', n_frame)

                frames_data[f.get_attr('Image Type')] = {'img': frame_img, 
                                                         'ds': ds, 
                                                         'rescale_intercept': rescale_intercept if rescale_intercept is not None else 0,
                                                         'rescale_slope': rescale_slope if rescale_slope is not None else 1,
                                                         'rescale_type': dicom_tools.get_tag_value(ds, 'Rescale Type', n_frame)
                                                         }
            
            # Magnitude/phase images
            if img_types == 'MP':  
                # we need to convert phase image from their original values ranges (e.g. [0:4095]) to [-pi:pi]. 
                if ds.Manufacturer.startswith('Philips'):
                    rescale_type = frames_data['P']['rescale_type']
                    if rescale_type is None or rescale_type == 'mrad':
                        pscale = 0.001
                    else:
                        raise RuntimeError(f'Unknown Phase RescaleType: {rescale_type}')
                else:
                    # 4096 comes from the max bits allocated that should be 12 (2^12 = 4096) - see dicom tag BitsAllocated
                    pscale = np.pi/4096
                    if not ds.Manufacturer.startswith('Siemens'):
                        print(f'Phase scaling value not specified for manufacturer {ds.Manufacturer}, using default (pi/4096)')

                magn_img = frames_data['M']['img'] * frames_data['M']['rescale_slope'] + frames_data['M']['rescale_intercept']
                phase_img = frames_data['P']['img'] * frames_data['P']['rescale_slope'] + frames_data['P']['rescale_intercept']
                c = magn_img * np.exp(1j * pscale * phase_img)
                
            # Real/imaginary images and Magnitude/real/imaginary images -> only use R and I images
            elif img_types in ['RI', 'MRI', 'MPRI']:
                
                real_part = frames_data['R']['img'] * frames_data['R']['rescale_intercept'] + frames_data['R']['rescale_slope']
                imag_part = frames_data['I']['img'] * frames_data['I']['rescale_intercept'] + frames_data['I']['rescale_slope']
                c =  real_part + 1j * (imag_part)

            else:
                raise RuntimeError('Unknown image types')
            
            img.append(c)

    img = np.array(img) * frame_coll.user_params['rescale']
    new_shape = (frame_coll.n_echo, frame_coll.n_slice_indexes, frame_coll.ny, frame_coll.nx)
    frame_coll.img = np.reshape(img, shape=new_shape)

    return frame_coll

# Save all data in output as DICOM images
def save(output: dict, frame_coll: FrameCollection) -> None:
    """ Save numpy array to DICOM image."""

    nz = frame_coll.n_slice_indexes
    # to keep values as precise as possible, map to full 16 bits values (max int 65535 instead of 4095 initially)
    nb_bits_used = 16
    max_pixel = (2**16) - 1  # rescale to make full usage of uint16 storage

    for map_type in output:
        # zero pad if was cropped and reshape to row,col,slice
        new_shape = (nz, frame_coll.ny, frame_coll.nx)
        output[map_type], nx, ny = pad_cropped(output[map_type].reshape(new_shape), frame_coll)
        output[map_type] = np.moveaxis(output[map_type], 0, -1)
        out_dir = frame_coll.user_params['out_dir'] / map_type
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'Writing images to {out_dir}')

        print('> Saving DICOM map')
        print(f'Map type: {map_type}')
        print(f'Output directory: {out_dir}')

        series_description = map_params_dict[map_type]['descr']
        series_number = map_params_dict[map_type]['seriesNumber']
        series_instance_uid = dicom_tools.get_series_instance_uid(frame_coll.user_params, series_description)
            
        # enhanced dicom have all slices in one file
        if frame_coll.is_enhanced:
            ds = pydicom.dcmread(frame_coll[0].path)
            img_vol = np.empty([nz, ny * nx], dtype='uint16')
        
        for frame in frame_coll:

            output_filename = f'{out_dir}/0.dcm' if frame_coll.is_enhanced else f'{out_dir}/{frame.z_slice}.dcm'

            # Extract slice, scale and type cast pixel data
            pixel_data = np.array(output[map_type][:, :, frame.z_slice].flatten())
            # pixel_data[pixel_data < 0] = 0
            rescale_intercept = np.floor(pixel_data.min())
            rescale_slope = (pixel_data.max() - rescale_intercept)/max_pixel
            pixel_data = (pixel_data - rescale_intercept) / rescale_slope
            pixel_data = np.round(pixel_data).astype('uint16')

            # for standard dicom, read the current slice file
            if not frame_coll.is_enhanced:
                ds = pydicom.dcmread(str(frame.path))
        
            # Change/add DICOM tags:
            dicom_tools.set_tag_value(ds, 'SOP Instance UID', dicom_tools.get_sop_instance_uid(), frame.frame_idx, 'UI') 
            dicom_tools.set_tag_value(ds, 'Series Instance UID', series_instance_uid, frame.frame_idx, 'UI')
            dicom_tools.set_tag_value(ds, 'Protocol Name', 'Derived Image', frame.frame_idx, 'LO')
            dicom_tools.set_tag_value(ds, 'Series Description', series_description, frame.frame_idx, 'LO')
            dicom_tools.set_tag_value(ds, 'Smallest Pixel Value', 0, frame.frame_idx)
            dicom_tools.set_tag_value(ds, 'Largest Pixel Value', max_pixel, frame.frame_idx)
            dicom_tools.set_tag_value(ds, 'Window Center', str(max_pixel/2), frame.frame_idx, 'DS')
            dicom_tools.set_tag_value(ds, 'Window Width', str(max_pixel), frame.frame_idx, 'DS')
            dicom_tools.set_tag_value(ds, 'Rescale Intercept', str(rescale_intercept), frame.frame_idx, 'DS')
            dicom_tools.set_tag_value(ds, 'Rescale Slope', format(rescale_slope, '.10e'), frame.frame_idx, 'DS')
            dicom_tools.set_tag_value(ds, 'Series Number', str(series_number), frame.frame_idx, 'IS')
            dicom_tools.set_tag_value(ds, 'Image Type', '["DERIVED", "PRIMARY"]', frame.frame_idx, 'IS')
            ds.BitsStored = nb_bits_used
            ds.BitsAllocated = nb_bits_used
            ds.HighBit = nb_bits_used - 1

            if frame_coll.is_enhanced:
                dicom_tools.set_tag_value(ds, 'Echo Time', 0., frame.frame_idx, 'FD')
                img_vol[frame.z_slice] = pixel_data
            else:
                dicom_tools.set_tag_value(ds, 'Echo Time', '0', frame.frame_idx, 'DS')
                ds.PixelData = pixel_data.tobytes()
                ds.save_as(str(output_filename))

        if frame_coll.is_enhanced:
            dicom_tools.set_tag_value(ds, 'Number of frames', str(nz))
            ds.PixelData = img_vol.tobytes()
            ds.save_as(output_filename)


def pad_cropped(cropped_image: np.array, frame_coll: FrameCollection) -> tuple[np.array, int, int]:
    """Zero pad back any cropped FOV.
    Returns the padded image as a numpy array, the new number of rows (nx) and the new number of columns (ny)"""
    if 'crop_fov' in frame_coll.user_params:
        image = np.zeros((frame_coll.n_slice_indexes, frame_coll.Ny, frame_coll.Nx))
        x1, x2, y1, y2 = frame_coll.user_params['crop_fov']
        image[:, y1:y2, x1:x2] = cropped_image
        return image, frame_coll.Nx, frame_coll.Ny
    else:
        return cropped_image, frame_coll.nx, frame_coll.ny


