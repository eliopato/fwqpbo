import pydicom
import numpy as np
import dicom_tools
import re

gyro = 42.58  # 1H gyromagnetic ratio

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
        self.frame_list = [s for s in self.frame_list if s.z_index in selected_z_indexes]

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
    ds = pydicom.read_file(str(data_param['files'][0]), stop_before_pixels=True)
    frame_coll = FrameCollection(dicom_tools.is_enhanced(ds), data_param)

    # read file headers to get meta data
    for file in data_param['files']:
        ds = pydicom.read_file(str(file), stop_before_pixels=True)
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
        print('dropping echoes?')
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
        frame_coll.select_slices(frame_coll.user_params['slice_list'])

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

            rescale_intercept = 0
            rescale_slope = 0
            n_frame = None
            frames_data = {}
        
            for f in frames:

                # read frames data
                n_frame = f.frame_idx
                ds = pydicom.read_file(str(f.path))
                frame_img = ds.pixel_array
                # read the current slice, depending on the array shape (enhanced images have the slice as the 1st dimension)
                if len(frame_img.shape) == 3:
                    frame_img = frame_img[z_index, y1:y2, x1:x2].flatten()
                elif len(frame_img.shape) == 2:
                    frame_img = frame_img[y1:y2, x1:x2].flatten()
                else:
                    raise Exception(f'the image shape should have 2 or 3 dimensions, not {len(img.shape)}')

                frames_data[f.get_attr('Image Type')] = {'img': frame_img, 'ds': ds}
            
            # Magnitude/phase images
            if img_types == 'MP':  
            
                # get rescale intercept
                rescale_intercept = np.abs(dicom_tools.get_tag_value(frames_data['P']['ds'], 'Rescale Intercept', n_frame))
                if rescale_intercept is None or rescale_intercept == 0:
                    print('No Rescale Intercept DICOM tag found. Using 4096.')
                    rescale_intercept = 4096
                # Abs val needed for Siemens data to get correct phase sign
                rescale_intercept = float(np.abs(rescale_intercept))

                # For some reason, intercept is used as slope (Siemens only?)
                c = frames_data['M']['img'] * np.exp(frames_data['P']['img']/rescale_intercept * 2 * np.pi * 1j)
            
            # Real/imaginary images and Magnitude/real/imaginary images -> only use R and I images
            elif img_types in ['RI', 'MRI', 'MPRI']:
                
                # Assumes real and imaginary slope/intercept are equal
                rescale_intercept = dicom_tools.get_tag_value(frames_data['R']['ds'], 'Rescale Intercept', n_frame)
                rescale_slope = dicom_tools.get_tag_value(frames_data['R']['ds'], 'Rescale Slope', n_frame)
                if rescale_intercept and rescale_slope:
                    offset = rescale_intercept/rescale_slope
                else:
                    offset = -2047.5

                c = (frames_data['R']['img'] + offset) + 1.0 * 1j * (frames_data['I']['img'] + offset)

            else:
                raise Exception('Unknown image types')
            
            img.append(c)

    img = np.array(img) * frame_coll.user_params['rescale']
    new_shape = (frame_coll.n_echo, frame_coll.n_slice_indexes, frame_coll.ny, frame_coll.nx)
    frame_coll.img = np.reshape(img, shape=new_shape)

    return frame_coll

# Set window so that percentile % of pixels are inside
def get_percentile_window(im, intercept, slope, percentile=95):
    lims = np.percentile(im, [(100-percentile)/2, percentile + (100-percentile)/2])
    width = lims[1]-lims[0]
    center = width/2.+lims[0]
    return center * slope + intercept, width * slope


# Save all data in output as DICOM images
def save(output: dict, frame_coll: FrameCollection) -> None:
    """ Save numpy array to DICOM image."""

    nz = frame_coll.n_slice_indexes

    for map_type in output:
        # zero pad if was cropped and reshape to row,col,slice
        new_shape = (nz, frame_coll.ny, frame_coll.nx)
        output[map_type] = np.moveaxis(pad_cropped(output[map_type].reshape(new_shape), frame_coll.user_params), 0, -1)
        out_dir = frame_coll.user_params['out_dir'] / map_type
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'Writing images to {out_dir}')

        print('> Saving DICOM map')
        print(f'Map type: {map_type}')
        print(f'Output directory: {out_dir}')

        series_description = map_params_dict[map_type]['descr']
        series_number = map_params_dict[map_type]['seriesNumber']
        series_instance_uid = dicom_tools.get_series_instance_uid(frame_coll.user_params, series_description)

        rescale_intercept = 0. # default value
        rescale_slope = 1. # default value
        if 'Rescale Intercept' in map_params_dict[map_type]:
            rescale_intercept = map_params_dict[map_type]['Rescale Intercept']
        if 'Rescale Slope' in map_params_dict[map_type]:
            rescale_slope = map_params_dict[map_type]['Rescale Slope']
            
        # enhanced dicom have all slices in one file
        if frame_coll.is_enhanced:
            ds = pydicom.read_file(frame_coll[0].path)
            img_vol = np.empty([nz, frame_coll.ny * frame_coll.nx], dtype='uint16')
        
        for frame in frame_coll:

            output_filename = f'{out_dir}/0.dcm' if frame_coll.is_enhanced else f'{out_dir}/{frame.z_slice}.dcm'

            # Extract slice, scale and type cast pixel data
            pixel_data = (np.array(output[map_type][:, :, frame.z_slice].flatten()) - rescale_intercept) / rescale_slope
            pixel_data[pixel_data < 0] = 0
            pixel_data = pixel_data.astype('uint16')
            # Set window so that 95% of pixels are inside
            window_center, window_width = get_percentile_window(pixel_data, rescale_intercept, rescale_slope, 95)
            
            # for standard dicom, read the current slice file
            if not frame_coll.is_enhanced:
                ds = pydicom.read_file(str(frame.path))
        
            # Change/add DICOM tags:
            dicom_tools.set_tag_value(ds, 'SOP Instance UID', dicom_tools.get_sop_instance_uid(), frame.frame_idx, 'UI') 
            dicom_tools.set_tag_value(ds, 'Series Instance UID', series_instance_uid, frame.frame_idx, 'UI')
            dicom_tools.set_tag_value(ds, 'Protocol Name', 'Derived Image', frame.frame_idx, 'LO')
            dicom_tools.set_tag_value(ds, 'Series Description', series_description, frame.frame_idx, 'LO')
            dicom_tools.set_tag_value(ds, 'Smallest Pixel Value', np.min(pixel_data), frame.frame_idx)
            dicom_tools.set_tag_value(ds, 'Largest Pixel Value', np.max(pixel_data), frame.frame_idx)
            dicom_tools.set_tag_value(ds, 'Window Center', str(window_center), frame.frame_idx, 'DS')
            dicom_tools.set_tag_value(ds, 'Window Width', str(window_width), frame.frame_idx, 'DS')
            dicom_tools.set_tag_value(ds, 'Rescale Intercept', str(rescale_intercept), frame.frame_idx, 'DS')
            dicom_tools.set_tag_value(ds, 'Rescale Slope', str(rescale_slope), frame.frame_idx, 'DS')
            dicom_tools.set_tag_value(ds, 'Series Number', str(series_number), frame.frame_idx, 'IS')

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


