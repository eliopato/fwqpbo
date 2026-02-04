import pydicom
import numpy as np
import tools
from tools import print_dt, load_nifti_as_array
import os
import shutil

gyro = 42.576  # 1H gyromagnetic ratio

# DICOM tags for output from fat/water separation
map_params_dict = {
    'phi': {'descr': 'Initial phase (degrees)', 'seriesNumber': 100},
    'wat': {'descr': 'Water-only', 'seriesNumber': 101},
    'fat': {'descr': 'Fat-only', 'seriesNumber': 102},
    'ip': {'descr': 'In-phase', 'seriesNumber': 103},
    'op': {'descr': 'Opposed-phase', 'seriesNumber': 104},    
    'ff': {'descr': 'Fat Fraction (%)', 'seriesNumber': 105},
    'r2': {'descr': 'R2* (msec-1)', 'seriesNumber': 106},
    'b0': {'descr': 'B0 inhomogeneity (ppm)', 'seriesNumber': 107},
    'CL': {'descr': 'FAC Chain length', 'seriesNumber': 108},
    'UD': {'descr': 'FAC Unsaturation degree', 'seriesNumber': 109},
    'PUD': {'descr': 'FAC Polyunsaturation degree', 'seriesNumber': 110}
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
        self.seg = None

    @property
    def slice_indexes(self) -> list:
        return sorted(list(set([s.z_slice for s in self.frame_list])))
    
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

    def set_slice_indexes(self) -> None:
        """Define the slice index of all frames in the frame list, depending on all frame locations in the list.
        To define if it's sagittal, axial or coronal we take the axis of biggest variation. It might be inacurrate in some cases"""
        x_locs = sorted(list(set([f.slice_location[0] for f in self.frame_list])))
        y_locs = sorted(list(set([f.slice_location[1] for f in self.frame_list])))
        z_locs = sorted(list(set([f.slice_location[2] for f in self.frame_list])))
        x_var = max(x_locs) - min(x_locs)
        y_var = max(y_locs) - min(y_locs)
        z_var = max(z_locs) - min(z_locs)
        if x_var > y_var and x_var > z_var:
            for f in self.frame_list:
                f.z_slice = x_locs.index(f.slice_location[0])
        elif y_var > x_var and y_var > z_var:
            for f in self.frame_list:
                f.z_slice = y_locs.index(f.slice_location[1])
        else:
            for f in self.frame_list:
                f.z_slice = z_locs.index(f.slice_location[2])

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
                    print_dt('Real/Imaginary images')
                return 'RI'
        else:    
            if num_tags['R'] == 0 and num_tags['I'] == 0 and num_tags['M'] == num_tags['P']:
                if print_type:
                    print_dt('Magnitude/Phase images')
                return 'MP'
            elif num_tags['M'] > 0 and num_tags['M'] == num_tags['R'] == num_tags['I']:
                if num_tags['P'] == 0:
                    if print_type:
                        print_dt('Magnitude/Real/Imaginary images')
                    return 'MRI'
                elif num_tags['M'] == num_tags['P']:
                    if print_type:
                        print_dt('Magnitude/Phase/Real/Imaginary images')
                    return 'MPRI'
        
        raise Exception(f'Unknown combination of image types: {num_tags}')

class Frame():

    def __init__(self, file_path: str, frame_idx: int | None, slice_location: list[float]):
        self.frame_idx = frame_idx
        self.z_slice = None
        self.slice_location = slice_location
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


def read_input_images(data_param: dict) -> FrameCollection:
    """update data_param with info retrieved from the DICOM files including image data"""
    # define if multiframe
    ds = pydicom.dcmread(str(data_param['files'][0]), stop_before_pixels=True)
    frame_coll = FrameCollection(tools.is_enhanced(ds), data_param)
    if frame_coll.is_enhanced:
        print_dt('ENHANCED DICOM')
    else:
        print_dt('STANDARD DICOM')

    # read file headers to get meta data
    for file in data_param['files']:
        ds = pydicom.dcmread(str(file), stop_before_pixels=True)
        if frame_coll.is_enhanced:
            frames = [(f, [float(v) for v in (tools.get_tag_value(ds, 'Image Position Patient', f))]) for f in range(len(tools.get_tag_value(ds, 'Frame sequence')))]
        else:
            frames = [(None, [float(v) for v in (tools.get_tag_value(ds, 'Image Position Patient'))])]
        
        for n_frame, slice_location in frames:
            new_slice = Frame(str(file), n_frame, slice_location)
            for attr in tools.req_attributes:
                new_slice.add_attr(attr, tools.get_tag_value(ds, attr, n_frame))
            frame_coll.add_frame(new_slice)
    
    frame_coll.set_slice_indexes()
    img_types = frame_coll.get_image_types(True)
    first_slice = frame_coll[0]
    frame_coll.dx = float(first_slice.get_attr('Pixel Spacing')[1])
    frame_coll.dy = float(first_slice.get_attr('Pixel Spacing')[0])
    frame_coll.dz = float(first_slice.get_attr('Slice Thickness'))
    frame_coll.b0 = first_slice.get_attr('Imaging Frequency')/gyro

    # get the sorted list of echo times
    frame_coll.echo_times = sorted(set([et for et in frame_coll.get_all_attr_values('Echo Time')]))
    
    # select specified echoes from parameter file
    if 'echoes' in data_param:
        print_dt('dropping echoes...')
        frame_coll.echo_times = [frame_coll.echo_times[i] for i in data_param['echoes']]
    
    # check the number of echoes
    if frame_coll.n_echo < 2:
        raise Exception(f'At least 2 echoes required, only {frame_coll.n_echo} found')
    
    ms_echo_times = [et/1000 for et in frame_coll.echo_times]
    frame_coll.t1 = ms_echo_times[0] 
    frame_coll.dt = np.mean(np.diff(ms_echo_times)) 
    if not 0.95 < np.max(np.diff(ms_echo_times))/frame_coll.dt < 1.05:
        print_dt('** Warning: echo inter-spacing varies more than 5%')

    # read the segmentation file
    if 'seg' in data_param:
        if 'file' in data_param['seg']:
            filepath =  data_param['seg']['file']
            if not filepath.endswith('nii.gz'):
                print_dt('ERROR: can\'t read segmentation file : should be in nifti format (.nii.gz)')
            else:
                print_dt(f'Reading segmentation file from {filepath}')
                frame_coll.seg = {'vol': load_nifti_as_array(filepath)}
                frame_coll.seg.update(data_param['seg'])

    # select specified slices, else take all the slices if nothing is specified in the params
    if 'slice_list' in frame_coll.user_params and frame_coll.user_params['slice_list']:
        selected_slices = frame_coll.user_params['slice_list']
        nb_slices = frame_coll.n_slice_indexes
        print_dt(f'selecting specified slices out of {nb_slices} slices... ({selected_slices})')
        # select slices in segmentation volume
        if frame_coll.seg  is not None:
            seg_shape = frame_coll.seg['vol'].shape
            if nb_slices not in seg_shape:
                print(f'ERROR: wrong number of slices in provided segmentation file (number of slices in input image: {nb_slices} / segmentation shape : {seg_shape}) ')
                seg_shape = None
            else:
                slice_dim = seg_shape.index(nb_slices)
                slices = [slice(None)] * len(seg_shape)  # Initialize with `:` for all dimensions
                slices[slice_dim] = selected_slices  # Replace the slice_dim with selected_slices
                frame_coll.seg['vol'] = frame_coll.seg['vol'][tuple(slices)]
        # select slices in input volume
        frame_coll.select_frames(selected_slices)

    frame_coll.nx = frame_coll[0].get_attr('Columns')
    frame_coll.ny = frame_coll[0].get_attr('Rows')

    if 'crop_fov' in frame_coll.user_params and frame_coll.user_params['crop_fov'] is not None:
        print_dt(f'Crop field of view to {frame_coll.user_params["crop_fov"]}')
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
                print_dt(f'Error: You should have {len(img_types)} frames for a given echo time/z_index for {img_types} images')
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

                rescale_intercept = tools.get_tag_value(ds, 'Rescale Intercept', n_frame)
                rescale_slope = tools.get_tag_value(ds, 'Rescale Slope', n_frame)

                frames_data[f.get_attr('Image Type')] = {'img': frame_img, 
                                                         'ds': ds, 
                                                         'rescale_intercept': rescale_intercept if rescale_intercept is not None else 0,
                                                         'rescale_slope': rescale_slope if rescale_slope is not None else 1,
                                                         'rescale_type': tools.get_tag_value(ds, 'Rescale Type', n_frame)
                                                         }
            
            # Magnitude/phase images
            if img_types == 'MP':  
                # we need to convert phase image from their original values ranges (e.g. [0:4095]) to [-pi:pi]. 
                if ds.Manufacturer.lower().startswith('philips'):
                    rescale_type = frames_data['P']['rescale_type']
                    if rescale_type is None or rescale_type == 'mrad':
                        pscale = 0.001
                    else:
                        raise RuntimeError(f'Unknown Phase RescaleType: {rescale_type}')
                else:
                    # 4096 comes from the max bits allocated that should be 12 (2^12 = 4096) - see dicom tag BitsAllocated
                    pscale = np.pi/4096
                    if not ds.Manufacturer.lower().startswith('siemens'):
                        print_dt(f'Phase scaling value not specified for manufacturer {ds.Manufacturer}, using default (pi/4096)')

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

    print_dt('done reading input')
    return frame_coll


def save(output: dict, frame_coll: FrameCollection) -> None:
    """ Save numpy array to DICOM image."""

    nz = frame_coll.n_slice_indexes
    # DICOM usually encodes image values on 12 bits and keeps 4 bits for overlays. 
    # you can set nb_bits_used to max 16 to make full usage of uint16 storage
    nb_bits_used = 12
    max_pixel = (2**nb_bits_used) - 1 

    for map_type in output:
        # zero pad if was cropped and reshape to row,col,slice
        new_shape = (nz, frame_coll.ny, frame_coll.nx)
        output[map_type], nx, ny = pad_cropped(output[map_type].reshape(new_shape), frame_coll)
        output[map_type] = np.moveaxis(output[map_type], 0, -1)
        dcm_out_dir = frame_coll.user_params['out_dir'] / map_type
        # remove any existing dicom directory 
        if os.path.exists(dcm_out_dir):
            shutil.rmtree(dcm_out_dir)
        dcm_out_dir.mkdir(parents=True, exist_ok=True)

        print_dt('> Saving DICOM map')
        print_dt(f'Map type: {map_type}')
        print_dt(f'Output directory: {dcm_out_dir}')

        series_description = map_params_dict[map_type]['descr']
        series_number = map_params_dict[map_type]['seriesNumber']
        series_instance_uid = tools.get_series_instance_uid(frame_coll.user_params, series_description)
            
        # enhanced dicom have all slices in one file
        if frame_coll.is_enhanced:
            ds = pydicom.dcmread(frame_coll[0].path)
            ds[tools.tag_dict['Frame sequence']].value = [ds[tools.tag_dict['Frame sequence']].value[n] for n in frame_coll.slice_indexes]
            img_vol = np.empty([nz, ny * nx], dtype='uint16')
        
        te1 = frame_coll.echo_times[0]

        for slice_index in frame_coll.slice_indexes:

            # Extract slice, scale and type cast pixel data
            z_index = frame_coll.slice_indexes.index(slice_index)            
            pixel_data = np.array(output[map_type][:, :, z_index].flatten())
            rescale_intercept = np.floor(pixel_data.min())
            rescale_slope = (pixel_data.max() - rescale_intercept)/max_pixel
            pixel_data = (pixel_data - rescale_intercept) / rescale_slope
            pixel_data = np.round(pixel_data).astype('uint16')

            # for standard dicom, read the current slice file
            if not frame_coll.is_enhanced:
                ds = pydicom.dcmread(str(frame_coll.get_frames(slice_index, te1)[0].path))
        
            # Change/add DICOM tags:
            tools.set_tag_value(ds, 'SOP Instance UID', tools.get_sop_instance_uid(), z_index, 'UI') 
            tools.set_tag_value(ds, 'Series Instance UID', series_instance_uid, z_index, 'UI')
            tools.set_tag_value(ds, 'Protocol Name', 'Derived Image', z_index, 'LO')
            tools.set_tag_value(ds, 'Series Description', series_description, z_index, 'LO')
            tools.set_tag_value(ds, 'Smallest Pixel Value', 0, z_index)
            tools.set_tag_value(ds, 'Largest Pixel Value', max_pixel, z_index)
            tools.set_tag_value(ds, 'Window Center', str(max_pixel//2), z_index, 'DS')
            tools.set_tag_value(ds, 'Window Width', str(max_pixel), z_index, 'DS')
            tools.set_tag_value(ds, 'Rescale Intercept', str(rescale_intercept), z_index, 'DS')
            tools.set_tag_value(ds, 'Rescale Slope', format(rescale_slope, '.10e'), z_index, 'DS')
            tools.set_tag_value(ds, 'Series Number', str(series_number), z_index, 'IS')
            tools.set_tag_value(ds, 'Image Type', ['DERIVED'], z_index, 'CS')
            ds.BitsAllocated = 16
            ds.BitsStored = nb_bits_used
            ds.HighBit = nb_bits_used - 1

            if frame_coll.is_enhanced:
                tools.set_tag_value(ds, 'Echo Time', 0., z_index, 'FD')
                img_vol[z_index] = pixel_data
            else:
                tools.set_tag_value(ds, 'Echo Time', '0', z_index, 'DS')
                ds.PixelData = pixel_data.tobytes()
                ds.save_as(f'{dcm_out_dir}/{z_index}.dcm')

        if frame_coll.is_enhanced:
            tools.set_tag_value(ds, 'Number of frames', str(nz))
            ds.PixelData = img_vol.tobytes()
            ds.save_as(f'{dcm_out_dir}/0.dcm')

        # saving in Nifti format
        output_filepath = f'{dcm_out_dir}.nii.gz'
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        cmd = f"dcm2niix -z y -f {map_type} -m y -b n -o {frame_coll.user_params['out_dir']} {dcm_out_dir}"# > /dev/null 2>&1"
        print(cmd)
        os.system(cmd)


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