#!/usr/bin/env python3

"""
Pydicom utility functions
Author: SCS
Date: 26.07.2021
"""

import os
import glob
import logging
import numpy as np

from pydicom import dcmread
from pydicom.dataset import FileDataset


def swap_ras_lps(values: np.array) -> np.array:
    """
    Swap between ras and lps anatomical coordinate system
    @param values:
    @return:
    """
    values = np.array(values)
    shape = np.array(values.shape)

    if values.ndim == 1:
        if len(values) != 3:
            raise ValueError(f'Swapping only defined for 3d coordinates ({len(values)})')
        values[:2] *= -1
    elif values.ndim == 2:
        if shape[1] != 3:
            raise ValueError(f'Swapping only defined for 3d coordinates ({shape[1]})')
        values[:, :2] *= -1
    elif values.ndim == 3:
        values[:, :, :] *= -1
    else:
        raise ValueError(f'Swapping not defined for ndim>3 ({values.ndim})')
    return values


def get_pixels_hu(ds) -> np.ndarray:
    """
    Get Houndsfeld units (HU) from DICOM images
    Note: Inspired from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
    @param ds: DICOM images (List of pydicom.dataset.FileDataset)
    @return: Array with HU
    """
    image = np.nan
    if not isinstance(ds, list) and isinstance(ds[0], FileDataset):
        logging.error(f'Expected multiple datasets (got {len(ds)})')
    else:
        image = np.stack([s.pixel_array for s in ds])
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 1
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        intercept = ds[0].RescaleIntercept
        slope = ds[0].RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
        image = np.array(image, dtype=np.int16)
    return image


def get_dcm_offset(ds) -> np.ndarray:
    """

    @param ds: DICOM image (pydicom.dataset.FileDataset)
    @return: DICOM offset / patient position in mm (x, y, z)
    """
    ret = (np.nan, np.nan, np.nan)
    ds_read = ds.copy()
    if isinstance(ds, list) and isinstance(ds[0], FileDataset) and len(ds) > 1:
        logging.warning(f'Passed multiple datasets {len(ds)}, use the first one.')
        ds_read = ds[0].copy()
    try:
        ret = np.array(ds_read.ImagePositionPatient, dtype=float)
    except Exception as e:
        print(e)
    return ret


def get_vol_size(ds) -> tuple[int, int, int]:
    """
    Get volume size of DICOM images.
    @param ds: DICOM images (List of pydicom.dataset.FileDataset)
    @return: Volume size (Nx x Ny x Nz)
    """
    first_slice = ds[0]
    # Note: XY dimensions are swapped because pydicom uses C-order indexing (see also get_image_voxel_data())
    Nx = first_slice.Columns
    Ny = first_slice.Rows
    Nz = len(ds)
    ret = int(Nx), int(Ny), int(Nz)
    return ret


def get_pixel_spacing(ds) -> np.ndarray:
    """
    Get pixel spacings from patient dataset.
    @param ds: DICOM images (List of pydicom.dataset.FileDataset)
    @return: DICOM pixel spacing
    """

    if len(ds) < 2:
        spacing = None
        logging.error(f'At least 2 DICOM datasets required (got {len(ds)})')
    else:
        first_slice = ds[0]
        second_slice = ds[1]

        res_xy = np.array(first_slice.PixelSpacing)
        res_z = second_slice.ImagePositionPatient[2] - first_slice.ImagePositionPatient[2]
        if res_z <= 0.1:
            # vector with direction cosines (ImageOrientationPatient) not aligned with major axes
            # TODO: Reconstruction interval - the spacing between adjacent slices - is independent of slice thickness
            #  in helical CT (but the same in step-and-shoot CT)
            logging.warning(f'Invalid spacing in z-axis {res_z:.2f}, use SliceThickness')
            res_z = first_slice.SliceThickness
            if res_z <= 0:
                raise ValueError(f'Failed to read spacing in z-axis {res_z}')
        spacing = np.array([res_xy[0], res_xy[1], res_z])
    return spacing


def print_dcm_meta(ds):
    if not isinstance(ds, list) and isinstance(ds[0], FileDataset):
        logging.error(f'Expected multiple datasets (got {len(ds)})')
    else:
        img_orient = np.array(ds[0].ImageOrientationPatient)
        dcm_offset_first = get_dcm_offset(ds[0])
        dcm_offset_last = get_dcm_offset(ds[-1])
        Nx, Ny, Nz = get_vol_size(ds)
        spacing = get_pixel_spacing(ds)

        print(f'ImageOrientationPatient: {img_orient}')
        print(f'ImagePositionPatient first slice: {dcm_offset_first}')
        print(f'ImagePositionPatient last slice: {dcm_offset_last}')
        print(f'Spacing X[mm]: {spacing[0]}')
        print(f'Spacing Y[mm]: {spacing[1]}')
        print(f'Spacing Z[mm]: {spacing[2]}')
        print(f'Volume size {Nx}x{Ny}x{Nz} (x, y, z)')
        print()


def load_dcm_from_path(dicom_path: str, filter_size: int = None, plot_hu=False,
                       force=False, check_img_orientation=False, verbose=False):
    """
    @param dicom_path: Path to load DICOM from.
    @param filter_size: Apply uniform filter if not None, return raw DICOM otherwise.
    @param plot_hu: Plot hounsfield dist if true
    @param force: force dicom reading
    @param check_img_orientation: Check for correct ImageOrientationPatient
    @param verbose: Print meta info if True

    @return: List of pydicom.dataset.FileDataset
    """

    if not os.path.exists(dicom_path):
        logging.error(f'Directory {dicom_path} does not exist. Load from SURFace instead?')
    ds = [dcmread(os.path.join(dicom_path, filename), force=force)
          for filename in glob.glob(os.path.join(dicom_path, '*.dcm'))]
    if len(ds) == 0:
        raise ValueError(f'{dicom_path} misses DICOM files.')

    if check_img_orientation:
        img_orient = np.array(ds[0].ImageOrientationPatient)
        if not np.array_equal(img_orient, [1, 0, 0, 0, 1, 0]):
            raise ValueError(f'Invalid ImageOrientationPatient: {img_orient}')

    if filter_size is not None:
        from scipy.ndimage import uniform_filter

        for i, img in enumerate(ds):
            # smooth3 in matlab pads the array by replicating -> nearest
            smoothed = uniform_filter(img.pixel_array, size=filter_size, mode='nearest')
            ds[i].PixelData = smoothed.tobytes()

    if verbose:
        print_dcm_meta(ds)
    if plot_hu:
        import plot_utils as plot_utils

        hu = get_pixels_hu(ds)
        plot_utils.plot_HU(hu)
    return ds


def get_image_voxel_data(ds) -> np.ndarray:
    """
    Get voxel data of image
    @param ds: DICOM images (List of pydicom.dataset.FileDataset)
    @return: Voxel matrix (Nx x Ny x Nz)
    """
    Nx, Ny, Nz = get_vol_size(ds)
    voxel_mat = np.zeros((Nx, Ny, Nz))
    for i in range(0, Nz):
        data = ds[i].pixel_array
        # Data indexing conventions https://github.com/mhe/pynrrd/issues/75
        # 3DSlicer and Matlab use Fortran-order [x, y, z, t],
        # but pydicom and numpy use C-order [t, z, y, x]
        # DICOM data comes from 3DSlicer, so swap axes
        voxel_mat[:, :, i] = np.swapaxes(data, 0, 1)
    return voxel_mat
