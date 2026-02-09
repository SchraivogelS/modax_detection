#!/usr/bin/env python3

"""
Functions for robust modiolar axis detection
Author: WW, NH, SCS
Date: 26.07.2021

###########################################################################
# Spiral Shape Recognition using Equiform Motions
# SNF Postdoc.Mobility Fellowship

# (C) 2018-2019 W. WIMMER
# HEARING RESEARCH LABORATORY, UNIVERSITY OF BERN (CH)
# EPIONE, INRIA SOPHIA ANTIPOLIS (FR)
###########################################################################
"""

import sys
import json
import os
import time
import logging

import numpy as np
import numpy.linalg as npl
import scipy.sparse.linalg as sla
import trimesh
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn import preprocessing
from typing import Tuple

import utils.matlab_util as mutil
import utils.pydicom_util as pydcm
import utils.plot_util as putil

# todo remove wildcard import
from modax.fit_ellipse import *
from utils.io_util import *
from utils.geometry_util import *


def base_dir() -> str:
    ret = 'data'
    return ret


def init_specimen(spec_dir, spec_name):
    specimen_data = load_landmarks(spec_dir, spec_name, read_all=True)
    loaded_ras = (specimen_data['coord_sys'] == 'RAS')

    for key, val in specimen_data['landmarks'].items():
        # 3DSlicer works with RAS internally but stores data with LPS anatomical coordinate
        # system on disk -> convert landmarks to LPS as well
        specimen_data['landmarks'][key] = np.array(pydcm.swap_ras_lps(val) if loaded_ras else val)

    specimen_data['coord_sys'] = 'LPS'
    return specimen_data


def create_bin_sphere(*, sphere_size, center, radius, show_sphere=False) -> np.ndarray:
    Nx, Ny, Nz = sphere_size
    Cx, Cy, Cz = center
    R = radius
    R_sq = R ** 2

    # 1. Calculate Bounding Box (BB) indices
    # Determine the smallest box that contains the sphere, clamped to volume limits
    i_start = max(0, int(Cx - R))
    i_end = min(Nx, int(Cx + R + 1))

    j_start = max(0, int(Cy - R))
    j_end = min(Ny, int(Cy + R + 1))

    k_start = max(0, int(Cz - R))
    k_end = min(Nz, int(Cz + R + 1))

    # 2. Generate Local Coordinates only for the BB
    # coords_local[0] is the I-axis array, coords_local[1] is J, coords_local[2] is K
    coords_local = np.ogrid[i_start:i_end, j_start:j_end, k_start:k_end]

    # 3. Calculate Local Squared Distance (Avoids slow np.sqrt and large float array)
    distance_sq_local = (coords_local[0] - Cx) ** 2 + \
                        (coords_local[1] - Cy) ** 2 + \
                        (coords_local[2] - Cz) ** 2

    # 4. Create Local Mask
    sphere_bool_local = (distance_sq_local <= R_sq)

    # 5. Embed Local Mask into the Global Mask
    # Create the final, full-sized boolean array (initialized to False)
    sphere_bool_global = np.full(sphere_size, False, dtype=bool)

    # Insert the computed small mask into the correct region
    sphere_bool_global[i_start:i_end, j_start:j_end, k_start:k_end] = sphere_bool_local

    if show_sphere:
        import scipy.ndimage as ndi

        plt.figure(figsize=(16, 10))
        ax = plt.axes(projection='3d')

        # Erode the sphere (peel one layer off)
        eroded_sphere = ndi.binary_erosion(sphere_bool_global)
        # The surface is the original sphere MINUS the eroded version
        surface_mask = sphere_bool_global & ~eroded_sphere

        # Only plot surface points
        pos = np.where(surface_mask == True)
        ax.scatter(pos[0], pos[1], pos[2], color='k', alpha=0.4, s=1)

        ax.scatter(center[0], center[1], center[2], color='r', s=50)
        ax.text(center[0], center[1], center[2], "C", color='r',
                horizontalalignment='center', verticalalignment='center',
                fontsize=30)

        ax.set_xlim3d(center[0] - R, center[0] + R)
        ax.set_ylim3d(center[1] - R, center[1] + R)
        ax.set_zlim3d(center[2] - R, center[2] + R)
        plt.show(block=True)

    return sphere_bool_global


def smooth_laplacian(verts, faces):
    mesh_smooth = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh_smooth = trimesh.smoothing.filter_mut_dif_laplacian(mesh_smooth)
    verts = mesh_smooth.vertices.view(np.ndarray)
    faces = mesh_smooth.faces.view(np.ndarray)
    return verts, faces


def skimage_marching_cubes(voxel_mat: np.ndarray, iso_level: int, *,
                           mask: np.ndarray = None, spacing: tuple = None, smooth: bool = False,
                           verbose: bool = True):
    from skimage.measure import marching_cubes

    verts, faces, _, _ = marching_cubes(voxel_mat, level=iso_level, method='lewiner', mask=mask, spacing=spacing)
    if verts.size == 0:
        if verbose:
            print("[x] Marching Cubes successful but returned zero vertices (no surface found).")
        return None, None
    elif verbose:
        print(f"Marching Cubes finished. Vertices: {verts.shape}, Faces: {faces.shape}")

    if smooth:
        verts, faces = smooth_laplacian(verts, faces)
    return verts, faces


def _marching_cubes(voxel_mat: np.ndarray, iso_th: int, *, spec_dir: str, spec_name: str,
                    mask_cochlea: bool, spacing: np.ndarray, offset: np.ndarray, vol_size: np.ndarray,
                    smooth: bool = False, show_mesh: bool = False, show_sphere: bool = False, verbose: bool = False):
    # check spacing
    th_thickness_mm = 0.8
    thickness_mm = spacing[2]
    if thickness_mm >= th_thickness_mm:
        raise ValueError(f'Slice thickness too high for accurate surface extraction '
                         f'({thickness_mm:.2f} mm >= {th_thickness_mm:.2f})')

    # mask cochlea to speed up marching cubes
    cochlear_sphere, center_vox = get_cochlear_sphere(mask_cochlea=mask_cochlea, spec_dir=spec_dir, spec_name=spec_name,
                                                      dcm_offset=offset, dcm_spacing=spacing, vol_size=vol_size,
                                                      verbose=verbose)
    if show_sphere:
        show_cochlear_sphere(voxel_mat=voxel_mat, center_vox=center_vox, sphere=cochlear_sphere)

    if verbose:
        print('Marching cubes to create cochlear mesh ...')
    time_start = time.time()

    # skimage mesh
    verts, faces = skimage_marching_cubes(voxel_mat=voxel_mat, iso_level=iso_th, mask=cochlear_sphere,
                                          spacing=tuple(spacing), smooth=smooth)
    verts_transformed = verts + offset  # verts already scaled correctly by skimage using spacing

    if len(verts_transformed) == 0:
        raise ValueError(f'No surface at given iso level')

    if verbose:
        print('Marching cubes took {:.2f}s\n'.format(time.time() - time_start))
        print(f'DICOM offset {offset}')
        print(f'Spacing {spacing}')

    if show_mesh:
        putil.iplot_mesh(verts, faces, title=f'Raw marching cubes from Python, iso={iso_th}')

    return verts, faces, verts_transformed


def update_mesh_data(dict_in, verts, faces):
    dict_in['vertices'] = verts
    dict_in['faces'] = faces


def lps_to_ijk_matrix(origin: np.array, spacing: np.array) -> np.array:
    """
    Affine spatial transformation matrix that maps from LPS anatomical coordinate system to ijk voxel coordinates.
    Transformation matrix is in LPI convention.

    @param origin: ImagePositionPatient
    @param spacing: PixelSpacing
    @return: LPS to IJK matrix (4x4)
    """
    ox, oy, oz = origin
    sx, sy, sz = spacing
    ret = np.array([
        [-1 / spacing[0], 0, 0, -ox / sx],
        [0, -1 / sy, 0, -oy / sy],
        [0, 0, 1 / sz, - oz / sz],
        [0, 0, 0, 1]
    ])
    return ret


def load_landmarks(spec_dir, spec_name, read_all=False):
    lm_file_path = os.path.abspath(
        os.path.join(spec_dir, '_desc', str(spec_name + '-landmarks-ras.json')))
    landmarks_ras = load_values_from_json(lm_file_path)
    if not read_all:
        landmarks_ras = {key: np.array(value) for key, value in landmarks_ras['landmarks'].items()}
    return landmarks_ras


def create_cochlear_sphere(spec_dir, spec_name, origin, spacing, vol_size):
    landmarks_ras = load_landmarks(spec_dir, spec_name)
    # fixme: why not working in lps?
    a_lps = landmarks_ras['A']  # pydcm.swap_ras_lps(landmarks_ras['A'])
    c_lps = landmarks_ras['C']  # pydcm.swap_ras_lps(landmarks_ras['C'])

    lps_to_ijk = lps_to_ijk_matrix(origin, spacing)
    a_ijk = np.round(transform_3D_point(a_lps, lps_to_ijk), 0)
    c_ijk = np.round(transform_3D_point(c_lps, lps_to_ijk), 0)

    sphere_center_ijk = (a_ijk + c_ijk) / 2.0

    # compute radius in voxels (pick the most conservative axis)
    radius_mm = 8.0
    sx, sy, sz = spacing
    r_vox = np.array([radius_mm / sx, radius_mm / sy, radius_mm / sz])
    sphere_radius = r_vox.min()

    cochlear_sphere = create_bin_sphere(sphere_size=(vol_size[0], vol_size[1], vol_size[2]),
                                        center=sphere_center_ijk, radius=sphere_radius)

    return cochlear_sphere, sphere_center_ijk


def ras_to_voxel(ras, origin_lps, spacing):
    """
    Convert a point in RAS (mm) into voxel indices (z,y,x) for a DICOM stack.

    Parameters
    ----------
    ras : array‐like, shape (3,)
        Point in RAS patient coords (Right, Anterior, Superior) in mm.
    origin_lps: array-like shape (3,)
    spacing : array‐like, shape (3,)
        Voxel spacing in (sz, sy, sx) mm.

    Returns
    -------
    ijk : np.ndarray, shape (3,)
        Voxel indices (z, y, x) as floats.
    """
    # 1) Convert RAS → LPS (DICOM uses LPS: Left, Posterior, Superior)
    ras = np.asarray(ras, dtype=float)
    lps = np.array([-ras[0], -ras[1], ras[2]], dtype=float)

    # 3) Simple axis‐aligned case: indices = (LPS − origin) / spacing
    ijk = (lps - origin_lps) / spacing

    # Note: the returned order is [z_idx, y_idx, x_idx] if spacing is [sz, sy, sx]
    return ijk


def transform_verts(verts_in, offset, spacing):
    verts = deepcopy(verts_in)
    scaled_x = (verts[:, 0] - 1) * spacing[0]
    scaled_y = (verts[:, 1] - 1) * spacing[1]
    scaled_z = (verts[:, 2] - 1) * spacing[2]
    ret = np.stack((scaled_x, scaled_y, scaled_z), axis=1)
    ret += offset
    return ret


def get_dicom_dir(spec_dir):
    ret = os.path.join(spec_dir, 'data', 'preop')
    return ret


def check_for_low_n_dicoms(dir_dicom: str, lim_n_dicom: int):
    n_files = len([name for name in os.listdir(dir_dicom) if name.endswith('.dcm')])
    if n_files <= lim_n_dicom:
        logging.warning(f'only {n_files} dicoms in {dir_dicom}')


def show_cochlear_sphere(*, voxel_mat: np.ndarray, center_vox: np.ndarray,
                         sphere: np.ndarray, block: bool = True):
    """
    Displays the 3 orthogonal planar views (Axial, Coronal, Sagittal)
    of the voxel data, centered on the sphere's center, with the
    sphere mask overlaid.

    Assumes voxel_mat and sphere are indexed as (X, Y, Z).
    """

    # 1. Check for valid voxel data
    if voxel_mat is None:
        print("[x] Warning: voxel_mat is None. Cannot plot sphere in relation to volume.")
        return

    # 2. Get integer center coordinates in (X, Y, Z) order
    x0, y0, z0 = np.round(center_vox).astype(int)

    # 3. Safety check: Clamp indices to be within the volume bounds
    Nx, Ny, Nz = voxel_mat.shape
    x0 = np.clip(x0, 0, Nx - 1)
    y0 = np.clip(y0, 0, Ny - 1)
    z0 = np.clip(z0, 0, Nz - 1)

    # 4. Extract the 2D slices
    # Axial (XY) plane at z = z0
    axial_slice_img = voxel_mat[:, :, z0]
    axial_slice_mask = sphere[:, :, z0]

    # Coronal (XZ) plane at y = y0
    coronal_slice_img = voxel_mat[:, y0, :]
    coronal_slice_mask = sphere[:, y0, :]

    # Sagittal (YZ) plane at x = x0
    sagittal_slice_img = voxel_mat[x0, :, :]
    sagittal_slice_mask = sphere[x0, :, :]

    # 5. Plot the three views
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # --- Axial Plot ---
    # We transpose (.T) so that X is horizontal and Y is vertical
    axes[0].imshow(axial_slice_img.T, cmap='gray', origin='lower')
    # Overlay the mask with a color map and transparency
    axes[0].imshow(axial_slice_mask.T, cmap='Reds', alpha=0.3, origin='lower')
    axes[0].set_title(f'Axial (XY) at Z={z0}')
    axes[0].set_xlabel('X-axis')
    axes[0].set_ylabel('Y-axis')
    axes[0].axis('image')  # Ensure correct aspect ratio

    # --- Coronal Plot ---
    # Transpose so X is horizontal and Z is vertical
    axes[1].imshow(coronal_slice_img.T, cmap='gray', origin='lower')
    axes[1].imshow(coronal_slice_mask.T, cmap='Reds', alpha=0.3, origin='lower')
    axes[1].set_title(f'Coronal (XZ) at Y={y0}')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Z-axis')
    axes[1].axis('image')

    # --- Sagittal Plot ---
    # Transpose so Y is horizontal and Z is vertical
    axes[2].imshow(sagittal_slice_img.T, cmap='gray', origin='lower')
    axes[2].imshow(sagittal_slice_mask.T, cmap='Reds', alpha=0.3, origin='lower')
    axes[2].set_title(f'Sagittal (YZ) at X={x0}')
    axes[2].set_xlabel('Y-axis')
    axes[2].set_ylabel('Z-axis')
    axes[2].axis('image')

    plt.tight_layout()
    plt.show(block=block)


def plot_voxel_distribution(voxel_mat: np.ndarray, iso_th: int, title: str):
    """Plots a histogram of voxel intensities and marks the current isothreshold."""

    # Flatten the 3D array into a 1D list of all voxel values
    voxel_values = voxel_mat.flatten()

    plt.figure(figsize=(10, 6))

    # Plot the histogram (use a logarithmic y-axis for better visibility of rare values)
    # Use bins that cover the relevant range (e.g., -1024 to 3000 for CT)

    plt.hist(voxel_values, bins=200, log=True, color='gray', edgecolor='black', range=[-1024, 3000])

    # Add a vertical line for the current ISO threshold
    plt.axvline(x=iso_th, color='red', linestyle='--', linewidth=2, label=f'Current ISO Threshold ({iso_th} HU)')

    # Highlight the relevant range for the histogram (e.g., -500 to 2000 HU)
    # Adjust xlim to focus on the area around the expected bone values
    plt.xlim(-500, 2000)

    print(f"Min: {np.min(voxel_values)}, Max: {np.max(voxel_values)}, Mean: {np.mean(voxel_values):.2f}, "
          f"Std Dev: {np.std(voxel_values):.2f}, Unique Values: {len(np.unique(voxel_values))}")
    print()

    plt.title(title)
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Voxel Count (Log Scale)")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show(block=True)


def _resample_to_standard_orientation(*, spec_dir: str, spec_name: str, iso_th: int = None,
                                      verbose: bool = False):
    import SimpleITK as sitk

    print("[x] WARNING: Dicom not in standard axial view")
    recon_nii_path = os.path.join(spec_dir, 'data', 'preop_recon', f'{spec_name}_resampled.nii')
    # Check if the reconstructed file exists
    if os.path.exists(recon_nii_path):
        print(f"> Loading pre-reconstructed volume from: {recon_nii_path}")
        fixed_volume = sitk.ReadImage(recon_nii_path)
    else:
        print("> Reconstructing...")
        import utils.vtk_util as vtk_util

        # Load DICOM series into a SimpleITK image
        dicom_dir = get_dicom_dir(spec_dir)
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
        if not series_ids:
            raise RuntimeError(f"No DICOM series found in: {dicom_dir}")

        # Choose a series (here: first one)
        series_id = series_ids[0]

        # Get sorted file names for that series
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_id)

        # Read the volume
        reader = sitk.ImageSeriesReader()
        reader.SetMetaDataDictionaryArrayUpdate(True)  # propagate per-slice tags
        reader.SetFileNames(dicom_names)
        sitk_volume = reader.Execute()

        # Resample (Fix Orientation)
        start_time = time.time()
        fixed_volume = vtk_util.resample_to_standard_orientation(sitk_volume)
        if verbose:
            print('Resampling took {:.2f}s\n'.format(time.time() - start_time))

        # Save the image in NIfTI format
        output_dir = os.path.join(spec_dir, 'data', 'preop_recon')
        output_filename = os.path.join(output_dir, f'{spec_name}_resampled.nii')
        # do not overwrite
        if not os.path.exists(output_filename):
            os.makedirs(output_dir, exist_ok=True)
            try:
                sitk.WriteImage(fixed_volume, output_filename)
                print(f"Resampled image successfully saved to: {output_filename}")
            except Exception as e:
                print(f"Error saving resampled image: {e}")

    # 3. Get image properties
    dcm_offset = np.array(fixed_volume.GetOrigin())
    Nx, Ny, Nz = fixed_volume.GetSize()
    vol_size = np.array([Nx, Ny, Nz])
    dcm_spacing = np.array(fixed_volume.GetSpacing())

    # 4. Convert back to NumPy array for your existing pipeline
    voxel_mat_sitk = sitk.GetArrayFromImage(fixed_volume)
    # Update Voxel Data in SITK (Z, Y, X) order to (X, Y, Z) order
    voxel_mat = np.swapaxes(voxel_mat_sitk, 0, 2)
    # Check the shape of the SITK output
    if voxel_mat.shape[0] != vol_size[0] or voxel_mat.shape[2] != vol_size[2]:
        raise ValueError("[x] WARNING: SHAPE MISMATCH AFTER SWAP. Review axis definition.")

    if verbose:
        print(f'Resampled Image: New Size: {vol_size}, New Spacing: {dcm_spacing}')
        plot_voxel_distribution(voxel_mat, iso_th, "Voxel Intensity Distribution Post-Resampling")
    return voxel_mat, dcm_offset, dcm_spacing, vol_size


def load_dicom(spec_dir: str, *, filter_size: int, verbose: bool = False):
    dicom_dir = get_dicom_dir(spec_dir)
    print(f'Load DICOM from {dicom_dir}')
    check_for_low_n_dicoms(dicom_dir, 50)

    # load DICOM files (in LPS if saved with 3DSlicer)
    start_time = time.time()
    ret = pydcm.load_DICOM_from_path(dicom_dir, filter_size=filter_size, verbose=True)

    if verbose:
        print('Dicom loading took {:.2f}s\n'.format(time.time() - start_time))
    return ret


def get_cochlear_sphere(*, mask_cochlea: bool, spec_dir: str, spec_name: str, dcm_offset: np.ndarray,
                        dcm_spacing: np.ndarray, vol_size: np.ndarray, verbose: bool = False):
    if mask_cochlea:
        cochlear_sphere, center_vox = create_cochlear_sphere(spec_dir, spec_name, dcm_offset,
                                                             dcm_spacing, vol_size)
        if verbose:
            print('Pre-cropping applied to voxel data.')
    else:
        cochlear_sphere = None
        center_vox = None
    return cochlear_sphere, center_vox


def mesh_from_dicom(spec_dir, spec_name, iso_th: int = 1000, filter_size: int = 3,
                    mask_cochlea=False, show_mesh=False, return_raw_verts=False,
                    verbose: bool = False):
    patient_ds = load_dicom(spec_dir, filter_size=filter_size, verbose=verbose)

    # check image orientation
    img_or = np.array(patient_ds[0].ImageOrientationPatient, dtype=float)
    is_standard_axial = np.allclose(img_or, np.array([1, 0, 0, 0, 1, 0]))
    if not is_standard_axial:
        voxel_mat, dcm_offset, dcm_spacing, vol_size = _resample_to_standard_orientation(spec_dir=spec_dir,
                                                                                         spec_name=spec_name,
                                                                                         iso_th=iso_th,
                                                                                         verbose=verbose)
    else:
        # standard orientation
        dcm_offset = pydcm.get_DICOM_offset(patient_ds[0])
        Nx, Ny, Nz = pydcm.get_vol_size(patient_ds)
        vol_size = np.array([Nx, Ny, Nz])
        dcm_spacing = pydcm.get_pixel_spacing(patient_ds)
        start_time = time.time()
        voxel_mat = pydcm.get_image_voxel_data(patient_ds)
        if verbose:
            print('Voxel data to RAM took {:.2f}s\n'.format(time.time() - start_time))

    # Extract cochlear surface
    verts, faces, verts_transformed = _marching_cubes(voxel_mat, iso_th, mask_cochlea=mask_cochlea, spec_dir=spec_dir,
                                                      spec_name=spec_name, offset=dcm_offset, spacing=dcm_spacing,
                                                      vol_size=vol_size, show_mesh=show_mesh, verbose=verbose)

    if return_raw_verts:
        return verts, verts_transformed, faces, dcm_offset, dcm_spacing

    return verts_transformed, faces


def mesh_from_surf(spec_data_dir, spec_name, show_mesh=False):
    # Define Path to Surface Data of selected Specimen
    surf_dir = os.path.abspath(os.path.join(spec_data_dir, 'uCT', 'SURF'))
    print(f'Load surface from {surf_dir}')

    if not os.path.exists(surf_dir):
        logging.error(f'Directory {surf_dir} does not exist. Load from DICOM instead?')
        sys.exit(1)
    surf_file_path = os.path.abspath(
        os.path.join(surf_dir, spec_name + '_uCT_SURF.json'))

    with open(surf_file_path, mode='r', encoding='utf8') as g:
        vertices_faces = json.load(g)

    vertices_array = np.array(vertices_faces['LABYRINTH']['vertices']).copy()
    faces_array = np.array(vertices_faces['LABYRINTH']['faces']).copy()

    # correct face indices, as Matlab starts from 1
    faces_array = faces_array - [[1, 1, 1]]

    if show_mesh:
        mesh = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)
        mesh.show()

    return vertices_array, faces_array


def get_verts_to_rm_dicom(verts_loc, landmarks_loc, crop_radius):
    n_rows_verts = verts_loc.shape[0]

    # Boundary sphere
    sphere_center = (landmarks_loc['A'] + landmarks_loc['C']) / 2
    sphere_radius = np.linalg.norm(landmarks_loc['A'] - landmarks_loc['C'])

    boundary_sphere = np.linalg.norm(verts_loc - sphere_center, axis=1) > (
            sphere_radius * crop_radius)
    # Boundary plane apex
    plane_apex = (landmarks_loc['C'] - landmarks_loc['A'])
    plane_apex /= np.linalg.norm(plane_apex)
    plane_apex_rep = np.array([plane_apex] * n_rows_verts)
    boundary_plane_apex = mutil.matlab_dot(plane_apex_rep, verts_loc, axis=1) < np.dot(plane_apex,
                                                                                       landmarks_loc[
                                                                                           'A'] + 0.5)
    # Boundary plane base
    plane_base = landmarks_loc['A'] - landmarks_loc['C']
    plane_base /= np.linalg.norm(plane_base)
    plane_base_rep = np.array([plane_base] * n_rows_verts)
    boundary_plane_base = mutil.matlab_dot(plane_base_rep, verts_loc, axis=1) < np.dot(plane_base,
                                                                                       landmarks_loc[
                                                                                           'C'])

    # Remove vertices outside boundaries
    verts_rm_row = np.array(np.logical_or(np.logical_or(boundary_sphere, boundary_plane_base),
                                          boundary_plane_apex))
    return verts_rm_row


def get_verts_to_rm_surf(verts_loc, landmarks_loc):
    n_rows_verts = verts_loc.shape[0]
    boundary_sphere = np.linalg.norm(verts_loc - landmarks_loc['V'], axis=1) < np.linalg.norm(
        landmarks_loc['RW'] - landmarks_loc['V'])
    plane_normal = (landmarks_loc['C'] - landmarks_loc['V'])
    plane_normal /= np.linalg.norm(plane_normal)
    plane_normal_rep = np.array([plane_normal] * n_rows_verts)
    boundary_plane_base = mutil.matlab_dot(plane_normal_rep, verts_loc, axis=1) < np.dot(
        plane_normal,
        landmarks_loc['RW'])
    # Remove vertices outside boundaries
    verts_rm_row = np.array(np.logical_or(boundary_sphere, boundary_plane_base))
    return verts_rm_row


def build_nx_graph(mesh: trimesh.Trimesh) -> nx.Graph:
    """
    Builds a NetworkX graph from a Trimesh object using edge lengths as weights.

    The graph nodes correspond to the mesh vertices, and edge weights are
    set to the geometric length of the mesh edges.

    :param mesh: The input Trimesh object.
    :type mesh: trimesh.Trimesh
    :return: A NetworkX graph where edges have a 'weight' attribute equal to their length.
    :rtype: networkx.Graph
    """
    g = nx.Graph()
    edges = mesh.edges_unique
    length = mesh.edges_unique_length
    for edge, L in zip(edges, length):
        # We use a sorted tuple for the edge to ensure consistency,
        # though nx.Graph handles undirected edges well.
        g.add_edge(edge[0], edge[1], weight=L)
    return g


def split_connected_component_by_bridge(mesh_cochlea_and_junk):
    """
    Identifies and removes a 'tiny bridge' connecting two large components
    within a single mesh component using NetworkX shortest path analysis.

    This function finds the shortest path between the mesh's two most extreme
    vertices (assumed to lie on the separate components). It removes all faces
    incident to this path (the guaranteed cut). Finally, it returns the largest
    resulting connected component (assumed to be the cochlea).

    :param mesh_cochlea_and_junk: A single Trimesh component containing the target
                                  structure (cochlea) and an unwanted connected
                                  piece of tissue.
    :type mesh_cochlea_and_junk: trimesh.Trimesh
    :return: A tuple containing the vertices and faces of the largest resulting
             connected component after the bridge cut. Returns the original
             component's data on failure.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    vertices = mesh_cochlea_and_junk.vertices
    axes = [0, 1, 2]  # X, Y, Z axes indices

    # Loop through all three primary axes
    for axis in axes:
        # 1. Setup Graph (G) remains the same
        G = build_nx_graph(mesh_cochlea_and_junk)

        # 2. Find Extremes along the current axis
        v_start_idx = np.argmin(vertices[:, axis])
        v_end_idx = np.argmax(vertices[:, axis])

        # Check if extremes are too close (optional but good practice)
        if np.linalg.norm(vertices[v_start_idx] - vertices[v_end_idx]) < mesh_cochlea_and_junk.scale / 4:
            continue  # Skip this axis if the extrema are too close

        # 2. Compute Shortest Path
        try:
            # Use Dijkstra's algorithm (default for weight='weight')
            path_v = nx.shortest_path(G, source=v_start_idx, target=v_end_idx, weight="weight")
        except nx.NetworkXNoPath:
            # Should not happen if the mesh is truly connected, but is a safe guard
            print("NetworkX could not find a path. Skipping bridge cut.")
            return mesh_cochlea_and_junk.vertices, mesh_cochlea_and_junk.faces

        if len(path_v) < 2:
            return mesh_cochlea_and_junk.vertices, mesh_cochlea_and_junk.faces

        # 3. Identify Faces on the Path (Bridge)
        unique_faces_on_path = set()
        face_verts_array = mesh_cochlea_and_junk.faces

        # Iterate over all faces and check for edges from the path
        for i in range(len(face_verts_array)):
            face_verts = face_verts_array[i]

            # Check if any two consecutive path vertices belong to this face
            for j in range(len(path_v) - 1):
                v1, v2 = path_v[j], path_v[j + 1]

                # Using basic containment check
                if v1 in face_verts and v2 in face_verts:
                    unique_faces_on_path.add(i)
                    break

        path_face_indices = np.array(list(unique_faces_on_path), dtype=int)

        if len(path_face_indices) == 0:
            return mesh_cochlea_and_junk.vertices, mesh_cochlea_and_junk.faces

        # 5. Guaranteed Cut and Re-Split Test
        faces_to_remove = path_face_indices

        if len(faces_to_remove) > 0:
            mesh_cut = mesh_cochlea_and_junk.copy()
            # ... perform cut (update_faces) ...
            mesh_cut.process(validate=True)

            sub_cc_cut = trimesh.graph.connected_components(mesh_cut.face_adjacency, min_len=1)

            if len(sub_cc_cut) > 1:
                # SUCCESS! Select the largest component and RETURN
                component_sizes = [len(cc) for cc in sub_cc_cut]
                largest_component_index = np.argmax(component_sizes)
                final_faces_index = sub_cc_cut[largest_component_index]

                mesh_cochlea = mesh_cut.submesh([final_faces_index], append=True)
                return mesh_cochlea.vertices, mesh_cochlea.faces

    # Fallback: if no cut worked on any axis, return the original component
    return mesh_cochlea_and_junk.vertices, mesh_cochlea_and_junk.faces


def select_cochlea_from_fragments(verts_fragments: np.ndarray, faces_fragments: np.ndarray,
                                  min_len: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects the cochlea as the main connected component from a fragmented mesh,
    applying a bridge-cutting heuristic if the initial component contains
    unwanted, weakly connected tissue.

    The function first finds the largest connected component above a size
    threshold (min_len). If this component is assumed to contain a bridge,
    it calls :py:func:`split_connected_component_by_bridge` to isolate the
    cochlea.

    :param verts_fragments: Vertices of the fragmented surface.
    :type verts_fragments: np.ndarray
    :param faces_fragments: Faces of the fragmented surface.
    :type faces_fragments: np.ndarray
    :param min_len: Minimum number of faces for a component to be considered.
                    Defaults to 1000 and is halved iteratively if no component
                    is found.
    :type min_len: int
    :return: A tuple containing the vertices and faces of the isolated cochlea.
             Returns empty arrays if no component is found.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # This is the main routine that calls the NetworkX bridge-cutting logic
    mesh_fragments = trimesh.Trimesh(vertices=verts_fragments,
                                     faces=faces_fragments, process=True, validate=True)

    # 1. Initial Connected Component Selection
    cc = trimesh.graph.connected_components(mesh_fragments.face_adjacency, min_len=min_len)

    current_min_len = min_len
    while (len(cc) == 0) and (current_min_len > 400):
        current_min_len = int(current_min_len / 2)
        cc = trimesh.graph.connected_components(mesh_fragments.face_adjacency, min_len=current_min_len)

    if len(cc) == 0:
        return np.array([]), np.array([])

    # Extract the main component (cochlea + junk)
    main_component_faces_index = cc[0]
    mesh_cochlea_and_junk = mesh_fragments.submesh([main_component_faces_index], append=True)
    mesh_cochlea_and_junk.process(validate=True)

    # 2. Attempt to split the single connected component
    verts_result, faces_result = split_connected_component_by_bridge(mesh_cochlea_and_junk)

    if verts_result is None or len(verts_result) == 0:
        # Fallback to the original large component if the split failed
        return mesh_cochlea_and_junk.vertices, mesh_cochlea_and_junk.faces

    return verts_result, faces_result


def select_cochlea_from_fragments_basic(verts_fragments, faces_fragments, min_len=1000):
    mesh_cochlea = trimesh.Trimesh(vertices=verts_fragments,
                                   faces=faces_fragments, process=True, validate=True)

    cc = trimesh.graph.connected_components(mesh_cochlea.face_adjacency, min_len=min_len)
    while (len(cc) == 0) and (min_len > 400):
        min_len = int(min_len / 2)
        cc = trimesh.graph.connected_components(mesh_cochlea.face_adjacency, min_len=min_len)

    if len(cc) > 0:
        mask = np.zeros(len(mesh_cochlea.faces), dtype=bool)
        mask[np.concatenate(cc)] = True
        mesh_cochlea.update_faces(mask)

    verts_cochlea_loc = mesh_cochlea.vertices
    faces_cochlea_loc = mesh_cochlea.faces

    return verts_cochlea_loc, faces_cochlea_loc


def remove_vertices_by_mask(verts: np.ndarray, faces: np.ndarray,
                            bool_rm_verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes vertices from a mesh defined by a boolean mask, cleans up faces,
    and recursively removes any resulting unreferenced vertices.

    Args:
        verts: The current array of vertex coordinates (N, 3).
        faces: The current array of face indices (M, 3).
        bool_rm_verts: Boolean mask (N,) where True indicates a vertex to be REMOVED.

    Returns:
        ret_verts, ret_faces: The cleaned mesh.
    """
    # 1. Uniquify Vertices (Necessary for consistent indexing)
    v_unique, idx_inv = np.unique(verts, axis=0, return_inverse=True)
    f = idx_inv[faces]  # faces re-indexed to v_unique
    v = v_unique

    # 1. Find the coordinates of vertices to remove.
    verts_to_rm_coords = verts[bool_rm_verts, :]
    # 2. Find the index of those coordinates in the unique list 'v'.
    _, idx_rm_verts = mutil.matlab_ismember(verts_to_rm_coords, v, rows=True)

    # Create the boolean mask for the unique vertex array 'v'
    logical_rm_unique = np.full(v.shape[0], False)
    logical_rm_unique[idx_rm_verts] = True

    # 2. Renumber Vertices
    tags_old = np.arange(0, len(v))
    logical_keep_verts = np.invert(logical_rm_unique)

    new_count = np.cumsum(logical_keep_verts)
    tags_new = tags_old.copy()
    tags_new[logical_keep_verts] = new_count[logical_keep_verts]
    tags_new -= 1  # index starts from 0

    # Use masked array to prevent invalid index lookups later
    tags_new_ma = np.ma.masked_array(tags_new, mask=logical_rm_unique)

    # 3. Delete Vertices
    v_new = v[logical_keep_verts, :]

    # 4. Delete Faces
    # Logical mask: True if ALL THREE vertices of a face are in the list of vertices to remove (idx_rm_verts).
    # delete faces that reference *any* vertex that was removed.

    # Find position (rows) of faces to delete (face contains at least one vertex to be removed)
    face_vertex_is_removed = logical_rm_unique[f]  # (M, 3) boolean array
    logical_rm_faces_by_vertex = face_vertex_is_removed.any(1)

    f_new = f[~logical_rm_faces_by_vertex]  # Keep faces that DO NOT reference a removed vertex

    # 5. Renumber Faces
    # Apply the new tags to the remaining faces
    f_new_renumbered = tags_new_ma[f_new]

    # Clean up the renumbered faces (this should be empty if the masking was perfect,
    # but serves as a final safety check)
    f_new_renumbered = f_new_renumbered.data

    # 6. Recursive Clean-up (Remove unreferenced vertices)
    ret_verts = v_new
    ret_faces = f_new_renumbered

    # Find vertices in v_new that are not referenced in f_new_renumbered
    tags_new = np.arange(0, len(v_new))
    # Check which vertices in the new list (0 to len(v_new)) are present in the new faces

    referenced_verts_mask = mutil.matlab_ismember(tags_new, f_new_renumbered)
    unused_verts_mask = np.invert(referenced_verts_mask)

    # Recursive call to clean up isolated fragments
    if np.any(unused_verts_mask):
        ret_verts, ret_faces = remove_vertices_by_mask(ret_verts, ret_faces,
                                                       unused_verts_mask)

    return ret_verts, ret_faces


def side_to_str(side: int) -> str:
    if not isinstance(side, int):
        raise ValueError(f'Side has wrong type ({type(side)})')
    if side == 1:
        ret = 'RIGHT'
    else:
        ret = 'LEFT'
    return ret


def get_local_coord_sys(specimen_data):
    # Determine side of specimen (right, left) and construct orthonormal coordination system
    temp_x = specimen_data['landmarks']['RW'] - specimen_data['landmarks']['C']
    temp_z = specimen_data['landmarks']['A'] - specimen_data['landmarks']['C']
    temp_y = np.cross(temp_z, temp_x)

    # left cochlea in left-handed, right cochlea in right-handed coordinate system
    temp_side_y = specimen_data['landmarks']['RW'] - specimen_data['landmarks']['OW']
    side = int(np.sign(np.dot(temp_y, temp_side_y)))  # see side_to_str()
    temp_y = np.cross(side * temp_z, temp_x)

    coord_sys = dict()
    coord_sys['Origin'] = specimen_data['landmarks']['C'].copy()
    coord_sys['Xaxis'] = temp_x / npl.norm(temp_x)
    coord_sys['Yaxis'] = temp_y / npl.norm(temp_y)
    coord_sys['Zaxis'] = side * np.cross(coord_sys['Xaxis'], coord_sys['Yaxis'])

    # Transformation
    translation = coord_sys['Origin']
    rotation_matrix = np.vstack([coord_sys['Xaxis'], coord_sys['Yaxis'], coord_sys['Zaxis']])
    # ht_matrix = np.r_[np.c_[rotation_matrix, translation], np.array([[0, 0, 0, 1]])]
    return translation, rotation_matrix, coord_sys, side


def transform_points(points, translation, rot_mat):
    p_trans = points.copy()
    p_trans -= translation
    rot_mat_inv = npl.inv(rot_mat)  # equals the transposed matrix if orthonormal matrix
    ret = p_trans @ rot_mat_inv
    return ret


def transform_labyrinth(specimen_data, translation, rot_mat):
    labyrinth_loc = defaultdict(dict)

    # Do not alter faces
    labyrinth_loc['faces'] = specimen_data['faces'].copy()

    # Translation and rotation of vertices and landmarks
    labyrinth_loc['vertices'] = transform_points(specimen_data['vertices'], translation, rot_mat)
    landmarks = specimen_data['landmarks'].copy()
    for key, value in landmarks.items():
        labyrinth_loc['landmarks'][key] = transform_points(landmarks[key], translation, rot_mat)

    return labyrinth_loc


def normalize_mesh(verts, faces, return_mesh=False, show_mesh=False):
    # Trimesh will do a light processing, which will
    # remove any NaN values and merge vertices that share position
    verts_normalized, centralizer, normalizer = my_normalizer(verts)
    mesh_normalized = trimesh.Trimesh(vertices=verts_normalized,
                                      faces=faces, process=True, validate=True)
    if show_mesh:
        mesh_normalized.show()

    verts_normalized = mesh_normalized.vertices.view(np.ndarray)
    faces_normalized = mesh_normalized.faces.view(np.ndarray)
    if return_mesh:
        return verts_normalized, faces_normalized, centralizer, normalizer, mesh_normalized
    else:
        return verts_normalized, faces_normalized, centralizer, normalizer


def parameter_space(verts: np.array, normals: np.array, extended: bool):
    space = dict()
    max_verts = len(verts)

    space['ze'] = np.zeros((max_verts, 1))
    space['on'] = np.ones((max_verts, 1))
    space['vx'] = ascol(verts[:, 0])
    space['vy'] = ascol(verts[:, 1])
    space['vz'] = ascol(verts[:, 2])
    space['nx'] = ascol(normals[:, 0])
    space['ny'] = ascol(normals[:, 1])
    space['nz'] = ascol(normals[:, 2])

    if extended:
        space.pop('on', None)

    return space


def init_standard_spiral_field(verts, normals) -> dict:
    gamma_np = dict()

    # Parameter space - standard spiral field
    space = parameter_space(verts, normals, extended=False)
    on = space['on']
    ze = space['ze']
    vx = space['vx']
    vy = space['vy']
    vz = space['vz']
    nx = space['nx']
    ny = space['ny']
    nz = space['nz']

    # Gradient part
    gamma_np['ngrad_x'] = np.concatenate((ze, vz, -vy, on, ze, ze, vx), axis=1)
    gamma_np['ngrad_y'] = np.concatenate((-vz, ze, vx, ze, on, ze, vy), axis=1)
    gamma_np['ngrad_z'] = np.concatenate((vy, -vx, ze, ze, ze, on, vz), axis=1)
    # Position part (if w_p > 0)
    gamma_np['pgrad_x'] = np.concatenate((ze, -nz, ny, ze, ze, ze, nx), axis=1)
    gamma_np['pgrad_y'] = np.concatenate((nz, ze, -nx, ze, ze, ze, ny), axis=1)
    gamma_np['pgrad_z'] = np.concatenate((-ny, nx, ze, ze, ze, ze, nz), axis=1)

    return gamma_np


def init_extended_spiral_field(verts: np.array, normals: np.array, gamma_np: dict) -> dict:
    gamma_np_ext = dict()

    # Parameter space - extended spiral field
    space_ext = parameter_space(verts, normals, extended=True)

    ze = space_ext['ze']
    vx = space_ext['vx']
    vy = space_ext['vy']
    vz = space_ext['vz']
    nx = space_ext['nx']
    ny = space_ext['ny']
    nz = space_ext['nz']
    n_dot_p = ascol(mutil.matlab_dot(verts, normals, axis=1))

    f_x = np.concatenate((-ascol((npl.norm(verts, axis=1)) ** 2), ze, ze), axis=1) + (verts * vx)
    f_y = np.concatenate((ze, -ascol((npl.norm(verts, axis=1)) ** 2), ze), axis=1) + (verts * vy)
    f_z = np.concatenate((ze, ze, -ascol((npl.norm(verts, axis=1)) ** 2)), axis=1) + (verts * vz)
    gamma_np_ext['ngrad_x'] = np.concatenate((f_x, gamma_np['ngrad_x']), axis=1)
    gamma_np_ext['ngrad_y'] = np.concatenate((f_y, gamma_np['ngrad_y']), axis=1)
    gamma_np_ext['ngrad_z'] = np.concatenate((f_z, gamma_np['ngrad_z']), axis=1)

    g_x = np.concatenate((n_dot_p, ze, ze), axis=1) - 2 * (verts * nx) + (normals * vx)
    g_y = np.concatenate((ze, n_dot_p, ze), axis=1) - 2 * (verts * ny) + (normals * vy)
    g_z = np.concatenate((ze, ze, n_dot_p), axis=1) - 2 * (verts * nz) + (normals * vz)
    gamma_np_ext['pgrad_x'] = np.concatenate((g_x, gamma_np['pgrad_x']), axis=1)
    gamma_np_ext['pgrad_y'] = np.concatenate((g_y, gamma_np['pgrad_y']), axis=1)
    gamma_np_ext['pgrad_z'] = np.concatenate((g_z, gamma_np['pgrad_z']), axis=1)

    return gamma_np_ext


def distance_extended_field_to_data(verts: np.array, normals: np.array, fit_torsion: np.array,
                                    fit_rotation: np.array,
                                    fit_translation: np.array, fit_scale: float,
                                    w_p: float) -> np.array:
    max_vertices = len(verts)
    vp_extended = np.cross(np.cross(fit_torsion, verts), verts) \
                  + np.cross(fit_rotation, verts) \
                  + fit_translation \
                  + fit_scale * verts
    vpn_extended = np.einsum('oi,oi->o', vp_extended, normals)
    ndotp = np.reshape(np.einsum('oi,oi->o', verts, normals), (max_vertices, 1))
    gradp_extended = fit_torsion * ndotp - 2 * verts * np.inner(fit_torsion, normals).transpose() \
                     + normals * np.inner(fit_torsion, verts).transpose() \
                     + np.cross(fit_rotation, normals) \
                     + fit_scale * normals
    gradps_extended = npl.norm(gradp_extended, axis=1) ** 2
    vps_extended = npl.norm(vp_extended, axis=1) ** 2
    vpst_extended = np.reshape(vps_extended, (1, max_vertices))

    delta_extended = vpn_extended / np.sqrt(vpst_extended + w_p * gradps_extended)
    return delta_extended


def get_gamma(verts, normals, extended) -> np.array:
    # Parameter space - standard spiral field
    p_cross_n = np.cross(verts, normals)
    n_dot_p = ascol(mutil.matlab_dot(verts, normals, axis=1))

    # Kinematic motion space
    if extended:
        # extended spiral field
        ncrossp_cross_p = np.cross(np.cross(normals, verts), verts)
        ret = np.concatenate((ncrossp_cross_p, p_cross_n, normals, n_dot_p), axis=1)
    else:
        # standard spiral field
        ret = np.concatenate((p_cross_n, normals, n_dot_p), axis=1)
    return ret


def init_mn_scale(max_verts: int) -> (np.array, np.array):
    M_scale = ones((max_verts, 1))
    N_scale = ones((max_verts, 1))
    return M_scale, N_scale


def get_mn_scale(gamma, gamma_np, params, max_verts, z, w_p):
    M_scale, N_scale = init_mn_scale(max_verts)

    dn = np.zeros((3, 1))
    dp = np.zeros((3, 1))

    for i in range(0, max_verts):
        # Evaluate data with parameters
        eval_data = np.dot(gamma[i, :], params)

        dn[0] = np.dot(gamma_np['ngrad_x'][i, :], params)
        dn[1] = np.dot(gamma_np['ngrad_y'][i, :], params)
        dn[2] = np.dot(gamma_np['ngrad_z'][i, :], params)
        dp[0] = np.dot(gamma_np['pgrad_x'][i, :], params)
        dp[1] = np.dot(gamma_np['pgrad_y'][i, :], params)
        dp[2] = np.dot(gamma_np['pgrad_z'][i, :], params)

        inv_grad = 1 / (np.dot(dn.T, dn) + w_p * np.dot(dp.T, dp))

        M_scale[i] = z[i] * inv_grad
        N_scale[i] = (eval_data ** 2) * z[i] * (inv_grad ** 2)
    return M_scale, N_scale


def get_mn_matrices(gamma: np.array, gamma_np: np.array, max_verts: int, w_p: float,
                    M_scale: np.array = None, N_scale: np.array = None,
                    M: np.array = 0, N: np.array = 0) -> (np.array, np.array):
    def _outer(a):
        return np.outer(a.T, a)

    if M_scale is None:
        M_scale, _ = init_mn_scale(max_verts)
    if N_scale is None:
        _, N_scale = init_mn_scale(max_verts)

    for i in range(0, max_verts):
        M = M + _outer(gamma[i, :]) * M_scale[i]
        N_ngrad = (_outer(gamma_np['ngrad_x'][i, :]) +
                   _outer(gamma_np['ngrad_y'][i, :]) +
                   _outer(gamma_np['ngrad_z'][i, :])) * N_scale[i]
        N_pgrad = (_outer(gamma_np['pgrad_x'][i, :]) +
                   _outer(gamma_np['pgrad_y'][i, :]) +
                   _outer(gamma_np['pgrad_z'][i, :])) * N_scale[i]
        N = N + N_ngrad + (N_pgrad * w_p)

    return M, N


def distance_field_to_data(verts: np.array, normals: np.array, fit_c: np.array, fit_c_bar: np.array,
                           fit_gamma: float,
                           w_p: float) -> np.array:
    max_vertices = len(verts)

    vp = np.cross(fit_c, verts) + fit_c_bar + fit_gamma * verts
    vpn = np.einsum('oi,oi->o', vp, normals)
    vps = npl.norm(vp, axis=1) ** 2
    vpst = np.reshape(vps, (1, max_vertices))
    gradp = np.cross(fit_c, normals) + fit_gamma * normals
    gradps = (npl.norm(gradp, axis=1) ** 2).transpose()

    delta = vpn / np.sqrt(vpst + w_p * gradps)
    return delta


def fit_from_eigenvector(v: np.array, extended: bool):
    if not isinstance(v, np.ndarray):
        raise ValueError(f'Expected vector as ndarray ({type(v)})')

    if extended:
        fit_torsion = v[0:3].transpose()
        fit_rotation = v[3:6].transpose()
        fit_translation = v[6:9].transpose()
        fit_scale = v[9]
        return fit_torsion, fit_rotation, fit_translation, fit_scale
    else:
        fit_c = v[0:3].transpose()
        fit_c_bar = v[3:6].transpose()
        fit_gamma = v[6]
        return fit_c, fit_c_bar, fit_gamma


def eigenvector_from_mn(M, N, extended):
    # Solve generalized eigenvalue problem for spiral field
    # First eigenvector of kinematic parameters gives spiral motion
    d, v = sla.eigsh(M, 1, N, which='SM')  # sigma=None as N is positive definite and symmetric
    v_scaled = v / npl.norm(v)

    if extended:
        fit_torsion, fit_rotation, fit_translation, fit_scale = fit_from_eigenvector(v_scaled,
                                                                                     extended)
        scalar = fit_scale
    else:
        fit_c, fit_c_bar, fit_gamma = fit_from_eigenvector(v_scaled, extended)
        scalar = fit_gamma
    if np.sign(scalar) == 1:
        v_scaled *= -1

    return v_scaled


def distance_field_to_data_helper(verts, normals, v_scaled, w_p, extended):
    if extended:
        fit_torsion, fit_rotation, fit_translation, fit_scale = fit_from_eigenvector(v_scaled,
                                                                                     extended)
        delta = distance_extended_field_to_data(verts, normals, fit_torsion, fit_rotation,
                                                fit_translation,
                                                fit_scale, w_p)
    else:
        fit_c, fit_c_bar, fit_gamma = fit_from_eigenvector(v_scaled, extended)
        delta = distance_field_to_data(verts, normals, fit_c, fit_c_bar, fit_gamma, w_p)
    return delta


def fit_normalized_space(verts, normals, w_p, extended):
    # Kinematic motion space of spiral field
    gamma = get_gamma(verts, normals, extended)
    gamma_np = init_standard_spiral_field(verts, normals)
    if extended:
        gamma_np = init_extended_spiral_field(verts, normals, gamma_np)

    # Compute fit from M and N
    max_verts = len(verts)

    M, N = get_mn_matrices(gamma, gamma_np, max_verts, w_p)
    v_scaled = eigenvector_from_mn(M, N, extended)

    # Compute distances between detected spiral field and data
    delta = distance_field_to_data_helper(verts, normals, v_scaled, w_p, extended)

    return v_scaled, delta


def fit_spiral_field(params, max_verts, gamma, gamma_np, M, N, w_p, z, extended):
    M_scale, N_scale = get_mn_scale(gamma, gamma_np, params, max_verts, z, w_p)

    M, N = get_mn_matrices(gamma, gamma_np, max_verts, w_p, M=M, N=N,
                           M_scale=M_scale, N_scale=N_scale)

    v_scaled = eigenvector_from_mn(M, N, extended)

    return v_scaled


def iterative_aml(verts, normals, max_iter, v_scaled, delta, w_p, extended, verbose):
    # Auxiliary variables
    max_verts = len(verts)
    if extended:
        M = np.zeros((10, 10))
        N = np.zeros(M.shape)
    else:
        M = np.zeros((7, 7))
        N = np.zeros(M.shape)

    # Kinematic motion space of spiral field
    gamma = get_gamma(verts, normals, extended)
    gamma_np = init_standard_spiral_field(verts, normals)
    if extended:
        gamma_np = init_extended_spiral_field(verts, normals, gamma_np)

    # Robust fitting: estimate degree of freedom nu and confidence w
    mu, S, nu, w = student_t_vectorized(delta)
    z = np.array(w).transpose()

    if verbose:
        field_str = 'extended' if extended else 'standard'
        print(f'Iterative AML of {field_str} spiral field')

    for iter_nr in range(0, max_iter):
        if verbose:
            print(f'Iteration {iter_nr + 1}/{max_iter}')

        # Last detected parameters
        if extended:
            fit_torsion, fit_rotation, fit_translation, fit_scale = fit_from_eigenvector(v_scaled,
                                                                                         extended)
            params = np.concatenate((fit_torsion, fit_rotation, fit_translation, ascol(fit_scale)),
                                    axis=1)
        else:
            fit_c, fit_c_bar, fit_gamma = fit_from_eigenvector(v_scaled, extended)
            params = np.concatenate((fit_c, fit_c_bar, ascol(fit_gamma)), axis=1).transpose()
        # iteratively fit spiral field
        v_scaled = fit_spiral_field(params, max_verts, gamma, gamma_np,
                                    M, N, w_p, z, extended)

    return v_scaled


def fit_cochlear_shape(verts, normals, max_iter, w_p, extended=False, verbose=False):
    time_start = time.time()

    # Initial fit
    v_scaled_0, delta_0 = fit_normalized_space(verts, normals, w_p, extended)
    # Iterative approximate maximum likelihood
    v_scaled = iterative_aml(verts, normals, max_iter, v_scaled_0, delta_0, w_p, extended, verbose)

    if verbose:
        print('Cochlea shape fit took {:.2f}s'.format(time.time() - time_start))

    if extended:
        fit_torsion, fit_rotation, fit_translation, fit_scale = fit_from_eigenvector(v_scaled,
                                                                                     extended)
        if verbose:
            print(f'fit_torsion: {fit_torsion}')
            print(f'fit_rotation: {fit_rotation}')
            print(f'fit_translation: {fit_translation}')

        return fit_torsion, fit_rotation, fit_translation, fit_scale
    else:
        fit_c, fit_c_bar, fit_gamma = fit_from_eigenvector(v_scaled, extended)
        if verbose:
            print(f'fit_c: {fit_c}')
            print(f'fit_c_bar: {fit_c_bar}')
            print(f'fit_gamma: {fit_gamma}')

        return fit_c, fit_c_bar, fit_gamma


def get_center_vzero(fit_c, fit_c_bar, fit_gamma):
    center_vzero = (np.cross(fit_gamma * fit_c, fit_c_bar) - fit_gamma ** 2 * fit_c_bar
                    - np.inner(fit_c, fit_c_bar) * fit_c) / (
                               fit_gamma * np.inner(fit_c, fit_c) + fit_gamma ** 2)
    center_vzero = center_vzero.flatten()
    return center_vzero


def get_rotation_axis(fit_c, fit_c_bar, fit_gamma):
    kin_A = fit_c
    kin_A_bar = (np.inner(fit_c, fit_c) * fit_c_bar - np.inner(fit_c, fit_c_bar) * fit_c
                 + np.cross(fit_gamma * fit_c, fit_c_bar)) / (
                            np.inner(fit_c, fit_c) + fit_gamma ** 2)
    kin_A = kin_A.flatten()
    kin_A_bar = kin_A_bar.flatten()
    return kin_A, kin_A_bar


def save_as_csv(values, dir_out, filename, delimiter=','):
    np.savetxt(os.path.join(dir_out, filename + '.csv'),
               values, delimiter=delimiter)


def save_as_txt(values, dir_out, filename):
    with open(os.path.join(dir_out, filename + '.txt'), 'w') as f:
        f.write(str(values))


def my_normalizer(my_array):
    """
    Loads an array which will be normalized and centralized.

    Input: array to be normalized in format n x 3
    Output: normalized array as n x 3, the used centralizer and the used normalizer

    to get only a normalized array x from my_array type e.g. x = my_normalizer(my_array)[0]
    """
    min_vec = my_array.min(axis=0)
    max_vec = my_array.max(axis=0)
    centralizer = (min_vec + max_vec) / 2
    normalizer = abs(max_vec - min_vec).max()
    return (my_array - centralizer) / normalizer, centralizer, normalizer


def ascol(arr):
    """
    If the dimensionality of 'arr' is 1, reshapes it to be a column matrix (N,1).
    """
    if len(arr.shape) == 1: arr = arr.reshape((arr.shape[0], 1))
    return arr


def asrow(arr):
    """
    If the dimensionality of 'arr' is 1, reshapes it to be a row matrix (1,N).
    """
    if len(arr.shape) == 1: arr = arr.reshape((1, arr.shape[0]))
    return arr


def arccos(val):
    if (val < -1) or (val > 1):
        logging.warning(f'Can not compute arccos({val}), as outside of possible domain.')
    theta = np.arccos(val)
    return theta


def vertex_normals_from_mesh(verts, faces):
    """
    Loads an array of vertices (N x 3) and faces (N x 3)
    Output: array of vertex normals

    The program first calculated the edges of each triangular face.
    The angles at a vertex between two adjoining edges are computed.
    The face normals are obtained as cross product between two edges.
    Vertex normals are calculated from the face normals weighted by
    the corresponding angle at the vertex for this face.

    Based on patchnormals.m written by D.Kroon University of Twente (June 2009).
    """
    Fa = faces[:, 0]
    Fb = faces[:, 1]
    Fc = faces[:, 2]
    e1 = verts[Fa, :] - verts[Fb, :]
    e2 = verts[Fb, :] - verts[Fc, :]
    e3 = verts[Fc, :] - verts[Fa, :]
    e1_norm = preprocessing.normalize(e1)
    e2_norm = preprocessing.normalize(e2)
    e3_norm = preprocessing.normalize(e3)

    edge_angle = np.zeros((len(e1), 3))
    # TODO: Correct if vertex normals point toward cochlear center instead of outside (minus sign for center)?
    for i in range(len(e1)):
        edge_angle[i, :] = [arccos(np.dot(e1_norm[i, :], -e3_norm[i, :])),
                            arccos(np.dot(e2_norm[i, :], -e1_norm[i, :])),
                            arccos(np.dot(e3_norm[i, :], -e2_norm[i, :]))]
    face_normals = np.zeros((len(e1), 3))
    for i in range(len(e1)):
        face_normals[i] = np.cross(e1[i, :], e3[i, :])

    vertex_normals_scaled = np.zeros((len(verts), 3))
    for i in range(len(Fa)):
        vertex_normals_scaled[Fa[i], :] = vertex_normals_scaled[Fa[i], :] + (
                    face_normals[i, :] * edge_angle[i, 0])
        vertex_normals_scaled[Fb[i], :] = vertex_normals_scaled[Fb[i], :] + (
                    face_normals[i, :] * edge_angle[i, 1])
        vertex_normals_scaled[Fc[i], :] = vertex_normals_scaled[Fc[i], :] + (
                    face_normals[i, :] * edge_angle[i, 2])
    return preprocessing.normalize(vertex_normals_scaled)


def fitt_optnu(x, delta, p):
    from scipy.special import psi

    nu = x
    # function to solve for ML nu estimate
    nu2 = nu / 2
    pnu2 = (p + nu) / 2
    w = (p + nu) / (delta + nu)

    f = -psi(nu2) + np.log(nu2) + (np.sum(np.log(w) - w) / len(delta)) + 1 + \
        psi(pnu2) - np.log(pnu2)
    return f


def student_t_vectorized(x, nu=5, eps=1e-8, max_iter=500, verbose=False):
    """
    Fit a t-distribution using the ECME algorithm (Lui & Rubin, 1995)
    Optimized and vectorized version using standard NumPy/SciPy functions.

    C Liu and D B Rubin, (1995) "ML estimation of the t distribution using EM and
    its extensions, ECM and ECME", Statistica Sinica, 5, pp19-39
    http://www3.stat.sinica.edu.tw/statistica/oldpdf/A5n12.pdf

    @param x: fit student_t to x (Expected shape: (N_trl, N_var) or (N_trl,) for 1D)
    @param nu: DOF
    @param eps: tolerance for entropy
    @param max_iter:
    @param verbose:
    """
    from scipy.optimize import fsolve
    from scipy.special import gammaln, psi, beta  # 'math.pi' can be replaced by 'np.pi'
    from scipy.linalg import solve_triangular

    x = np.array(x, dtype=np.float64)

    # --------------------------------------------------------------------------
    # 1. Input Handling and Initialization (Vectorized)
    # --------------------------------------------------------------------------
    if x.ndim == 1:
        # Standard case: N_trl observations of N_var=1 variable
        x = x.reshape(-1, 1)  # Ensure x is (N_trl, 1)

    Ntrl, Nvar = x.shape
    p = Nvar

    # Handle the singular case (p=1, covariance S is a scalar)
    is_scalar = (p == 1)

    if Ntrl == 0:
        raise ValueError("Input array 'x' is empty (Ntrl=0). Cannot fit a t-distribution to empty data.")

    if Ntrl <= p:
        # Ntrl <= p leads to a singular (non-invertible) covariance matrix S,
        # which will cause cholesky to fail or produce NaNs.
        # While technically possible for t-distribution fitting, it's ill-conditioned.
        # We warn and proceed, but for Ntrl=0, we raise an error.
        if verbose:
            print(f"Warning: Ntrl ({Ntrl}) is not greater than Nvar ({Nvar}). Covariance matrix may be singular.")

    # Initial estimates
    # Use ddof=1 for sample covariance/variance calculation
    if is_scalar:
        S = np.cov(x.flatten(), ddof=1)  # S is a scalar
    else:
        S = np.cov(x, rowvar=False, ddof=1)  # S is a p x p matrix if Ntrl>1, or a scalar/NaNs if Ntrl=1

        # Check if np.cov returned a scalar (0-dim array) or a 1x1 array.
        # If it's not a proper p x p matrix, we must switch to the univariate (p=1) logic.
        if S.ndim < 2 or (S.ndim == 2 and S.shape != (p, p)):
            # This occurs when Ntrl <= p. We cannot proceed with the multivariate fit.
            # Force the problem to be treated as univariate (p=1) and S as the variance of the flattened array.

            # S becomes the variance of all values treated as a single distribution.
            S = np.cov(x.flatten(), ddof=1)

            # Reset parameters to reflect univariate mode
            is_scalar = True
            p = 1
            x = x.flatten().reshape(-1, 1)  # x becomes (N, 1) for consistent loop math
            Ntrl, Nvar = x.shape  # Ntrl is now N, Nvar is 1

            if verbose:
                print(f"Warning: Ntrl ({Ntrl}) was less than Nvar ({Nvar}). Forcing switch to univariate fit.")
    mean = np.mean(x, axis=0)  # mean is (1, p) or (p,)

    H = 0.0
    converged = False
    i = 0

    # Pre-calculate constants (Vectorized)
    p2 = p / 2
    # log_npip2 = np.log((nu * np.pi) ** p2 * beta(p2, nu / 2)) - gammaln(p2)
    # Safe calculation using log-domain arithmetic
    nu2 = nu / 2
    nup2 = (nu + p) / 2
    log_npip2 = p2 * np.log(nu * np.pi) + gammaln(nu2) - gammaln(nup2)

    while (i < max_iter) and not converged:
        H_old = H
        i += 1

        # ----------------------------------------------------------------------
        # E step (Vectorized Mahalanobis Distance)
        # ----------------------------------------------------------------------

        # Calculate Cholesky decomposition L such that S = L L^T.
        # The Mahalanobis distance delta = (x - mean) S^{-1} (x - mean)^T
        # If S = L L^T, then S^{-1} = (L^T)^{-1} L^{-1}.
        # di = x - mean. We need ||L^{-1} di^T||^2.

        # Recalculate di here, as the CM-1 step updates mean and S
        di = x - mean  # (Ntrl, p)

        # Ensure cholesky is only called for matrices
        if is_scalar:
            # S is a scalar variance. L = sqrt(S).
            chS_diag = np.sqrt(S)  # S is a scalar variance

            # M = di / chS_diag (Vectorized)
            M = di / chS_diag  # (Ntrl, 1)
        else:
            # S is a matrix. L = cholesky(S)
            L = np.linalg.cholesky(S)  # L is lower-triangular

            # Use solve_triangular (Vectorized)
            M_T = solve_triangular(L, di.T, lower=True)
            M = M_T.T  # (Ntrl, p)

        # Mahalanobis distance delta = sum(M * M, axis=1) (Vectorized)
        delta = np.sum(M * M, axis=1)  # (Ntrl,)

        # Weights w (Vectorized)
        w = (p + nu) / (delta + nu)  # (Ntrl,)

        # ----------------------------------------------------------------------
        # CM-1 Step (Vectorized Mean and Covariance)
        # ----------------------------------------------------------------------

        # New mean
        mean = np.sum(x * w[:, np.newaxis], axis=0) / np.sum(w)  # (p,)

        # New centered data di using updated mean
        di = x - mean  # (Ntrl, p)

        # New covariance S
        # S = (di * w) * di.T / Ntrl
        # S = (di.T @ (di * w[:, np.newaxis])) / Ntrl

        # Since w is (Ntrl,), use broadcasting with np.newaxis to multiply
        # diw is (Ntrl, p). Inner product np.inner(diw, diw) is for 1D arrays
        # Use matrix multiplication for correct covariance matrix (Vectorized)
        diw = di * np.sqrt(w[:, np.newaxis])  # (Ntrl, p)

        if is_scalar:
            # S remains a scalar (variance)
            S = np.sum(diw * diw) / Ntrl  # Scalar
        else:
            # S remains a matrix (covariance)
            S = (diw.T @ diw) / Ntrl  # (p, p)

        # ----------------------------------------------------------------------
        # CM-2 Step (Only every other iteration)
        # ----------------------------------------------------------------------
        if (i % 2) == 0:
            # E step again (re-calculate w with new S)
            if is_scalar:
                chS_diag = np.sqrt(S)
                M = di / chS_diag
            else:
                L = np.linalg.cholesky(S)
                M_T = solve_triangular(L, di.T, lower=True)
                M = M_T.T

            delta = np.sum(M * M, axis=1)
            w = (p + nu) / (delta + nu)

            # Optimization for nu (Vectorized args for fsolve)
            # You must define fitt_optnu outside this function!
            # The args for fsolve are (delta, p)
            sol, _, success, msg = fsolve(fitt_optnu, x0=nu, args=(delta, p),
                                          xtol=1e-10, full_output=True)

            if not success:
                raise Exception(f'minimize did not converge: {msg}')
            nu = sol[0]

            # Convergence detection (Vectorized Entropy Calculation)
            nu2 = nu / 2
            nup2 = (nu + p) / 2

            if is_scalar:
                chS_log_det = np.log(np.sqrt(S))
            else:
                # Cholesky of S is L. log|chS| = log|L| = sum(log(diag(L)))
                chS_log_det = np.sum(np.log(np.diag(L)))

            # Entropy H (Vectorized)
            H = chS_log_det + log_npip2 + nup2 * (psi(nup2) - psi(nu2))

            converged = (np.abs(H - H_old) < eps)

            if verbose:
                print(f'Iteration {i}: nu={nu:.4f}, H={H:.4f}, |H-H_old|={np.abs(H - H_old):.2e}')

    if not converged:
        raise Exception(f'fitt:ECME algorithm did not converge (MAXITER {max_iter} exceeded)')
    if verbose:
        print(f'ECME algorithm converged to {nu} after {i} iterations')

    # Re-calculate final weights for output
    if i % 2 != 0:
        # If the loop stopped on an odd iteration, w needs to be re-calculated
        di = x - mean
        if is_scalar:
            chS_diag = np.sqrt(S)
            M = di / chS_diag
        else:
            L = np.linalg.cholesky(S)
            M_T = solve_triangular(L, di.T, lower=True)
            M = M_T.T

        delta = np.sum(M * M, axis=1)
        w = (p + nu) / (delta + nu)

    return mean, S, nu, w.flatten()


def compare_results_to_matlab(spec_name, kin_A, kin_Z):
    kin_A_matlab = {
        'I02a': [0.522806096680986, -0.807116589274900, 0.274292902916652],
        'I02b': [-0.557114367235595, -0.738772583802516, 0.379260663978533],
        'I03': [0.586675838274943, -0.801615442819785, 0.114996272188417],
        'I05': [-0.633333809296599, -0.667823803989488, 0.391023851992266],
        'I08': [-0.639110944889273, -0.689100556605813, 0.341581063597352],
        'I38': [0.416646968973850, -0.905811870846562, -0.0768775511990012],
        'I43': [-0.541187643048854, -0.829898235529106, 0.135590758080717],
        'I44': [-0.517195716069934, -0.839075674921787, -0.168702706064412],
        'SNF_F1': [-0.509005488244715, -0.859759424291127, -0.0415589374184357]
    }

    kin_Z_matlab = {
        'I02a': [30.0231275085860, -82.7031314359936, 632.128018424096],
        'I02b': [-38.3440425934378, -84.3063612510527, 634.218034190995],
        'I03': [37.1899077448487, -131.178931404099, -2.87590261781617],
        'I05': [-39.2157611413150, -84.1298072833371, -17.6546000833935],
        'I08': [-38.6926699047382, -164.189751764726, 405.236583256064],
        'I38': [37.0071603154898, -8.04622503213508, 1.27127356760424],
        'I43': [-38.9508641211396, -17.7662900527465, -9.49451539381198],
        'I44': [-36.2333000288773, 7.24282405594772, -6.55234080042511],
        'SNF_F1': [-8.67820778668994, -87.5038964030964, -54.1677305409391]
    }

    if kin_A_matlab.keys() != kin_Z_matlab.keys():
        logging.warning(
            f'Missing keys for matlab data ({kin_A_matlab.keys(), kin_Z_matlab.keys()})')
    elif spec_name not in kin_A_matlab.keys():
        logging.warning(f"'{spec_name}' missing in matlab keys ({kin_A_matlab.keys()})")
    else:
        kin_A_deviation = angle_between_3D_vectors(kin_A, kin_A_matlab[spec_name], unit='degree')
        print('Compared to Matlab, rotational axis differs by an angle of {:.2f}°'.format(
            kin_A_deviation))
        v_zero_deviation = np.linalg.norm(kin_Z - kin_Z_matlab[spec_name])
        print('Compared to Matlab, center of velocity zero differs by a distance of {:.2f}'.format(
            v_zero_deviation))
