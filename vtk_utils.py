#!/usr/bin/env python3

"""
VTK utility functions
Author: SCS
Date: 26.07.2021
"""

import numpy as np
import SimpleITK as sitk


def resample_to_standard_orientation(sitk_image: sitk.Image) -> sitk.Image:
    """
    Resamples a SimpleITK image to a standard axial orientation (identity direction).
    Uses the Bounding Box of the original image to determine the output extent.
    """

    # 1. Setup Resampler and Default Value
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    default_value = float(np.min(sitk.GetArrayFromImage(sitk_image)))
    resampler.SetDefaultPixelValue(default_value)

    # 2. Define the Target Grid Direction (Standard)
    identity_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    resampler.SetOutputDirection(identity_direction)

    # 3. Define Output Spacing (Maintain Original Spacing)
    spacing = sitk_image.GetSpacing()
    resampler.SetOutputSpacing(spacing)

    # 4. Calculate the New Bounding Box
    size = sitk_image.GetSize()

    # Get the 8 corners of the original image in IJK space (NumPy array of NumPy integers)
    corners_ijk = np.array([[0, 0, 0], [size[0] - 1, 0, 0], [0, size[1] - 1, 0], [0, 0, size[2] - 1],
                            [size[0] - 1, size[1] - 1, 0], [size[0] - 1, 0, size[2] - 1],
                            [0, size[1] - 1, size[2] - 1], [size[0] - 1, size[1] - 1, size[2] - 1]])

    # Convert corners to physical space (ensure correct type as int)
    corners_physical = [sitk_image.TransformIndexToPhysicalPoint([int(i) for i in c])
                        for c in corners_ijk]

    # Find the min/max coordinates (the bounding box)
    corners_physical_np = np.array(corners_physical)
    new_origin = np.min(corners_physical_np, axis=0)
    max_coords = np.max(corners_physical_np, axis=0)

    # Calculate New Size based on the bounding box
    new_size_phys = max_coords - new_origin

    # Calculate voxel size and round up to the nearest integer
    new_size_voxels = np.ceil(new_size_phys / spacing).astype(int)

    # 5. Set the New Output Parameters
    resampler.SetOutputOrigin(new_origin)
    resampler.SetSize([int(i) for i in new_size_voxels])

    # Set the transform to be identity since we are only changing direction/origin
    transform = sitk.Transform(3, sitk.sitkIdentity)
    resampler.SetTransform(transform)

    # 6. Execute Resampling
    resampled_image = resampler.Execute(sitk_image)

    return resampled_image
