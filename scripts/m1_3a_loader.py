"""Validated offline OAM-TCD patch loader for M1-3A."""
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from m1_2d_geometry_audit import require_hash


def check_split(sample, expected):
    assigned = {661: 'train', 1972: 'val'}
    if sample['split'] != expected or assigned.get(sample['image_id']) != expected:
        raise ValueError('Sample/split mismatch')


def load_patch(root, sample, expected_split, size):
    check_split(sample, expected_split)
    if not isinstance(size, int) or size <= 0:
        raise ValueError('Positive integer patch size required')
    folder = Path(root) / expected_split / str(sample['image_id'])
    arrays = {}
    for name, expected_hash in sample['output_hashes'].items():
        path = folder / name
        require_hash(path, expected_hash)
        with rasterio.open(path) as src:
            if (str(src.crs) != sample['output_crs'] or src.transform != Affine(*sample['output_transform'])
                    or list(src.shape) != sample['output_shape']
                    or not np.allclose(src.bounds, sample['output_bounds'], atol=1e-8, rtol=0)):
                raise ValueError('RGB/teacher grid mismatch')
            if not (src.dataset_mask() == 255).all() or src.nodata is not None:
                raise ValueError('Unexpected nodata')
            arrays[name] = src.read()
    rgb = arrays['rgb.tif']
    mask = arrays['canopy_mask.tif']
    fraction = arrays['canopy_fraction.tif']
    valid = arrays['valid_fraction.tif']
    if rgb.dtype != np.uint8 or rgb.shape != (3, *sample['output_shape']):
        raise ValueError('Expected uint8 RGB CHW')
    if mask.dtype != np.uint8 or mask.shape != (1, *sample['output_shape']) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected binary uint8 teacher')
    for array in (fraction, valid):
        if array.dtype != np.float32 or array.shape != mask.shape or not np.isfinite(array).all() or not ((array >= 0) & (array <= 1)).all():
            raise ValueError('Invalid fraction dtype/shape/range')
    if not np.all(valid == 1) or not np.array_equal(mask, (fraction >= 0.5).astype(np.uint8)):
        raise ValueError('Validity/teacher mismatch')
    if size > min(rgb.shape[1:]):
        raise ValueError('Padding forbidden: patch exceeds valid source extent')
    # Deterministic top-left crop shared by every input field.
    crop = {name: array[:, :size, :size].copy() for name, array in arrays.items()}
    images = torch.from_numpy(crop['rgb.tif']).float().unsqueeze(0) / 255
    targets = torch.from_numpy(crop['canopy_mask.tif']).long()
    if images.shape != (1, 3, size, size) or targets.shape != (1, size, size):
        raise ValueError('Invalid model input shape')
    if not torch.isfinite(images).all() or not ((images >= 0) & (images <= 1)).all():
        raise ValueError('Invalid normalized RGB')
    return images, targets, crop
