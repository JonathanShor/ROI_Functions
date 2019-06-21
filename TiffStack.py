"""Manage an imaging session tiff stack and ROI masks.
"""
import glob

import h5py
import numpy as np
import pandas as pd
from skimage import io


class TiffStack:
    def __init__(self, tiffPattern="*.tif*"):
        self._tiffPattern = tiffPattern
        self.timeseries = self.open_TIFF_stack(tiffPattern)
        self.masks = []

    def open_TIFF_stack(self, tiffsPattern):
        """Open and concatenate a set of tiffs.

        Arguments:
            tiffsPattern {str} -- File glob pattern.)

        Returns:
            ndarray -- Timeseries (frames by x-coord by y-coord).
        """
        tiffFNames = sorted(glob.glob(tiffsPattern))
        stack = io.imread(tiffFNames[0])
        for fname in tiffFNames[1:]:
            next = io.imread(fname)
            stack = np.concatenate((stack, next), axis=0)
        return stack

    def add_masks(self, maskPattern="*.bmp"):
        self.masks.append(self._get_masks(maskPattern))

    def _get_masks(self, maskPattern):
        """Get ROI masks from files.

        Arguments:
            maskPattern {str} -- File glob pattern.

        Returns:
            [2d ndarray] -- List of masks.
        """
        maskFNames = sorted(glob.glob(maskPattern))
        masks = []
        for maskFName in maskFNames:
            mask = io.imread(maskFName)
            invMask = mask == 0
            masks += [mask] if np.sum(mask) < np.sum(invMask) else [invMask]
        return masks

    def cut_to_averages(self):
        """Get average intensity of each ROI at each frame.

        Returns:
            ndarray -- Frames by ROI.
        """
        ROIaverages = np.empty((self.timeseries.shape[0], len(self.masks)))
        for i_mask, mask in enumerate(self.masks):
            ROIaverages[:, i_mask] = np.mean(self.timeseries[:, self.mask], axis=1)
        return ROIaverages
