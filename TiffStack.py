"""Manage an imaging session tiff stack and ROI masks.
"""
import glob
import logging
import os
import sys
import time
from typing import List, Sequence

import numpy as np
from skimage import io
from tqdm import tqdm

logger = logging.getLogger("TiffStack")
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.WARNING)


class TiffStack:
    def __init__(self, tiffPattern: str = "*.tif*"):
        self._tiffPattern = tiffPattern
        self.timeseries = self.open_TIFF_stack(tiffPattern)
        self.masks: List[np.ndarray] = []

    def open_TIFF_stack(self, tiffsPattern: str) -> np.ndarray:
        """Open and concatenate a set of tiffs.

        Arguments:
            tiffsPattern {str} -- File glob pattern.)

        Returns:
            ndarray -- Timeseries (frames by x-coord by y-coord).
        """
        startTime = time.time()
        logger.info(f"Opening files matching tiffsPattern = {tiffsPattern}")
        tiffFNames = sorted(glob.glob(tiffsPattern))
        stacks = []
        filePattern = tiffsPattern.split(os.sep)[-1]
        for fname in tqdm(tiffFNames, desc=filePattern, unit="file"):
            stacks.append(io.imread(fname))
        logger.info("Concatenating.")
        stack = np.concatenate(stacks, axis=0)
        logger.debug(f"TiffStack creation time: {time.time() - startTime}")
        return stack

    def add_masks(self, maskPattern: str = "*.bmp") -> None:
        self.masks += self._get_masks(maskPattern)

    def _get_masks(self, maskPattern: str) -> Sequence[np.ndarray]:
        """Get ROI masks from files.

        Arguments:
            maskPattern {str} -- File glob pattern.

        Returns:
            [2d ndarray] -- List of masks.
        """
        maskFNames = sorted(glob.glob(maskPattern))
        masks: List[np.ndarray] = []
        for maskFName in maskFNames:
            mask = io.imread(maskFName)
            invMask = mask == 0
            masks += [mask] if np.sum(mask) < np.sum(invMask) else [invMask]
        return masks

    def cut_to_averages(self, forceNew=False) -> np.ndarray:
        """Get average intensity of each ROI at each frame.

        Keyword Arguments:
            forceNew {bool} -- Force calculating averages even if .averages already
                exists. (default: {False})

        Returns:
            ndarray -- Frames by ROI.
        """
        if forceNew or (not hasattr(self, "averages")):
            ROIaverages = np.empty((self.timeseries.shape[0], len(self.masks)))
            for i_mask, mask in enumerate(self.masks):
                ROIaverages[:, i_mask] = np.mean(self.timeseries[:, mask], axis=1)
            self.averages = ROIaverages
        else:
            ROIaverages = self.averages
        return ROIaverages
