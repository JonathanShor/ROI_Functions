"""Process ROIs
"""
import glob

import numpy as np
from skimage import io
import pandas as pd
import h5py


def get_masks(maskPattern="*.bmp"):
    """Get ROI masks from files.

    Keyword Arguments:
        maskPattern {str} -- File glob pattern. (default: {"*.bmp"})

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


def get_bounds_index(mask, maxBound):
    maskIndexes = mask.nonzero()
    maxIndex = [max(i) for i in maskIndexes]
    assert all([maski <= boundi for maski, boundi in zip(maxIndex, maxBound)])
    minIndex = [min(i) for i in maskIndexes]
    assert all([maski >= 0 for maski in minIndex])
    return maxIndex, minIndex


def cut_to_bounding_boxes(timeseries, masks):
    """Cut up timeseries into ROI bounding boxes.

    Arguments:
        timeseries {ndarray} -- Source timeseries. First dimension is time.
        masks {[ndarray]} -- Sequence of binary masks.

    Returns:
        [ndarray] -- List of timeseries, each entry corresponding to an ROI as defined by
            masks.
    """
    ROIs = []
    for mask in masks:
        maxIndex, minIndex = get_bounds_index(mask, timeseries[0].shape)
        masked = timeseries[:, mask]
        ROIs += [masked[:, minIndex[0] : maxIndex[0], minIndex[1] : maxIndex[1]]]
    return ROIs


def cut_to_averages(timeseries, masks):
    """Get average intensity of each ROI at each frame.

    Arguments:
        timeseries {ndarray} -- Source timeseries. First dimension is time.
        masks {[ndarray]} -- Sequence of binary masks.

    Returns:
        ndarray -- Frames by ROI.
    """
    ROIaverages = np.empty((timeseries.shape[0], len(masks)))
    for i_mask, mask in enumerate(masks):
        ROIaverages[:, i_mask] = np.mean(timeseries[:, mask], axis=1)
    return ROIaverages


def get_dF_F(timeseries, width=20):
    """Calculate dF/F using local temporal window mean F.

    Arguments:
        timeseries {ndarray} -- Source timeseries. Expects time is first dim.

    Keyword Arguments:
        width {int} -- Frames before and after to include in mean. (default: {20})
    """
    numFrames = timeseries.shape[0]
    meanF = np.zeros_like(timeseries)
    # dF[0:width, :, :] =
    for i_frame, frame in enumerate(timeseries):
        meanF[i_frame] = np.mean(
            timeseries[max(0, i_frame - width) : min(numFrames, i_frame + width)],
            axis=0,
        )
    dF = (timeseries - meanF) / meanF
    dF = np.nan_to_num(dF)
    return dF


def open_TIFF_stack(tiffsPattern="*.tif*"):
    """Open and concatenate a set of tiffs.

    Keyword Arguments:
        tiffsPattern {str} -- File glob pattern. (default: {"*.tif*"})

    Returns:
        ndarray -- Timeseries (frames by x-coord by y-coord).
    """
    tiffFNames = sorted(glob.glob(tiffsPattern))
    stack = io.imread(tiffFNames[0])
    for fname in tiffFNames[1:]:
        next = io.imread(fname)
        stack = np.concatenate((stack, next), axis=0)
    return stack


def get_trials_metadata(h5Filename):
    trialsMeta = pd.read_hdf(h5Filename, key="Trials")
    return trialsMeta


def get_flatten_trial_data(h5Filename, key):
    """Extract data for key from all Trials in h5Filename.

    Arguments:
        h5Filename {str} -- Path to h5 file.

    Returns:
        ndarray -- Flat ndarray of the key data.
    """
    with h5py.File(h5Filename, "r") as h5File:
        # Assumes h5File contains Trial000# keys and one Trials key
        numTrials = len(h5File)
        keyData = np.empty(
            0, dtype=h5File["Trial{:04d}".format(numTrials - 1)][key][0].dtype
        )
        for i_trial in range(1, numTrials):
            trial = h5File["Trial{:04d}".format(i_trial)]
            for data in trial[key]:
                keyData = np.append(keyData, data)
    return keyData


def frame_from_timestamp(h5Filename, timestamps):
    frameTriggers = get_flatten_trial_data(h5Filename, "frame_triggers")
    sorters = np.argsort(frameTriggers, kind="mergesort")
    # assert all(frameTriggers == np.sort(frameTriggers))
    return np.searchsorted(frameTriggers, timestamps, sorter=sorters)


def process(tiffPattern, maskDir):
    timeseries = open_TIFF_stack(tiffPattern)
    masks = get_masks(maskDir)
    ROIaverages = cut_to_averages(timeseries, masks)
    dF_F = get_dF_F(ROIaverages)
    return dF_F
