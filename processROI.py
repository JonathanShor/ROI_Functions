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
    """Retrieve "Trials" metadata from an h5 file.

    Arguments:
        h5Filename {str} -- Source filepath

    Returns:
        pd.DataFrame -- Each DataFrame label corresponds to a h5 dataset.
    """
    trialsMeta = pd.read_hdf(h5Filename, key="Trials")
    return trialsMeta


def get_flatten_trial_data(h5Filename, key, clean=False):
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
    if clean:
        keyData = clean_data(keyData, key)
    return keyData


def clean_data(data, key):
    """Dispatch for key specific cleaning methods.

    Arguments:
        data {ndarray} -- Source data, a flattened h5 dataset.
        key {str} -- Key describing data

    Returns:
        ndarray -- Cleaned data
    """
    if key == "frame_triggers":
        data = clean_frame_trigger_data(data)
    return data


def analyze_frame_triggers(data):
    """Analyze frame_trigger data for anomalies.

    Arguments:
        data {ndarray} -- Flattened frame trigger dataset.
    """
    print("analyze_frame_triggers NOT implemented.")


def clean_frame_trigger_data(data, frameRate=100 / 3):
    """Correct common frame trigger indexing errors.

    Cleans mislabeled triggers, and imputes missing triggers.

    Arguments:
        data {ndarray} -- Numpy array of timestamps in ms.

    Keyword Arguments:
        frameRate {float} -- Expected distance between adjacent triggers. (default:
            {100/3})

    Returns:
        ndarray -- Cleaned data.
    """
    analyze_frame_triggers(data)
    # np.unique removes duplicates AND sorts
    cleanData = np.unique(data)
    interFrameIntervals = cleanData[1:] - cleanData[:-1]
    largeGapsIndexes = (interFrameIntervals > np.ceil(frameRate)).nonzero()[0]
    filledData = cleanData.copy()
    # Process gaps in reverse order so inserting fillins doesn't change indexing for
    # subsequent gap processing
    for i_gap, gapIndex in enumerate(largeGapsIndexes[::-1]):
        numFillIns = round(interFrameIntervals[gapIndex] / frameRate) - 1
        fillIns = (
            np.round((np.arange(numFillIns) + 1) * frameRate) + cleanData[gapIndex]
        )
        filledData = np.insert(filledData, gapIndex + 1, fillIns)
    assert np.all(np.abs((filledData[1:] - filledData[:-1]) - frameRate) <= 1)
    return filledData


def frame_from_timestamp(frameTriggers, timestamps):
    """Map timestamps to frames

    Arguments:
        frameTriggers {[long]} -- Sequence of (sorted) frame trigger timestamps.
        timestamps {[long] or long} -- Sequence or scalar of timestamps.

    Returns:
        ndarray -- Sequence of the indexes corresponding to the first timestamp greater
            than or equal to each element in timestamps.
    """
    sorters = np.argsort(frameTriggers, kind="mergesort")
    assert all(frameTriggers == frameTriggers[sorters])
    return np.searchsorted(frameTriggers, timestamps, sorter=sorters)


def upsample(signal, frameTimpstamps, targetFramerate=1.0):
    """Linearly upsample a signal.

    Note: May have difficulty with non-integer targetFramerates in some corner cases.

    Arguments:
        signal {[float]} -- Source signal.
        frameTimpstamps {[int]} -- Timestamps for each sample in signal. Required to
            be in ascending order and have no duplicate values.
        targetFramerate {float} -- Framerate to achieve.

    Returns:
        ndarray -- Upsampled signal.
    """
    targetTimestamps = np.append(
        np.round(np.arange(frameTimpstamps[0], frameTimpstamps[-1], targetFramerate)),
        frameTimpstamps[-1],
    )
    upsampledSignal = np.interp(
        targetTimestamps, frameTimpstamps, signal, right=np.nan, left=np.nan
    )
    assert not np.any(np.isnan(upsampledSignal))
    return upsampledSignal


def process(tiffPattern, maskDir, h5Filename):
    timeseries = open_TIFF_stack(tiffPattern)
    masks = get_masks(maskDir)
    ROIaverages = cut_to_averages(timeseries, masks)
    frameTriggers = get_flatten_trial_data(h5Filename, "frame_triggers", clean=True)
    dF_F = get_dF_F(ROIaverages)
    return dF_F
