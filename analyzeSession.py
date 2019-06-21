"""Analyze a 2P imaging session from tiff stack, h5 metadata, and ROI masks.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import processROI
from ImagingSession import ImagingSession
from TiffStack import TiffStack


def process(tiffPattern, maskDir, h5Filename):
    timeseries = TiffStack(tiffPattern)
    timeseries.add_masks(maskDir)
    ROIaverages = timeseries.cut_to_averages()
    frameTriggers = processROI.get_flatten_trial_data(
        h5Filename, "frame_triggers", clean=True
    )
    trialsMeta = processROI.get_trials_metadata(h5Filename)
    session = ImagingSession(trialsMeta["inh_onset"], frameTriggers)
    meanFs = session.get_meanFs(ROIaverages, frameWindow=2)
    dF_Fs = {}
    for condition in meanFs:
        dF_Fs[condition] = session.get_trial_average_data(
            ROIaverages, meanFs[condition], condition
        )
    lockOffset = list(
        range(-session.zeroFrame, session.maxSliceWidth - session.zeroFrame)
    )
    plt.plot(lockOffset, np.nanmean(dF_Fs[condition], axis=1))
    plt.title(condition + ", all ROI")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workingDir", default=".", help="Working directory containing tifs to analyze"
    )
    parser.add_argument(
        "--pattern",
        # Default will include all tif
        default="*.tif",
        help="File name pattern to select tifs, i.e. Run0034_00*.tif.",
    )
    args = parser.parse_args()
