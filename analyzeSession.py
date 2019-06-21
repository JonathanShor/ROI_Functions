"""Analyze a 2P imaging session from tiff stack, h5 metadata, and ROI masks.
"""
import argparse

import ImagingSession
import processROI


def process(tiffPattern, maskDir, h5Filename):
    timeseries = processROI.open_TIFF_stack(tiffPattern)
    masks = processROI.get_masks(maskDir)
    ROIaverages = processROI.cut_to_averages(timeseries, masks)
    frameTriggers = processROI.get_flatten_trial_data(
        h5Filename, "frame_triggers", clean=True
    )
    trialsMeta = processROI.get_trials_metadata(h5Filename)
    session = ImagingSession(trialsMeta["inh_onset"], frameTriggers)


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
