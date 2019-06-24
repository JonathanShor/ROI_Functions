"""Analyze a 2P imaging session from tiff stack, h5 metadata, and ROI masks.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    return dF_Fs, session


def pick_layout(numPlots):
    numColumns = np.round(np.sqrt(numPlots))
    numRows = np.ceil(numPlots / numColumns)
    return int(numRows), int(numColumns)


def visualize(dF_Fs, session, axis, title="", figDims=(10, 9)):
    """Generate plots of dF/F traces.

    Each condition is plotted in its own subplot. All subplots in one figure.

    Arguments:
        dF_Fs {{str: ndarray}} -- Condition: dF/F trace. Typical trace might be frames by
            trials by ROI, but only constraint is that it play nicely with whatever is
            passed as axis argument.
        session {ImageSession} -- Imaging session metadata.
        axis {int or tuple of ints} -- The axes of each dF/F to sum over.

    Keyword Arguments:
        title {str} -- Printed above each subplot, along with that subplot's condition.
            (default: {""})
        figDims {tuple} -- Dimensions of the figure bounding all subplots (default: {(10,
            9)})

    Returns:
        matplotlib.figure.Figure -- The generated figure.
    """
    sns.set(rc={"figure.figsize": figDims})
    numConditions = len(dF_Fs)
    layout = pick_layout(numConditions)
    lockOffset = session.get_lock_offset()
    fig, axarr = plt.subplots(layout[0], layout[1], sharex=True)
    i_condition = -1
    for condition, dF_f in dF_Fs.items():
        i_condition += 1
        plotLocation = np.unravel_index(i_condition, layout)
        axarr[plotLocation].plot(lockOffset, np.nanmean(dF_Fs[condition], axis=axis))
        axarr[plotLocation].title.set_text(condition + ", " + title)
    return fig


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
