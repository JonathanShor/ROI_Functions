"""Analyze a 2P imaging session from tiff stack, h5 metadata, and ROI masks.
"""
import argparse
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import processROI
from ImagingSession import ImagingSession
from TiffStack import TiffStack


def process(tiffPattern, maskDir, h5Filename, sessionDetails):
    timeseries = TiffStack(tiffPattern)
    timeseries.add_masks(maskDir)
    ROIaverages = timeseries.cut_to_averages()
    frameTriggers = processROI.get_flatten_trial_data(
        h5Filename, "frame_triggers", clean=True
    )
    trialsMeta = processROI.get_trials_metadata(h5Filename)
    session = ImagingSession(trialsMeta["inh_onset"], frameTriggers, sessionDetails)
    meanFs = session.get_meanFs(ROIaverages, frameWindow=2)
    dF_Fs = {}
    for condition in meanFs:
        dF_Fs[condition] = session.get_trial_average_data(
            ROIaverages, meanFs[condition], condition
        )
    return dF_Fs, session


def pick_layout(numPlots: int) -> Tuple[int, int]:
    numColumns = np.round(np.sqrt(numPlots))
    numRows = np.ceil(numPlots / numColumns)
    return int(numRows), int(numColumns)


def plot_conditions_by_ROI(
    dF_Fs: Dict[str, np.ndarray],
    session: ImagingSession,
    title: str = "",
    figDims: Tuple[int, int] = (10, 9),
    palette: str = "Reds_d",
    maxSubPlots: int = 25,
) -> List[plt.Figure]:
    conditions = tuple(dF_Fs)
    sns.set(rc={"figure.figsize": figDims})
    sns.set_palette(palette, len(conditions))
    dF_Fs_all = np.stack(tuple(dF_Fs.values()), axis=-1)
    numROI = dF_Fs_all.shape[2]
    figs = []
    for i_fig in range(int(np.ceil(numROI / maxSubPlots))):
        ROIOffset = maxSubPlots * i_fig
        selectedROIs = list(range(0 + ROIOffset, min(maxSubPlots + ROIOffset, numROI)))
        fig = generate_conditions_by_ROI_plot(dF_Fs_all, session, title, selectedROIs)
        fig.legend(conditions, loc="lower right")
        figs.append(fig)
    return figs


def generate_conditions_by_ROI_plot(
    dF_Fs_all: np.ndarray,
    session: ImagingSession,
    title: str,
    selectedROIs: Sequence[int],
) -> plt.Figure:
    numROI = len(selectedROIs)
    numPlots = numROI
    layout = pick_layout(numPlots)
    fig, axarr = plt.subplots(layout[0], layout[1], sharex=True, squeeze=False)
    lockOffset = session.get_lock_offset()
    for i_ROI, roi in enumerate(selectedROIs):
        plotLocation = np.unravel_index(i_ROI, layout)
        axarr[plotLocation].title.set_text("ROI #" + str(roi) + ", " + title)
        plotData = np.nanmean(dF_Fs_all[:, :, roi, :], axis=1, keepdims=True)
        plot_dF_F_timeseries(axarr[plotLocation], lockOffset, np.squeeze(plotData))
        # Keep subplot axis labels only for edge plots; minimize figure clutter
        if plotLocation[1] > 0:
            axarr[plotLocation].set_ylabel("")
        if i_ROI < (numPlots - layout[1]):
            axarr[plotLocation].set_xlabel("")
    return fig


def plot_dF_F_timeseries(
    ax: plt.Axes, frameData: Sequence[int], plotData: np.ndarray, alpha=1.0
) -> None:
    ax.set_xlabel("Frame (ms)")
    ax.set_ylabel("dF/F")
    ax.plot(frameData, np.squeeze(plotData), alpha=alpha)
    ax.plot([frameData[0], frameData[-1]], [0, 0], "--k", zorder=-1)


def visualize_conditions(
    dF_Fs: Dict[str, np.ndarray],
    session: ImagingSession,
    axis: Union[int, Sequence[int]],
    title="",
    figDims=(10, 9),
    palette="Spectral",
) -> plt.Figure:
    """Generate plots of dF/F traces by condition.

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
    numConditions = len(dF_Fs)
    numPlots = numConditions
    layout = pick_layout(numPlots)
    lockOffset = session.get_lock_offset()
    sns.set(rc={"figure.figsize": figDims})
    numLines = max(np.mean(dF_Fs[list(dF_Fs)[0]], axis=axis, keepdims=True).shape[1:])
    sns.set_palette(palette, numLines)
    fig, axarr = plt.subplots(layout[0], layout[1], sharex=True, squeeze=False)
    i_condition = -1
    for condition, dF_f in dF_Fs.items():
        i_condition += 1
        plotLocation = np.unravel_index(i_condition, layout)
        plotData = np.nanmean(dF_Fs[condition], axis=axis, keepdims=True)
        axarr[plotLocation].title.set_text(condition + ", " + title)
        plot_dF_F_timeseries(axarr[plotLocation], lockOffset, np.squeeze(plotData))
        # Keep subplot axis labels only for edge plots; minimize figure clutter
        if plotLocation[1] > 0:
            axarr[plotLocation].set_ylabel("")
        if i_condition < (numPlots - layout[1]):
            axarr[plotLocation].set_xlabel("")
        if (numLines > 1) & (numLines < 10):
            # This assigns the conditions to the legend, even tho this if statement only
            # indirectly confirms this is appropriate.
            axarr[plotLocation].legend(
                session.trialGroups[condition].values, ncol=2, fontsize="xx-small"
            )
    plt.figtext(
        0.4, 0.93, "Legends indicate trial numbers.", style="italic", fontsize="small"
    )
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
