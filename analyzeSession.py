"""Analyze a 2P imaging session from tiff stack, h5 metadata, and ROI masks.
"""
import logging
import os
import sys
from typing import Callable, Dict, List, Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm, trange

import processROI
from ImagingSession import ImagingSession
from TiffStack import TiffStack

logger = logging.getLogger("analyzeSession")
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.DEBUG)


def process(
    tiffPattern: str, maskDir: str, h5Filename: str
) -> Tuple[Dict[str, np.ndarray], ImagingSession]:
    """Process imaging session.

    Assembles ImagingSession and dF/F traces.

    Arguments:
        tiffPattern {str} -- Path to tiff(s). Shell wildcard characters acceptable.
        maskDir {str} -- Path to directory containing .bmp ROI masks.
        h5Filename {str} -- Path to h5 metadata file.

    Returns:
        Tuple[Dict[str, np.ndarray], ImagingSession] -- dF/Fs, ImagingSession
    """
    timeseries = TiffStack(tiffPattern)
    timeseries.add_masks(maskDir)
    ROIaverages = timeseries.cut_to_averages()
    frameTriggers = processROI.get_flatten_trial_data(
        h5Filename, "frame_triggers", clean=True
    )
    trialsMeta = processROI.get_trials_metadata(h5Filename)
    session = ImagingSession(trialsMeta["inh_onset"], frameTriggers, h5Filename)
    meanFs = session.get_meanFs(ROIaverages, frameWindow=2)
    dF_Fs = {}
    for condition in meanFs:
        dF_Fs[condition] = session.get_trial_average_data(
            ROIaverages, meanFs[condition], condition
        )
    return dF_Fs, session


def pick_layout(numPlots: int) -> Tuple[int, int]:
    """Define gridsize to fit a number of subplots.

    Arguments:
        numPlots {int} -- Subplots to be arranged.

    Returns:
        Tuple[int, int] -- Subplot grid size.
    """
    numColumns = np.round(np.sqrt(numPlots))
    numRows = np.ceil(numPlots / numColumns)
    return int(numRows), int(numColumns)


def plot_downsampled_trials_by_ROI_per_condition(
    dF_Fs: Dict[str, np.ndarray],
    session: ImagingSession,
    downsampleAxis: int,
    downsampleFactor: int,
    title: str = "",
    supTitle: str = "",
    figDims: Tuple[int, int] = (10, 9),
    palette: str = "Reds_r",
    maxSubPlots: int = 25,
) -> List[plt.Figure]:
    """Plot downsampled trials for each ROI for each condition.

    Average groups of consecutive trials for plotting.

    Arguments:
        dF_Fs {Dict[str, np.ndarray]} -- Dict of condition: timeseries.
        session {ImagingSession} -- Session specifics.
        downsampleAxis {int} -- Axis of dF_Fs to be downsampled.
        downsampleFactor {int} -- This many trials along downsampleAxis in dF_Fs will be
            averaged together into one trial in the resultant plotted data.

    Keyword Arguments:
        title {str} -- Subplot title (default: {""})
        supTitle {str} -- Figure title. The condition will be suffixed to it. (default:
            {""})
        figDims {Tuple[int, int]} -- Size of figures in inches. (default: {(10, 9)})
        palette {str} -- Seaborn palette for plot lines. (default: {"Reds_d"})
        maxSubPlots {int} -- Max number of subplots per figure (default: {25})

    Returns:
        List[plt.Figure] -- List of figures produce.
    """
    dF_FsDownsampled: Dict[str, np.ndarray] = {}
    for condition, dF_F in dF_Fs.items():
        oldShape = dF_F.shape
        newShape = (
            oldShape[:downsampleAxis]
            + (oldShape[downsampleAxis] // downsampleFactor,)
            + oldShape[downsampleAxis + 1 :]
        )
        dF_FsDownsampled[condition] = processROI.downsample(dF_F, newShape)
    figs = plot_trials_by_ROI_per_condition(
        dF_FsDownsampled,
        session,
        title=title,
        supTitle=supTitle,
        figDims=figDims,
        palette=palette,
        maxSubPlots=maxSubPlots,
    )
    for fig in figs:
        for leg in fig.legends:
            leg.remove()
        numLines = sum([line.get_ls() == "-" for line in fig.axes[0].get_lines()])
        fig.legend(list(range(1, numLines + 1)), loc="lower right")
    return figs


def plot_trials_by_ROI_per_condition(
    dF_Fs: Dict[str, np.ndarray],
    session: ImagingSession,
    title: str = "",
    supTitle: str = "",
    figDims: Tuple[int, int] = (10, 9),
    palette: str = "Reds_d",
    maxSubPlots: int = 25,
) -> List[plt.Figure]:
    """Plot all trials for each ROI for each condition.

    Arguments:
        dF_Fs {Dict[str, np.ndarray]} -- Dict of condition: timeseries
        session {ImagingSession} -- Session specifics.

    Keyword Arguments:
        title {str} -- Subplot title (default: {""})
        supTitle {str} -- Figure title. The condition will be suffixed to it. (default:
            {""})
        figDims {Tuple[int, int]} -- Size of figures in inches. (default: {(10, 9)})
        palette {str} -- Seaborn palette for plot lines. (default: {"Reds_d"})
        maxSubPlots {int} -- Max number of subplots per figure (default: {25})

    Returns:
        List[plt.Figure] -- List of figures produce.
    """
    sns.set(rc={"figure.figsize": figDims})
    conditions = tuple(dF_Fs)
    figs = []
    for condition in conditions:
        dF_F = dF_Fs[condition]
        numTrials = dF_F.shape[1]
        sns.set_palette(palette, numTrials)
        numROI = dF_F.shape[2]
        for i_fig in range(int(np.ceil(numROI / maxSubPlots))):
            ROIOffset = maxSubPlots * i_fig
            selectedROIs = list(
                range(0 + ROIOffset, min(maxSubPlots + ROIOffset, numROI))
            )
            fig = create_ROI_plot(
                lambda roi: dF_F[:, :, roi],
                session.get_lock_offset(),
                session.odorCodesToNames,
                title,
                selectedROIs,
                alpha=0.8,
            )
            fig.legend(session.trialGroups[condition], loc="lower right")
            fig.suptitle(supTitle + f" for {condition}")
            figs.append(fig)
    return figs


def plot_conditions_by_ROI(
    dF_Fs: Dict[str, np.ndarray],
    session: ImagingSession,
    title: str = "",
    figDims: Tuple[int, int] = (10, 9),
    palette: str = "Reds_d",
    maxSubPlots: int = 25,
) -> List[plt.Figure]:
    """Plot trial-mean for each condition on each ROI.

    Arguments:
        dF_Fs {Dict[str, np.ndarray]} -- Dict of condition: timeseries
        session {ImagingSession} -- Session specifics.

    Keyword Arguments:
        title {str} -- Subplot title (default: {""})
        figDims {Tuple[int, int]} -- Size of figures in inches. (default: {(10, 9)})
        palette {str} -- Seaborn palette for plot lines. (default: {"Reds_d"})
        maxSubPlots {int} -- Max number of subplots per figure (default: {25})

    Returns:
        List[plt.Figure] -- List of figures produce.
    """
    conditions = tuple(dF_Fs)
    sns.set(rc={"figure.figsize": figDims})
    sns.set_palette(palette, len(conditions))
    dF_Fs_all = np.stack(tuple(dF_Fs.values()), axis=-1)
    numROI = dF_Fs_all.shape[2]
    figs = []
    for i_fig in range(int(np.ceil(numROI / maxSubPlots))):
        ROIOffset = maxSubPlots * i_fig
        selectedROIs = list(range(0 + ROIOffset, min(maxSubPlots + ROIOffset, numROI)))
        fig = create_ROI_plot(
            lambda roi: np.nanmean(dF_Fs_all[:, :, roi, :], axis=1, keepdims=True),
            session.get_lock_offset(),
            session.odorCodesToNames,
            title,
            selectedROIs,
        )
        fig.legend(conditions, loc="lower right")
        figs.append(fig)
    return figs


def create_ROI_plot(
    # plotDataGenerator's output's first dimension must match the length of frameAxis
    plotDataGenerator: Callable[[int], np.ndarray],
    frameAxis: Sequence[int],
    odorNames: Mapping[str, str],
    title: str,
    selectedROIs: Sequence[int],
    alpha=0.8,
    **plotKwargs,
) -> plt.Figure:
    """Helper function: creates figure with subplots for each selected ROI.

    Arguments:
        plotDataGenerator {Callable[[int], np.ndarray]} -- Handle to a function that
            accepts an integer ROI # and returns the corresponding subplot data (y-axis).
            (Example: (lambda roi: dF_F[:, :, roi]))
        frameAxis {Sequence[int]} -- Frame index (plot x-axis labels).
        odorNames {Mapping[str, str]} -- Mapping from odor code in condition names to full
            odor name.
        title {str} -- Subplot title. Appears after ROI #.
        selectedROIs {Sequence[int]} -- Indexes of the ROIs to plot

    Keyword Arguments:
        alpha {float} -- Transparency of plotted lines. (default: {0.8})
        **plotKwargs -- Any additional keyword arguments are passed on to plt.plot.

    Returns:
        plt.Figure -- Resultant figure.
    """
    numROI = len(selectedROIs)
    numPlots = numROI
    layout = pick_layout(numPlots)
    fig, axarr = plt.subplots(layout[0], layout[1], sharex=True, squeeze=False)
    for i_ROI, roi in enumerate(selectedROIs):
        plotLocation = np.unravel_index(i_ROI, layout)
        axarr[plotLocation].title.set_text("ROI #" + str(roi) + ", " + title)
        plotData = plotDataGenerator(roi)
        plot_dF_F_timeseries(
            axarr[plotLocation],
            frameAxis,
            np.squeeze(plotData),
            alpha=alpha,
            **plotKwargs,
        )
        # Keep subplot axis labels only for edge plots; minimize figure clutter
        if plotLocation[1] > 0:
            axarr[plotLocation].set_ylabel("")
        if i_ROI < (numPlots - layout[1]):
            axarr[plotLocation].set_xlabel("")
    subtitle("Odor key: " + f"{odorNames}".replace("'", ""))
    return fig


def plot_dF_F_timeseries(
    ax: plt.Axes, frameData: Sequence[int], plotData: np.ndarray, **kwargs
) -> None:
    """Plot dF/F timeseries with consistent format.

    Arguments:
        ax {plt.Axes} -- Axes to plot onto.
        frameData {Sequence[int]} -- X-axis data.
        plotData {np.ndarray} -- Y-axis data.

    Keyword Arguments:
        **plotKwargs -- Any keyword arguments are passed on to plt.plot.
    """
    ax.set_xlabel("Frame")
    ax.set_ylabel("dF/F")
    ax.plot(frameData, np.squeeze(plotData), **kwargs)
    ax.plot([frameData[0], frameData[-1]], [0, 0], "--k", zorder=-1)


def visualize_correlation(
    correlationsByROI: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    odorNames: Mapping[str, str],
    savePath: str = None,
    title: str = "",
    figDims: Tuple[int, int] = (10, 9),
    maxSubPlots: int = 16,
) -> List[plt.Figure]:
    sns.set(rc={"figure.figsize": figDims})
    numROI = len(correlationsByROI)
    assert len(masks) == numROI
    figs = []
    for i_fig in trange(int(np.ceil(numROI / maxSubPlots))):
        ROIOffset = maxSubPlots * i_fig
        selectedROIs = list(range(0 + ROIOffset, min(maxSubPlots + ROIOffset, numROI)))
        fig = plot_correlations_by_ROI(
            correlationsByROI, masks, odorNames, title, selectedROIs
        )
        figs.append(fig)
    if savePath:
        # with PdfPages(savePath + ".pdf") as pp:
        #     for fig in figs:
        #         pp.savefig(fig)
        for i_fig, fig in enumerate(figs):
            fig.savefig(savePath + " " + str(i_fig) + ".png")
            plt.close(fig)
    return figs


def plot_correlations_by_ROI(
    correlationsByROI: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    odorNames: Mapping[str, str],
    suptitle: str,
    selectedROIs: Sequence[int],
    clipCorrelation: float = 0.5,  # value to clip heatmap colorbar
    colormap=sns.diverging_palette(255, 0, sep=round(0.2 * 256), as_cmap=True),
) -> plt.Figure:
    numPlots = len(selectedROIs)
    layout = pick_layout(numPlots)
    fig, axarr = plt.subplots(
        layout[0], layout[1], sharex=True, sharey=True, squeeze=False
    )
    colorbar = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    for i_ROI, roi in enumerate(selectedROIs):
        plotLocation = np.unravel_index(i_ROI, layout)
        axarr[plotLocation].title.set_text("ROI #" + str(roi))
        plotData = correlationsByROI[roi]
        sns.heatmap(
            plotData,
            ax=axarr[plotLocation],
            cmap=colormap,
            vmin=-clipCorrelation,
            vmax=clipCorrelation,
            center=0,
            xticklabels=100,
            yticklabels=200,
            cbar=i_ROI == 0,
            cbar_ax=None if i_ROI else colorbar,
        )
        # Keep y-ticklabels horizontal
        axarr[plotLocation].set_yticklabels(
            axarr[plotLocation].get_yticklabels(), rotation=0
        )
        # Draw outline of ROI for reference
        axarr[plotLocation].contour(masks[roi], colors="black", linewidths=0.3)
    fig.suptitle(suptitle)
    subtitle("Odor key: " + f"{odorNames}".replace("'", ""))
    return fig


def visualize_conditions(
    dF_Fs: Dict[str, np.ndarray],
    session: ImagingSession,
    axis: Union[int, Sequence[int]],
    title="",
    figDims=(10, 9),
    palette="Reds_r",
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
        palette {str} -- Seaborn palette for plot lines. (default: {"Reds_r"})

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
    subtitle(
        "Legends indicate trial numbers. Odor key: "
        + f"{session.odorCodesToNames}".replace("'", "")
    )
    return fig


def subtitle(text: str) -> None:
    """Create a subtitle on the active plt.Figure.

        Position selected for a plt.subplots figure with Seaborn settings.

    Arguments:
        text {str} -- Subtitle text.
    """
    plt.figtext(
        0.5, 0.93, text, style="italic", fontsize="small", horizontalalignment="center"
    )


def process_and_viz_correlations(
    roiStackPattern: str,
    corStackPatterns: Sequence[str],
    maskDir: str,
    session: ImagingSession,
    savePath: str,
):
    """Process correlation tracing imaging session.

    Arguments:
        roiStackPattern {str} -- Path to tiffStack containing ROI to correlate against.
            Shell wildcard characters acceptable. This is fed directly to TiffStack
            constructor.
        corStackPatterns {Sequence[str]} -- Paths to each tiffStack to correlate against
            each ROI. Shell wildcard characters acceptable. This is fed directly to
            TiffStack constructor.
        maskDir {str} -- Path to directory containing .bmp ROI masks.
        session {ImagingSession} -- Framing details for the session.
    """
    roiStack = TiffStack(roiStackPattern)
    roiStack.add_masks(maskDir)
    roiAverages = roiStack.cut_to_averages()
    roiMeanFs = session.get_meanFs(roiAverages)
    roiDF_Fs = {}
    for condition in roiMeanFs:
        roiDF_Fs[condition] = session.get_trial_average_data(
            roiAverages, roiMeanFs[condition], condition
        )
    logger.debug(f"Tiff stack patterns to correlate against: {corStackPatterns}")
    for i_stack, tiffPattern in enumerate(tqdm(corStackPatterns, unit="stack")):
        stack = TiffStack(tiffPattern)
        pixelMeanFs = session.get_meanFs(stack.timeseries)
        assert list(roiMeanFs) == list(pixelMeanFs)
        for condition in tqdm(pixelMeanFs, unit="condition"):
            pixeldF_Fs = session.get_trial_average_data(
                stack.timeseries, pixelMeanFs[condition], condition
            )
            correlationsByROI = []
            numROI = roiDF_Fs[condition].shape[2]
            for i_roi in trange(numROI, unit="ROI"):
                # Average across trials
                roiTimeseries = np.nanmean(roiDF_Fs[condition][:, :, i_roi], axis=1)
                pixelsTimeseries = np.nanmean(pixeldF_Fs, axis=1)
                correlationsByROI.append(
                    processROI.pixelwise_correlate(pixelsTimeseries, roiTimeseries)
                )
            title = "stack" + str(i_stack) + "_" + condition
            visualize_correlation(
                correlationsByROI,
                roiStack.masks,
                session.odorCodesToNames,
                title=title,
                savePath=os.path.join(savePath, title),
            )
