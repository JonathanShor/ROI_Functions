#!/usr/bin/env python3
"""Analyze a 2P imaging session from tiff stack, h5 metadata, and ROI masks.
"""
import argparse
import logging
import os
import sys
import time
import warnings
from glob import glob
from typing import Callable, Dict, List, Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm, trange

import processROI
from ImagingSession import H5Session, ImagingSession
from TiffStack import TiffStack

logger = logging.getLogger("analyzeSession")
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.INFO)

SignalByCondition = Dict[str, np.ndarray]


def process_dF_Fs(timeseries: np.ndarray, session: ImagingSession) -> SignalByCondition:
    """Produce dF/F data for given signal and session details.

    The signals are broken into conditions and timelocked to them according to session.

    Args:
        timeseries (np.ndarray): Timeframes by trial/ROI/etc.
        session (ImagingSession): Defines what time windows belong to what conditions.

    Returns:
        Dict[str, np.ndarray]: condition: dF/F. dF/F is frames by trials by ROI.
    """
    meanFs = session.get_meanFs(timeseries, frameWindow=2)
    dF_Fs = {}
    for condition in meanFs:
        dF_Fs[condition] = session.get_trial_average_data(
            timeseries, meanFs[condition], condition
        )
    return dF_Fs


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
) -> List[plt.Figure]:
    sns.set(rc={"figure.figsize": figDims})
    avgRoiSize = np.median(
        [
            np.sum(mask) / np.prod(correlationsByROI[i_mask].shape)
            for i_mask, mask in enumerate(masks)
        ]
    )
    gridSizes = [x ** 2 for x in range(1, 6)]
    maxSubPlots = max(
        filter(
            lambda gridSize: (100 * np.prod(figDims) / 2 / gridSize * avgRoiSize) > 0.5,
            gridSizes,
        )
    )
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
        logger.info(f"{str(i_fig)} figures saved to {savePath}.")
    return figs


def plot_correlations_by_ROI(
    correlationsByROI: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    odorNames: Mapping[str, str],
    suptitle: str,
    selectedROIs: Sequence[int],
    clipCorrelation: float = 1.0,  # value to clip heatmap colorbar
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
        axarr[plotLocation].contour(
            masks[roi], colors="black", alpha=0.7, linestyle="dashed", linewidths=0.3
        )
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
    sharey=False,
    showLegend=True,
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
    fig, axarr = plt.subplots(
        layout[0], layout[1], sharex=True, sharey=sharey, squeeze=False
    )
    i_condition = -1
    for condition, dF_f in dF_Fs.items():
        i_condition += 1
        plotLocation = np.unravel_index(i_condition, layout)
        with warnings.catch_warnings():
            # TODO: Catch and log numpy all-NAN warnings, instead of ignore
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            plotData = np.nanmean(dF_Fs[condition], axis=axis, keepdims=True)
        axarr[plotLocation].title.set_text(condition + ", " + title)
        plot_dF_F_timeseries(axarr[plotLocation], lockOffset, np.squeeze(plotData))
        # Keep subplot axis labels only for edge plots; minimize figure clutter
        if plotLocation[1] > 0:
            axarr[plotLocation].set_ylabel("")
        if i_condition < (numPlots - layout[1]):
            axarr[plotLocation].set_xlabel("")
        if ((numLines > 1) & (numLines < 10)) and showLegend:
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
    roiDF_Fs: SignalByCondition,
    roiMasks: Sequence[np.ndarray],
    corrStack: TiffStack,
    corrSession: ImagingSession,
    savePath: str,
    window: slice = None,
):
    """Process correlation tracing imaging session.

    Arguments:
        roiDF_Fs {Dict[str, ndarray]} -- Condition: dF/F trace.
        roiMasks {Sequence[np.ndarray]} -- Boorlean masks for the dF/F traces
            corresponding to each ROI.
        corrStack {TiffStack} -- TiffStack to correlate against each ROI.
        corrSession {ImagingSession} -- Framing details for the session to correlate.
        savePath {str} -- Path at which to save figures.

    Keyword Arguments:
        window {slice} -- Correlation will only apply within given window. Default uses
            full window as defined by corrSession. (default: {None})
    """
    pixelMeanFs = corrSession.get_meanFs(corrStack.timeseries)
    assert list(roiDF_Fs) == list(pixelMeanFs), "ref and target conditions do not match"
    for condition in tqdm(pixelMeanFs, unit="condition"):
        pixeldF_Fs = corrSession.get_trial_average_data(
            corrStack.timeseries, pixelMeanFs[condition], condition
        )
        correlationsByROI = []
        numROI = roiDF_Fs[condition].shape[2]
        for i_roi in trange(numROI, desc=f"{condition}", unit="ROI"):
            # Average across trials
            roiTimeseries = np.nanmean(roiDF_Fs[condition][:, :, i_roi], axis=1)
            with warnings.catch_warnings():
                # TODO: Catch and log numpy all-NAN warnings, instead of ignore
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                pixelsTimeseries = np.nanmean(pixeldF_Fs, axis=1)
            if np.isnan(pixelsTimeseries).all():
                logger.warning(
                    f"During session {corrSession.title}, ROI#{i_roi}"
                    + ", pixelsTimeseries was *all* NAN."
                )
            if window:
                assert len(roiTimeseries) == 60
                roiTimeseries = roiTimeseries[window]
                assert pixelsTimeseries.shape[0] == 60
                pixelsTimeseries = pixelsTimeseries[window]
            correlationsByROI.append(
                processROI.pixelwise_correlate(pixelsTimeseries, roiTimeseries)
            )
        title = "stack" + str(corrSession.title) + "_" + condition
        visualize_correlation(
            correlationsByROI,
            roiMasks,
            corrSession.odorCodesToNames,
            title=title,
            savePath=os.path.join(savePath, title),
        )


def run_condition_visualization(
    dF_Fs: SignalByCondition,
    session: ImagingSession,
    axis: Union[int, Sequence[int]],
    title: str,
) -> plt.Figure:
    fig = visualize_conditions(dF_Fs, session, axis=axis, title=title)
    fig.suptitle(title)
    return fig


def get_best_ROI(dF_F: np.ndarray, numReturned: int) -> np.ndarray:
    """Return bool mask corresponding to highest signal ROIs.

    Means across trials, then sums absolute value of each frame to evaluate.

    Args:
        dF_F (np.ndarray): Frames by trials by ROI.
        numReturned (int, optional): Number of ROI to return. Defaults to 10.

    Returns:
        np.ndarray: Bool mask with shape = dF_F.shape[2:].
    """
    strengths = np.sum(np.abs(np.mean(dF_F, axis=1)), axis=0)
    roiShape = strengths.shape
    bestMask = np.full(roiShape, False, dtype=np.bool)
    bestMask[
        np.unravel_index(np.argsort(strengths, axis=None)[-numReturned:], roiShape)
    ] = True
    return bestMask


def plot_allROI_best_performers(
    dF_Fs: SignalByCondition, session: ImagingSession, roiPerPlot: int = 5
) -> List[plt.Figure]:
    figs = []
    for condition, dF_F in tqdm(
        dF_Fs.items(), desc="All ROI, best performers", unit="fig"
    ):
        bestROI = get_best_ROI(dF_Fs[condition], numReturned=roiPerPlot)
        title = f"Top {roiPerPlot}"
        bestOnlyDF_Fs = {
            condition: dF_F[:, :, bestROI] for condition, dF_F in dF_Fs.items()
        }
        fig = visualize_conditions(
            bestOnlyDF_Fs,
            session,
            axis=1,
            title=title,
            palette="Spectral",
            sharey=True,
            showLegend=False,
        )
        fig.suptitle(f"Most Active {roiPerPlot} ROI for {condition}")
        figs.append(fig)
    return figs


def save_figs(
    figs: Sequence[plt.Figure],
    title: str,
    saveTo: str,
    figFileType: str,
    unit: str = "figure",
    setSuptitle: bool = False,
) -> None:
    padding = 2 if (len(figs) > 9) else 1
    for i_fig, fig in enumerate(tqdm(figs, unit=unit)):
        if setSuptitle:
            fig.suptitle(title)
        figID = ("{:0" + str(padding) + "d}").format(i_fig)
        figFname = saveTo + title.replace(" ", "_") + f"_{figID}.{figFileType}"
        fig.savefig(figFname)
        logger.debug(f"Saved {figFname}.")
        plt.close(fig)


def launch_traces(
    h5Filename: str,
    tiffFilenames: Sequence[str],
    maskFilenames: Sequence[str],
    saveDir: str,
    savePrefix: str,
    figFileType: str = "png",
    **kwargs,
) -> None:
    refStack = TiffStack(tiffFilenames, maskFilenames=maskFilenames)
    roiAverages = refStack.cut_to_averages()
    refSession = H5Session(h5Filename)
    dF_Fs = process_dF_Fs(roiAverages, refSession)
    os.makedirs(saveDir, exist_ok=True)
    saveTo = os.path.join(saveDir, savePrefix.replace(" ", "_"))

    figSettings: List[Tuple[Union[int, Sequence[int]], str]] = []
    if kwargs["allROI"] or kwargs["allFigs"]:
        numROI = roiAverages.shape[1]
        if numROI <= 15:
            figSettings.append((1, "All ROI"))
        else:
            figs = plot_allROI_best_performers(dF_Fs, refSession)
            save_figs(figs, "allROI, Best Performers", saveTo, figFileType)
    if kwargs["replicants"] or kwargs["allFigs"]:
        figSettings.append((2, str(list(dF_Fs.values())[0].shape[1]) + " Replicants"))
    if kwargs["crossMean"] or kwargs["allFigs"]:
        figSettings.append(((1, 2), "Cross-trial Mean"))
    for axis, title in tqdm(figSettings, unit="meanFig"):
        fig = run_condition_visualization(dF_Fs, refSession, axis, title)
        figFname = saveTo + title.replace(" ", "_") + "." + figFileType
        fig.savefig(figFname)
        logger.debug(f"Saved {figFname}.")
        plt.close(fig)
    if figSettings:
        logger.info("Mean trace figures done.")

    # TODO: Somehow unify this all into same single figSettings loop
    if kwargs["condsROI"] or kwargs["allFigs"]:
        figs = plot_conditions_by_ROI(dF_Fs, refSession)
        title = "Conditions by ROI"
        save_figs(figs, title, saveTo, figFileType, unit="condROIFig", setSuptitle=True)
        logger.info(f"Conditions by ROI figures done. {len(figs)} figures produced.")

    if kwargs["ROIconds"] or kwargs["allFigs"]:
        title = "All ROI by Condition"
        figs = plot_trials_by_ROI_per_condition(
            dF_Fs, refSession, supTitle=title, maxSubPlots=16
        )
        save_figs(figs, title, saveTo, figFileType, unit="ROIcondsFig")
        logger.info(f"ROIs per condition figures done. {len(figs)} figures produced.")


def launch_correlation(
    h5Filename: str,
    tiffFilenames: Sequence[str],
    maskFilenames: Sequence[str],
    saveDir: str,
    savePrefix: str,
    corrPatternsFile: str,
    corrH5sFile: str,
    squashConditions: bool = False,
) -> None:
    refSession = H5Session(h5Filename, unified=squashConditions)
    refStack = TiffStack(tiffFilenames, maskFilenames=maskFilenames)
    refDF_Fs = process_dF_Fs(refStack.cut_to_averages(), refSession)
    saveTo = os.path.join(saveDir, savePrefix)
    os.makedirs(saveTo, exist_ok=True)
    corrH5s = read_h5s_file(corrH5sFile)
    corrPatterns = read_stack_patterns_file(corrPatternsFile)
    logger.debug(f"Starting correlation analysis. Ref: {refStack._tiffFilenames[0]}")
    for i_stack, corrStackPattern in enumerate(tqdm(corrPatterns, unit="correlation")):
        logger.debug(f"Loading metadata from {corrH5s[i_stack]}")
        corrSession = H5Session(
            corrH5s[i_stack], title=str(i_stack), unified=squashConditions
        )
        corrStack = TiffStack(sorted(glob(corrStackPattern)))
        process_and_viz_correlations(
            refDF_Fs, refStack.masks, corrStack, corrSession, saveTo
        )


def read_h5s_file(filename: str) -> List[str]:
    return read_stack_patterns_file(filename)


def read_stack_patterns_file(filename: str) -> List[str]:
    patterns: List[str] = []
    with open(filename, mode="r") as patternsFile:
        for line in patternsFile:
            patterns.append(line)
    return patterns


def get_common_parser():
    commonParser = argparse.ArgumentParser(add_help=False)
    # commonParser.add_argument(
    #     "--cluster",
    #     action="store_true",
    #     help="Execute via submission script to the cluster.",
    # )
    commonParser.add_argument(
        "--h5",
        dest="h5Filename",
        required=True,
        help="H5 file containing reference session metadata.",
    )
    commonParser.add_argument(
        "-T",
        "--tiffs",
        dest="tiffFilenames",
        nargs="+",
        required=True,
        metavar="tiffile",
        help="List of tiff filenames, in order, that form reference stack.",
    )
    commonParser.add_argument(
        "-M",
        "--masks",
        dest="maskFilenames",
        nargs="+",
        required=True,
        metavar="maskfile",
        help="List of ROI mask .bmp files.",
    )
    commonParser.add_argument("--saveDir", default="figures/")
    commonParser.add_argument(
        "--savePrefix", default="", help="Figure filename prefix."
    )
    return commonParser


if __name__ == "__main__":
    startTime = time.time()
    commonParser = get_common_parser()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    tracesParser = subparsers.add_parser(
        "traces", parents=[commonParser], help="Produce trace plots."
    )
    tracesParser.set_defaults(func=launch_traces)
    vizTracersGroup = tracesParser.add_argument_group("visualization")
    vizTracersGroup.add_argument(
        "--allROI", action="store_true", help="Produce All ROI plot."
    )
    vizTracersGroup.add_argument(
        "--replicants", action="store_true", help="Produce Replicants plot."
    )
    vizTracersGroup.add_argument(
        "--crossMean", action="store_true", help="Produce Cross-trial Mean plot."
    )
    vizTracersGroup.add_argument(
        "--condsROI", action="store_true", help="Produce Conditions by ROI plots."
    )
    vizTracersGroup.add_argument(
        "--ROIconds", action="store_true", help="Produce ROI per Conditions plots."
    )
    vizTracersGroup.add_argument(
        "--all", "-A", action="store_true", dest="allFigs", help="Produce all plots."
    )

    correlationParser = subparsers.add_parser(
        "correlation",
        parents=[commonParser],
        help="Produce correlation maps between tiff stacks.",
    )
    correlationParser.add_argument(
        "--corrPatternsFile",
        help="File cantaining filepath patterns (i.e. /path/to/Run0034Ref_00*.tif) for "
        + "each tiff stack to correlate, one per line. Order must match corrH5sFile.",
    )
    correlationParser.add_argument(
        "--corrH5sFile",
        help="File cantaining filepaths for each H5, one per line. Order must match "
        + "corrPatternsFile.",
    )
    correlationParser.set_defaults(func=launch_correlation)

    args = parser.parse_args()
    logger.debug(args)
    args.func(**vars(args))
    logger.info(f"Total run time: {time.time() - startTime:.2f} sec")
