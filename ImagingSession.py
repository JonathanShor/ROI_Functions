"""Manages imaging session details.
"""
import numpy as np
import pandas as pd
import processROI
from TiffStack import TiffStack


class ImagingSession:
    """[summary]

    Parameters:
        trialAlignmentTimes {pd.Series} --
        frameTimestamps {} --
        numBlocks {int} -- (default: {3})
        numTrials {int} --  (default: {45})
        blockTemplate {dict} --   (default: {
            "100% A, 0% B": [2, 15],
            "90% A, 10% B": [3, 14],
            "75% A, 25% B": [4, 13],
            "50% A, 50% B": [5, 12],
            "25% A, 75% B": [6, 11],
            "10% A, 90% B": [7, 10],
            "0% A, 100% B": [8, 9],
        })
        preWindow {int} --  (default: {500})
        postWindow {int} --  (default: {1500})
        timeseries -- If provided, should be either a tiffPattern to create a new
            TifFStack object, or should be a TiffStack object itself.

    Attributes:
        lockFrames {pd.DataFrame} -- List of frame indexes to mean around for each
            condition.
    """

    def __init__(
        self,
        trialAlignmentTimes,  # Must be pd.Series
        frameTimestamps,
        numBlocks=3,
        numTrials=45,
        blockTemplate={
            "100% A, 0% B": [2, 15],
            "90% A, 10% B": [3, 14],
            "75% A, 25% B": [4, 13],
            "50% A, 50% B": [5, 12],
            "25% A, 75% B": [6, 11],
            "10% A, 90% B": [7, 10],
            "0% A, 100% B": [8, 9],
        },
        preWindow=500,
        postWindow=1500,
        timeseries=None,
    ):
        # self.trialAlignmentTimes = trialAlignmentTimes
        self.blockTemplate = blockTemplate
        self.numBlocks = numBlocks
        self.numTrials = numTrials
        self.preWindow = preWindow
        self.postWindow = postWindow

        try:
            self.timeseries = TiffStack(timeseries)
        except TypeError:
            self.timeseries = timeseries

        trialGroups = {
            condition: (
                np.tile(value, (numBlocks, 1))
                + (np.arange(numBlocks) * numTrials // numBlocks).reshape(numBlocks, 1)
            ).flatten()
            - 1
            for condition, value in blockTemplate.items()
        }
        self.trialGroups = pd.DataFrame(data=trialGroups)
        self.set_timestamps(trialAlignmentTimes)
        self.set_frameWindows(frameTimestamps)

    def set_timestamps(self, timelocks):
        trialPreTimelocks = pd.DataFrame()
        trialPostTimelocks = pd.DataFrame()
        trialTimelocks = pd.DataFrame()
        for condition in self.trialGroups:
            trialPreTimelocks[condition] = (
                timelocks.iloc[self.trialGroups[condition]].values - self.preWindow
            )
            trialPostTimelocks[condition] = (
                timelocks.iloc[self.trialGroups[condition]].values + self.postWindow
            )
            trialTimelocks[condition] = timelocks.iloc[
                self.trialGroups[condition]
            ].values
        self.preTrialTimestamps = trialPreTimelocks
        self.postTrialTimestamps = trialPostTimelocks
        self.trialAlignments = trialTimelocks

    def set_frameWindows(self, frameTriggers):
        self.preFrames = pd.DataFrame()
        self.lockFrames = pd.DataFrame()
        self.postFrames = pd.DataFrame()
        for condition in self.trialGroups:
            self.preFrames[condition] = processROI.frame_from_timestamp(
                frameTriggers, self.preTrialTimestamps[condition]
            )
            self.lockFrames[condition] = processROI.frame_from_timestamp(
                frameTriggers, self.postTrialTimestamps[condition]
            )
            self.postFrames[condition] = processROI.frame_from_timestamp(
                frameTriggers, self.trialAlignments[condition]
            )

    def get_meanFs(self, ROIaverages, frameWindow=2):
        """Calculate mean signal in a frameWindow around each lockFrame.

        Arguments:
            ROIaverages {ndarray} -- 2D, frames by ROI, containing signal.

        Keyword Arguments:
            frameWindow {int} -- Width of window. (default: {2})

        Returns:
            pd.DataFrame -- 2D, lockFrames by ROI, containing mean signal.
        """
        meanFs = {}
        for condition in self.lockFrames:
            meanF = [
                np.mean(
                    ROIaverages[
                        max(0, lockFrame - frameWindow) : lockFrame + frameWindow, :
                    ],
                    axis=0,
                )
                for lockFrame in self.lockFrames[condition]
            ]
            meanFs[condition] = np.array(meanF)
        return meanFs

    def get_trial_average_data(self, ROIaverages, meanF, condition):
        numROI = ROIaverages.shape[1]
        numTrials = len(self.lockFrames[condition])
        self.maxSliceWidth = max(
            [
                post - pre
                for pre, post in zip(
                    self.preFrames[condition], self.postFrames[condition]
                )
            ]
        )
        preWindows = [
            slice(pre, lock)
            for pre, lock in zip(self.preFrames[condition], self.lockFrames[condition])
        ]
        self.zeroFrame = max(window.stop - window.start for window in preWindows)
        trialAverageData = np.zeros((self.maxSliceWidth, numTrials, numROI))
        for i_slice, slce in enumerate(preWindows):
            tempData = ROIaverages[slce, :]
            trialAverageData[
                self.zeroFrame - tempData.shape[0] : self.zeroFrame, i_slice, :
            ] = tempData

        postWindows = [
            slice(lock, post)
            for lock, post in zip(
                self.lockFrames[condition], self.postFrames[condition]
            )
        ]
        for i_slice, slce in enumerate(postWindows):
            tempData = ROIaverages[slce, :]
            trialAverageData[
                self.zeroFrame : self.zeroFrame + tempData.shape[0], i_slice, :
            ] = tempData

        dF_F = trialAverageData / meanF.reshape(1, numTrials, numROI) - 1
        dF_F[dF_F == -1] = np.nan
        return dF_F

    def get_lock_offset(self):
        lockOffset = list(range(-self.zeroFrame, self.maxSliceWidth - self.zeroFrame))
        return lockOffset
