"""Manages imaging session processing.
"""
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import processROI
from SessionDetails import SessionDetails


class ImagingSession:
    """Manages imaging session and timelock alignment.

    Parameters:
        trialAlignmentTimes {pd.Series} -- Timestamps at which to align each trial.
        frameTimestamps {Sequence[float]} -- Timestamps of each frame in tiffstack.
        sessionDetails {SessionDetails} -- Metadata for session.

    Attributes:
        lockFrames {pd.DataFrame} -- List of frame indexes to mean around for each
            condition.
        preTrialTimestamps {pd.DataFrame} -- Timestamp for start of averaging windows of
            each trial. Grouped by condition.
        postTrialTimestamps {pd.DataFrame} -- Timestamp for end of averaging windows of
            each trial. Grouped by condition.
        trialAlignments {pd.DataFrame} -- Timestamp for timelock frame of each trial.
            Grouped by condition.
        preFrames {pd.DataFrame} -- Tiffstack frame corresponding to start of averaging
            windows for each trial. Grouped by condition.
        lockFrames {pd.DataFrame} -- Tiffstack frame corresponding to timelock of each
            trial. Grouped by condition.
        postFrames {pd.DataFrame} -- Tiffstack frame corresponding to end of averaging
            windows for each trial. Grouped by condition.
        maxSliceWidth {int} -- Widest averaging window in frames.
        zeroFrame {int} -- The timelock frame's index in the averaging window domain
            (aka range(maxSliceWidth)).
    """

    def __init__(
        self,
        trialAlignmentTimes: pd.Series,
        frameTimestamps: Sequence[float],
        sessionDetails: SessionDetails,
    ) -> None:
        self.sessionDetails = sessionDetails

        trialGroups = {
            condition: (
                np.tile(value, (sessionDetails.numCycles, 1))
                + (
                    np.arange(sessionDetails.numCycles)
                    * sessionDetails.numTrialsPerCycles
                ).reshape(sessionDetails.numCycles, 1)
            ).flatten()
            - 1
            for condition, value in sessionDetails.cycleTemplate.items()
        }
        self.trialGroups = pd.DataFrame(data=trialGroups)
        self._set_timestamps(trialAlignmentTimes)
        self._set_frameWindows(frameTimestamps)

    def _set_timestamps(self, timelocks: pd.DataFrame) -> None:
        trialPreTimelocks = pd.DataFrame()
        trialPostTimelocks = pd.DataFrame()
        trialTimelocks = pd.DataFrame()
        for condition in self.trialGroups:
            trialPreTimelocks[condition] = (
                timelocks.iloc[self.trialGroups[condition]].values
                - self.sessionDetails.preWindow
            )
            trialPostTimelocks[condition] = (
                timelocks.iloc[self.trialGroups[condition]].values
                + self.sessionDetails.postWindow
            )
            trialTimelocks[condition] = timelocks.iloc[
                self.trialGroups[condition]
            ].values
        self.preTrialTimestamps = trialPreTimelocks
        self.postTrialTimestamps = trialPostTimelocks
        self.trialAlignments = trialTimelocks

    def _set_frameWindows(self, frameTriggers: Sequence[float]) -> None:
        self.preFrames = pd.DataFrame()
        self.lockFrames = pd.DataFrame()
        self.postFrames = pd.DataFrame()
        for condition in self.trialGroups:
            self.preFrames[condition] = processROI.frame_from_timestamp(
                frameTriggers, self.preTrialTimestamps[condition]
            )
            self.lockFrames[condition] = processROI.frame_from_timestamp(
                frameTriggers, self.trialAlignments[condition]
            )
            self.postFrames[condition] = processROI.frame_from_timestamp(
                frameTriggers, self.postTrialTimestamps[condition]
            )

    def get_meanFs(
        self, ROIaverages: np.ndarray, frameWindow: int = 2
    ) -> Dict[str, np.ndarray]:
        """Calculate mean signal in a frameWindow around each lockFrame.

        Arguments:
            ROIaverages {ndarray} -- 2D, frames by ROI, containing signal.

        Keyword Arguments:
            frameWindow {int} -- Width of window. (default: {2})

        Returns:
            {str: 2D ndarray} -- 2D, lockFrames by ROI, containing mean signal.
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

    def get_trial_average_data(
        self, ROIaverages: np.ndarray, meanF: np.ndarray, condition: str
    ) -> np.ndarray:
        numROI = ROIaverages.shape[1]
        numTrials = len(self.lockFrames[condition])
        self.maxSliceWidth = max(
            post - pre
            for pre, post in zip(self.preFrames[condition], self.postFrames[condition])
        )
        preWindows = [
            slice(pre, lock)
            for pre, lock in zip(self.preFrames[condition], self.lockFrames[condition])
        ]
        self.zeroFrame = max(
            lock - pre
            for pre, lock in zip(self.preFrames[condition], self.lockFrames[condition])
        )
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

    def get_lock_offset(self) -> List[int]:
        """Index of frame offsets. Useful as x-axis input for timeseries plots.

        Returns:
            List[int] -- List of each frame's offset from zeroFrame.
        """
        lockOffset = list(range(-self.zeroFrame, self.maxSliceWidth - self.zeroFrame))
        return lockOffset
