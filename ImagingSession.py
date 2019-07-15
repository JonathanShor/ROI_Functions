"""Manages imaging session processing.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import processROI

IGNORE_ODORS = ["None", "empty"]


class ImagingSession:
    """Manages imaging session and timelock alignment.

    Parameters:
        trialAlignmentTimes {pd.Series} -- Timestamps at which to align each trial.
        frameTimestamps {Sequence[float]} -- Timestamps of each frame in tiffstack.
        h5Filename {str} -- Full filepath of h5 metadata for session.

    Attributes:
        preWindowSize {int} -- Size of pre-timelock averaging window in milliseconds.
            (default: {500})
        postWindowSize {int} -- Size of post-timelock averaging window in milliseconds.
            (default: {1500})
        title {str} -- Title addendum for figures, if provided. (default: {""})
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
        h5Filename: str,
        preWindowSize: int = 500,
        postWindowSize: int = 1500,
        title: str = "",
    ) -> None:
        self.preWindowSize = preWindowSize
        self.postWindowSize = postWindowSize
        self.title = title

        sessionData = processROI.get_trials_metadata(h5Filename)
        odorCodeGenerator = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.odorCodesToNames: Dict[str, str] = {}
        self.odorNamesToCodes: Dict[str, str] = {}
        trialGroups: Dict[str, List[int]] = {}
        for i_trial, trial in sessionData.iterrows():
            stimuli = []
            odors = [trial["olfas:olfa_0:odor"], trial["olfas:olfa_1:odor"]]
            for i_odor, odor in enumerate(odors):
                if odor in IGNORE_ODORS:
                    continue
                if odor not in self.odorCodesToNames.values():
                    nextCode = next(odorCodeGenerator)
                    self.odorCodesToNames[nextCode] = odor
                    self.odorNamesToCodes[odor] = nextCode
                stimuli.append(
                    (
                        trial["olfas:olfa_" + str(i_odor) + ":mfc_1_flow"],
                        self.odorNamesToCodes[odor],
                    )
                )
            condition = self.standardize_condition_text(stimuli)
            if condition:
                trialGroups[condition] = trialGroups.get(condition, []) + [i_trial]
        self.trialGroups = pd.DataFrame(data=trialGroups)
        self._set_timestamps(trialAlignmentTimes)
        self._set_frameWindows(frameTimestamps)

    @staticmethod
    def standardize_condition_text(stimuli: Sequence[Tuple[float, str]]) -> str:
        """Produce consistent condition text from odor name and flow rates.

        Examples:
        standardize_condition_text([(1.8, 'A'), (0.2,'B')]) -> "90% A, 10% B"
        standardize_condition_text([(0.9, 'A')]) -> "100% A"
        standardize_condition_text([]) -> ""

        Arguments:
            stimuli {Sequence[Tuple[float, str]]} -- Sequence of each presented stimuli as
                a tuple of odor flow rate and name.

        Returns:
            str -- [description]
        """
        flowSum = sum([stimulus[0] for stimulus in stimuli])
        conditions = [
            "{}% ".format(round(100 * stimulus[0] / flowSum)) + stimulus[1]
            for stimulus in stimuli
        ]
        return ",".join(conditions)

    def _set_timestamps(self, timelocks: pd.DataFrame) -> None:
        trialPreTimelocks = pd.DataFrame()
        trialPostTimelocks = pd.DataFrame()
        trialTimelocks = pd.DataFrame()
        for condition in self.trialGroups:
            trialPreTimelocks[condition] = (
                timelocks.iloc[self.trialGroups[condition]].values - self.preWindowSize
            )
            trialPostTimelocks[condition] = (
                timelocks.iloc[self.trialGroups[condition]].values + self.postWindowSize
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
        shapeROI = ROIaverages.shape[1:]
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
        trialAverageData = np.zeros((self.maxSliceWidth, numTrials) + shapeROI)
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

        dF_F = trialAverageData / meanF.reshape((1, numTrials) + shapeROI) - 1
        dF_F[np.isinf(dF_F)] = np.nan
        dF_F[dF_F == -1] = np.nan
        return dF_F

    def get_lock_offset(self) -> List[int]:
        """Index of frame offsets. Useful as x-axis input for timeseries plots.

        Returns:
            List[int] -- List of each frame's offset from zeroFrame.
        """
        lockOffset = list(range(-self.zeroFrame, self.maxSliceWidth - self.zeroFrame))
        return lockOffset


class H5Session(ImagingSession):
    def __init__(
        self,
        h5Filename: str,
        preWindowSize: int = 500,
        postWindowSize: int = 1500,
        title: str = "",
    ):
        frameTimestamps = processROI.get_flatten_trial_data(
            h5Filename, "frame_triggers", clean=True
        )
        trialAlignmentTimes = processROI.get_trials_metadata(h5Filename)["inh_onset"]
        return super().__init__(
            trialAlignmentTimes,
            frameTimestamps,
            h5Filename,
            preWindowSize=preWindowSize,
            postWindowSize=postWindowSize,
            title=title,
        )
