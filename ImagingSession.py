import numpy as np
import pandas as pd
import processROI


class ImagingSession:
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
    ):
        # self.trialAlignmentTimes = trialAlignmentTimes
        self.blockTemplate = blockTemplate
        self.numBlocks = numBlocks
        self.numTrials = numTrials
        self.preWindow = preWindow
        self.postWindow = postWindow

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
        self.set_frameWindows(self, frameTimestamps)

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
        self.trialAlignments = timelocks

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
