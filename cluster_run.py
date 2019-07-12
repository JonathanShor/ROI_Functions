import logging
import os
import sys
from dataclasses import dataclass
from typing import Sequence

from tqdm import tqdm

import analyzeSession
import debugtoolbox
import processROI
from ImagingSession import ImagingSession

logger = logging.getLogger("cluster_run")
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.addHandler(debugtoolbox.get_FileHandler())
logger.setLevel(logging.DEBUG)


@dataclass
class TrialFilePath:
    refPattern: str
    maskDir: str
    h5Filename: str
    corPatterns: Sequence[str]
    saveDir: str


ROOTPATH = "/gpfs/scratch/jds814/2P-data/HN1953/"
trialFilePaths = {
    "190701_field2": TrialFilePath(
        os.path.join(ROOTPATH, "190701/aligned", "HN1953_190701_field2_00001_000*.tif"),
        os.path.join(ROOTPATH, "190701/aligned", "field2_masks/*.bmp"),
        os.path.join(ROOTPATH, "190701", "1953_1_01_D2019_7_1T11_52_7_odor.h5"),
        [
            os.path.join(
                ROOTPATH,
                "190701/aligned",
                "HN1953_190701_field" + str(i_field) + "_00001_000*.tif",
            )
            for i_field in range(2, 12)
        ],
        os.path.join(ROOTPATH, "190701/figures/field2"),
    ),
    "190701_field11": TrialFilePath(
        os.path.join(
            ROOTPATH, "190701/aligned", "HN1953_190701_field11_00001_000*.tif"
        ),
        os.path.join(ROOTPATH, "190701/aligned", "field11_masks/*.bmp"),
        os.path.join(ROOTPATH, "190701", "1953_1_10_D2019_7_1T13_48_29_odor.h5"),
        [
            os.path.join(
                ROOTPATH,
                "190701/aligned",
                "HN1953_190701_field" + str(i_field) + "_00001_000*.tif",
            )
            for i_field in range(11, 1, -1)
        ],
        os.path.join(ROOTPATH, "190701/figures/field11"),
    ),
}


def test_process_and_viz_correlations(trialFilePath):
    frameTriggers = processROI.get_flatten_trial_data(
        trialFilePath.h5Filename, "frame_triggers", clean=True
    )
    trialsMeta = processROI.get_trials_metadata(trialFilePath.h5Filename)
    session = ImagingSession(
        trialsMeta["inh_onset"], frameTriggers, trialFilePath.h5Filename
    )
    roiStackPattern = trialFilePath.refPattern
    maskDir = trialFilePath.maskDir
    savePath = trialFilePath.saveDir
    os.makedirs(savePath, exist_ok=True)
    analyzeSession.process_and_viz_correlations(
        roiStackPattern, trialFilePath.corPatterns, maskDir, session, savePath
    )


if __name__ == "__main__":
    for paths in tqdm(trialFilePaths.values(), unit="field"):
        test_process_and_viz_correlations(paths)
