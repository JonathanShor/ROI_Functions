import logging
import os
import sys
from dataclasses import dataclass
from typing import Sequence

from tqdm import tqdm

import analyzeSession
import debugtoolbox
from ImagingSession import H5Session
from TiffStack import TiffStack

logger = logging.getLogger("cluster_run")
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.addHandler(debugtoolbox.get_FileHandler())
logger.setLevel(logging.DEBUG)


@dataclass
class TrialFilePath:
    refPattern: str
    maskDir: str
    h5Filename: str
    saveDir: str


@dataclass
class CorrTrialFilePath:
    refPattern: str
    maskDir: str
    refH5Filename: str
    corrH5Filenames: Sequence[str]
    corrPatterns: Sequence[str]
    saveDir: str


ROOTPATH = "/gpfs/scratch/jds814/2P-data/HN1953/"
trialFilePaths = {
    "190701_field2": CorrTrialFilePath(
        os.path.join(ROOTPATH, "190701/aligned", "HN1953_190701_field2_00001_000*.tif"),
        os.path.join(ROOTPATH, "190701/aligned", "field2_masks/*.bmp"),
        os.path.join(ROOTPATH, "190701", "1953_1_01_D2019_7_1T11_52_7_odor.h5"),
        [
            os.path.join(ROOTPATH, "190701", h5)
            for h5 in [
                "1953_1_01_D2019_7_1T11_52_7_odor.h5",
                "1953_1_02_D2019_7_1T12_4_2_odor.h5",
                "1953_1_03_D2019_7_1T12_17_13_odor.h5",
                "1953_1_04_D2019_7_1T12_30_15_odor.h5",
                "1953_1_05_D2019_7_1T12_43_18_odor.h5",
                "1953_1_06_D2019_7_1T12_57_14_odor.h5",
                "1953_1_07_D2019_7_1T13_9_27_odor.h5",
                "1953_1_08_D2019_7_1T13_21_55_odor.h5",
                "1953_1_09_D2019_7_1T13_34_54_odor.h5",
                "1953_1_10_D2019_7_1T13_48_29_odor.h5",
            ]
        ],
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
    "190701_field11": CorrTrialFilePath(
        os.path.join(
            ROOTPATH, "190701/aligned", "HN1953_190701_field11_00001_000*.tif"
        ),
        os.path.join(ROOTPATH, "190701/aligned", "field11_masks/*.bmp"),
        os.path.join(ROOTPATH, "190701", "1953_1_10_D2019_7_1T13_48_29_odor.h5"),
        [
            os.path.join(ROOTPATH, "190701", h5)
            for h5 in [
                "1953_1_01_D2019_7_1T11_52_7_odor.h5",
                "1953_1_02_D2019_7_1T12_4_2_odor.h5",
                "1953_1_03_D2019_7_1T12_17_13_odor.h5",
                "1953_1_04_D2019_7_1T12_30_15_odor.h5",
                "1953_1_05_D2019_7_1T12_43_18_odor.h5",
                "1953_1_06_D2019_7_1T12_57_14_odor.h5",
                "1953_1_07_D2019_7_1T13_9_27_odor.h5",
                "1953_1_08_D2019_7_1T13_21_55_odor.h5",
                "1953_1_09_D2019_7_1T13_34_54_odor.h5",
                "1953_1_10_D2019_7_1T13_48_29_odor.h5",
            ]
        ][::-1],
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


def test_process_and_viz_correlations(
    trialFilePath, window=None, squashCondition=False
):
    roiStackPattern = trialFilePath.refPattern
    maskDir = trialFilePath.maskDir
    roiStack = TiffStack(roiStackPattern)
    roiStack.add_masks(maskDir)
    roiAverages = roiStack.cut_to_averages()
    refSession = H5Session(trialFilePath.refH5Filename, unified=squashCondition)
    roiMeanFs = refSession.get_meanFs(roiAverages)
    roiDF_Fs = {}
    for condition in roiMeanFs:
        roiDF_Fs[condition] = refSession.get_trial_average_data(
            roiAverages, roiMeanFs[condition], condition
        )

    savePath = trialFilePath.saveDir
    os.makedirs(savePath, exist_ok=True)
    for i_session, sessionH5 in enumerate(
        tqdm(trialFilePath.corrH5Filenames, unit="corrSession")
    ):
        corrSession = H5Session(
            sessionH5, title=str(i_session), unified=squashCondition
        )
        analyzeSession.process_and_viz_correlations(
            roiDF_Fs,
            roiStack.masks,
            trialFilePath.corrPatterns[i_session],
            corrSession,
            savePath,
            window=window,
        )


if __name__ == "__main__":
    if sys.argv[1] == "0":
        field = "190701_field2"
    else:
        field = "190701_field11"
    extraTitle = sys.argv[2]
    paths = trialFilePaths[field]
    paths.saveDir = os.path.join(paths.saveDir, extraTitle)
    if len(sys.argv) > 3:
        window = slice(int(sys.argv[3]), int(sys.argv[4]))
        logger.info("Running field {}, window {}\n{}".format(field, window, paths))
        test_process_and_viz_correlations(paths, window=window)
    else:
        logger.info("Running field {}, window {}\n{}".format(field, None, paths))
        test_process_and_viz_correlations(paths)
