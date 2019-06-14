"""Find all zero crossings from voltage traces.
"""
import os
import h5py
import numpy as np


def findAllCrossings(sequence):
    signs = np.sign(sequence)
    next_signs = signs[1:]
    # Prepend False -- first position cannot be a crossing
    crossings = np.append(False, signs[:-1] != next_signs)
    return crossings


def findCrossings(sequence, window=20, minWidth=5):
    """Return mask of sequence where series of window positive values is followed by
        window negative entries.

    Arguments:
        sequence {[numerical]} -- Sequence of values.

    Keyword Arguments:
        window {int} -- Length of positive then negative streaks to test for. (default:
            {20})

    Returns:
        [bool] -- Boolean mask of sequence.
    """
    assert window > 0
    assert minWidth > 0
    sequence = np.array(sequence)
    # Preload with Falses
    posWindow = np.zeros((len(sequence),), dtype=bool)
    negWindow = np.zeros((len(sequence),), dtype=bool)
    for i in range(minWidth, len(sequence) - minWidth):
        posWindow[i] = all(sequence[max(0, i - window) : i] >= 0)
        negWindow[i] = all(sequence[i : min(len(sequence), i + window)] <= 0)
    crossings = np.logical_and(posWindow, negWindow)
    return crossings


def getCrossingsFromTrial(trialDataset):
    crossings = np.empty_like(trialDataset)
    for i_data, data in enumerate(trialDataset):
        crossings[i_data] = findCrossings(data)
    return crossings


def getCrossingsFromFiles(filenames, rootpath=""):
    results = {}
    for filename in filenames:
        with h5py.File(os.path.join(rootpath, filename), "r") as h5File:
            trialsResults = []  # np.empty(len(h5File) - 1)
            # meta = h5File["Trials"]
            for i_trial in range(1, len(h5File)):
                trial = h5File["Trial{:04d}".format(i_trial)]
                trialsResults.append(getCrossingsFromTrial(trial["sniff"]))
        results[filename] = trialsResults
    return results


if __name__ == "__main__":
    # Unittest
    rootpath = "./data/"
    files = ["190603/1953_1_04_D2019_6_3T12_29_13_odor.h5"]
    # filenames = list(map(lambda x: os.path.join(rootpath, x), files))
    results = getCrossingsFromFiles(files, rootpath)
    assert len(results) == 2
    T18_26 = results["2P_Data/odor_test/JG24831/a_01_D2019_1_26T18_26_19_beh.h5"]
    assert len(T18_26) == 60
    assert len(T18_26[0]) == 44
    assert np.all(T18_26[0][0][:4] == [False, True, True, False])
