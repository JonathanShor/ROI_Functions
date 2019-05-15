"""Find all zero crossings from voltage traces.
"""
import os
import h5py
import numpy as np


def findCrossings(sequence):
    signs = np.sign(sequence)
    next_signs = signs[1:]
    # Prepend False -- first position cannot be a crossing
    crossings = np.append(False, signs[:-1] != next_signs)
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
    rootpath = "/Volumes/research/rinberglabspace/Users/Jonathan/"
    files = [
        "2P_Data/odor_test/JG24831/a_01_D2019_1_26T18_26_19_beh.h5",
        "2P_Data/odor_test/JG24831/a_02_D2019_1_26T18_42_55_beh.h5",
    ]
    # filenames = list(map(lambda x: os.path.join(rootpath, x), files))
    results = getCrossingsFromFiles(files, rootpath)
    assert len(results) == 2
    T18_26 = results["2P_Data/odor_test/JG24831/a_01_D2019_1_26T18_26_19_beh.h5"]
    assert len(T18_26) == 60
    assert len(T18_26[0]) == 44
    assert np.all(T18_26[0][0][:4] == [False, True, True, False])
