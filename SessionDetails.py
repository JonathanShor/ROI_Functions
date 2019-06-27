from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class SessionDetails:
    numCycles: int
    # Number of repeats of cycleTemplate in the entire session
    numTrialsPerCycles: int
    # Length of each cycle. If numCycles==1, this has no effect.
    cycleTemplate: Dict[str, Sequence[int]]
    # Name of each condition and the corresponding trial indices in the first cycle
    # Do NOT enter zero-indexed. Enter a 2 for the second trial, etc.
    # An example:
    # = {
    #     "100% A, 0% B": [2, 15],
    #     "90% A, 10% B": [3, 14],
    #     "75% A, 25% B": [4, 13],
    #     "50% A, 50% B": [5, 12],
    #     "25% A, 75% B": [6, 11],
    #     "10% A, 90% B": [7, 10],
    #     "0% A, 100% B": [8, 9],
    # }
    odorNames: Dict[str, str]
    # = {"A": "Ethylbutyrate", "B": "Propionic Acid"}
    preWindow: int = 500
    postWindow: int = 1500


HN1953_190603 = SessionDetails(
    3,
    15,
    {
        "100% A, 0% B": [2, 15],
        "90% A, 10% B": [3, 14],
        "71% A, 29% B": [4, 13],
        "50% A, 50% B": [5, 12],
        "29% A, 71% B": [6, 11],
        "10% A, 90% B": [7, 10],
        "0% A, 100% B": [8, 9],
    },
    {"A": "Ethylbutyrate", "B": "Propionic Acid"},
)
HN1953_190617 = SessionDetails(
    2,
    32,
    {
        "100% A, 0% B": [3, 25],
        "90% A, 10% B": [4, 24],
        "71% A, 29% B": [5, 23],
        "50% A, 50% B": [6, 22],
        "29% A, 71% B": [7, 21],
        "10% A, 90% B": [8, 20],
        "0% A, 100% B": [9, 19],
        "100% C, 0% D": [11, 33],
        "90% C, 10% D": [12, 32],
        "71% C, 29% D": [13, 31],
        "50% C, 50% D": [14, 30],
        "29% C, 71% D": [15, 29],
        "10% C, 90% D": [16, 28],
        "0% C, 100% D": [17, 27],
    },
    {
        "A": "Propionic Acid",
        "B": "Ethylbutyrate",
        "C": "Benzaldehyde",
        "D": "ButyricAcid",
    },
)
HN1953_190617_field2_00003 = SessionDetails(
    numCycles=1,
    numTrialsPerCycles=32,
    cycleTemplate={
        "100% A, 0% B": list(range(3, 18)),
        "0% A, 100% B": list(range(19, 34)),
    },
    odorNames={"A": "Methylvalerate", "B": "Ethyltiglate"},
)
