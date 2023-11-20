from pyrbit.mem_utils import BlockBasedSchedule


import numpy


def test_BlockBasedSchedule():
    interblock_time=[100, 200, 400]
    nitems = 3
    intertrial_time = 1
    repet_trials = 1
    schedule = BlockBasedSchedule(nitems, intertrial_time, interblock_time, repet_trials=repet_trials)

    


if __name__ == "__main__":
    test_BlockBasedSchedule()
