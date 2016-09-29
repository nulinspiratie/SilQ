import numpy as np


class PulseSequence:
    def __init__(self):
        self.pulses = []

    def __getitem__(self, index):
        return self.pulses[index]

    def add(self, pulse):
        # TODO deal with case when pulse is a string (e.g. 'trigger')
        self.pulses.append(pulse)
        self.sort()

    def sort(self):
        t_start_list = np.array([pulse.t_start for pulse in self.pulses])
        idx_sorted = np.argsort(t_start_list)
        self.pulses = [self.pulses[idx] for idx in idx_sorted]
        return self.pulses

    def clear(self):
        self.pulses = []