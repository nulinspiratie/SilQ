# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from copy import deepcopy, copy

from qcodes import Parameter
from silq.pulses import DCPulse, SinePulse
from silq.pulses.pulse_sequences import ElectronReadoutPulseSequence

# class TimeSuite:
#     """
#     An example benchmark that times the performance of various kinds
#     of iterating over dictionaries in Python.
#     """
#     def setup(self):
#         self.d = {}
#         for x in range(500):
#             self.d[x] = None
#
#     def time_keys(self):
#         for key in self.d.keys():
#             pass
#
#     def time_iterkeys(self):
#         for key in self.d.iterkeys():
#             pass
#
#     def time_range(self):
#         d = self.d
#         for key in range(500):
#             x = d[key]
#
#     def time_xrange(self):
#         d = self.d
#         for key in xrange(500):
#             x = d[key]
#
#
# class MemSuite:
#     def mem_list(self):
#         return [0] * 256


class ParameterSuite:
    def setup(self):
        self.parameter = Parameter(name='param')

    def time_create_parameter(self):
        Parameter('parameter_name')

    def time_copy_parameter(self):
        copy(self.parameter)

    def time_deepcopy_parameter(self):
        deepcopy(self.parameter)


def new_sine_pulse(N=1):
    if N == 1:
        return SinePulse('ESR', frequency=28e9, power=16, duration=1e-5)
    else:
        return [SinePulse('ESR', frequency=28e9, power=16, duration=1e-5) for _ in range(N)]


class PulseSuite:
    def setup(self):
        self.pulse = new_sine_pulse()

    def time_create_sine_pulse(self):
        new_sine_pulse()

    def time_copy_connect_pulse(self):
        self.pulse.copy(connect_to_config=True)

    def time_copy_no_connect_pulse(self):
        self.pulse.copy(connect_to_config=False)

    def time_deepcopy_pulse(self):
        deepcopy(self.pulse)

class PulseSequenceSuite:
    def setup(self):
        pseq = ElectronReadoutPulseSequence()
        pseq.settings.RF_pulses = new_sine_pulse(10)
        pseq.settings.stage_pulse = DCPulse('plunge', amplitude=1, duration=1e-3)
        pseq.settings.read_pulse = DCPulse('read', amplitude=0, duration=1e-3)
        self.pseq = pseq

    def time_create_sequence(self):
        self.setup()

    def time_generate_sequence(self):
        self.pseq.generate()

    def time_copy_connect_pseq(self):
        self.pseq.copy(connect_to_config=True)

    def time_copy_no_connect_pseq(self):
        self.pseq.copy(connect_to_config=False)

    def time_deepcopy_pseq(self):
        deepcopy(self.pseq)

    def time_deepcopy_pulse_settings(self):
        deepcopy(self.pseq.pulse_settings)

    def time_update_pulse_settings(self):
        self.pseq._update_latest_pulse_settings()
