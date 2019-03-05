import unittest
from functools import wraps
from time import time
from copy import copy, deepcopy
from profilehooks import profile

from silq.pulses import DCPulse, DCRampPulse, PulseSequence

from qcodes import Parameter

class Timer():
    def __init__(self, name):
        self.name = name
        self.t_start = time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'{self.name}: {time() - self.t_start:.3f} s')


def timing(N):
    def timing_decorator(fun):
        @wraps(fun)
        def timing_wrapper(*args, **kwargs):
            t0 = time()
            for k in range(N):
                fun(*args, **kwargs)
            call_duration = (time() - t0) / N
            print(f'{fun.__name__} duration: {call_duration:.3g} s')
        return timing_wrapper
    return timing_decorator


class TestParameterSpeed(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.parameter = Parameter(name='param')
        super().__init__(*args, **kwargs)

    @timing(2000)
    def test_create_parameter(self):
        p = Parameter('parameter_name')

    @timing(2000)
    def test_copy_parameter(self):
        copy(self.parameter)

    @timing(2000)
    def test_deepcopy_parameter(self):
        deepcopy(self.parameter)


# class TestPulseCreation(unittest.TestCase):
#     # @profile(sort='time')
#     @timing(3000)
#     def test_create_pulse(self):
#         DCPulse('DC', t_start=0, duration=5, amplitude=1)


class TestPulseSequenceCreation(unittest.TestCase):
    pulse_sequence = None

    def __init__(self, *args, **kwargs):
        self.pulse_sequence = self.create_large_pulse_sequence()
        super().__init__(*args, **kwargs)

    def create_large_pulse_sequence(self):
        pulse_sequence = PulseSequence()
        duration = 10e-3
        for k in range(2):
            pulse_sequence.add(DCRampPulse('DC_ramp', amplitude_start=0,
                                           amplitude_stop=1,
                                           t_start=k*duration,
                                           duration=duration))
        return pulse_sequence

    # @timing(1)
    # def test_create_large_pulse_sequence(self):
    #     self.create_large_pulse_sequence()

    # @timing(1)
    # def test_copy_large_pulse_sequence(self):
    #     copy(self.pulse_sequence)


if __name__ == '__main__':
    unittest.main()