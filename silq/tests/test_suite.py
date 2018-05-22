import unittest

from .test_pulses import *
from .test_config import *
from .test_pulse_sequences import *
from qcodes.tests.test_parameter_node import *
from qcodes.tests.test_parameter import (TestCopyParameter,
                                         TestParameterSignal,
                                         TestParameterLogging,
                                         TestParameterSnapshotting)

if __name__ == '__main__':
    unittest.main()