import unittest

from silq.tests.test_pulses import *
from silq.tests.test_config import *
from silq.tests.test_pulse_sequences import *
from qcodes.tests.test_parameter_node import *
from qcodes.tests.test_parameter import (TestCopyParameter,
                                         TestParameterSignal,
                                         TestParameterLogging,
                                         TestParameterSnapshotting,
                                         TestParameterPickling,
                                         TestCopyParameterCount)
from qcodes.tests.test_measurement import *

if __name__ == '__main__':
    unittest.main()
