import unittest

from silq.tests.test_pulses import *
from silq.tests.test_config import *
from silq.tests.test_pulse_sequences import *
from qcodes.tests.test_parameter_node import *
from qcodes.tests.test_parameter import (
    TestValsandParseParameter,
    TestCopyParameter,
    TestParameterSignal,
    TestParameterLogging,
    TestParameterSnapshotting,
    TestParameterPickling,
    TestCopyParameterCount,
)
from qcodes.tests.test_measurement import *
from qcodes.tests.test_combined_par import *
from qcodes.tests.test_command import *
from qcodes.tests.test_data import *
from qcodes.tests.test_deferred_operations import *
from qcodes.tests.test_format import *
from qcodes.tests.test_generic_formatter import *
from qcodes.tests.test_hdf5formatter import *
from qcodes.tests.test_helpers import *
from qcodes.tests.test_instrument import *
from qcodes.tests.test_json import *
from qcodes.tests.test_location_provider import *
from qcodes.tests.test_metadata import *
# from qcodes.tests.test_sweep_values import *
# from qcodes.tests.test_validators import *
from qcodes.tests.test_visa import *



if __name__ == "__main__":
    unittest.main()
