from functools import partial
from importlib import reload
import unittest

from silq.pulses import PulseSequence, DCPulse, TriggerPulse, SinePulse
from silq.instrument_interfaces import get_instrument_interface
from silq.meta_instruments.chip import Chip
from silq.meta_instruments.layout import Layout
from silq.meta_instruments.mock_instruments import MockArbStudio

from qcodes import Instrument

if __name__ == "__main__":
    # from silq.tests.test_instruments import TestInnerInstrument, \
    #     TestOuterInstrument
    #
    # inner_instrument = TestInnerInstrument('inner')
    # outer_instrument = TestOuterInstrument('outer',
    #                                        instruments=[inner_instrument],
    #                                        server_name='outer')
    # outer_instrument.get_instruments()

    arbstudio = MockArbStudio('mock_arbstudio')
    chip = Chip('chip')
    layout = Layout('layout', instruments=[arbstudio, chip],
                    server_name='layout')