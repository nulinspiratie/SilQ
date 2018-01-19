import pytest
from copy import copy

from silq.meta_instruments.layout import Layout, SingleConnection
from silq.meta_instruments.chip import Chip

from silq.instrument_interfaces.Tektronix.AWG520_interface import AWG520Interface
from silq.instrument_interfaces.chip_interface import ChipInterface
from silq.instrument_interfaces.interface import InstrumentInterface
from silq.pulses.pulse_sequences import PulseSequence
from silq.pulses.pulse_types import DCPulse, SinePulse

from qcodes import Instrument


class MockInterface(ChipInterface):
    def has_pulse_implementation(self, pulse):
        return pulse

    def get_pulse_implementation(self, pulse, connections=None):
        pulse_copy = copy(pulse)
        pulse_copy.implementation = 'something'
        return pulse_copy


@pytest.fixture
def setup():
    Instrument.close_all()
    Instrument('AWG520')
    AWG_interface = AWG520Interface('AWG520')

    Chip('chip', channels=['ch1', 'ch2'])
    chip_interface = ChipInterface('chip')

    trigger_instrument = Chip('triggerer', channels=['ch1', 'ch2'])
    trigger_interface = MockInterface('triggerer')

    layout = Layout(instrument_interfaces=[AWG_interface,
                                           chip_interface,
                                           trigger_interface])
    layout.load_connections(connections_dicts=[
        {"output_arg": "AWG520.ch1",
         "input_arg": "chip.ch1"},
        {"output_arg": "triggerer.ch1",
         "input_arg": "AWG520.trig_in",
         "trigger": True},
    ])
    return {'AWG_interface': AWG_interface,
            'chip_interface': chip_interface,
            'trigger_interface': trigger_interface,
            'layout': layout}


def test_setup_initialization(setup):
    pass

def test_single_DC_pulse(setup):
    layout = setup['layout']
    AWG_interface = setup['AWG_interface']

    DC_pulse = DCPulse(t_start=0, t_stop=10e-3, amplitude=1,
                       connection_requirements={'output_arg': 'AWG520.ch1'})
    pulse_sequence = PulseSequence(pulses=[DC_pulse])

    layout.pulse_sequence = pulse_sequence

    print(repr(AWG_interface.pulse_sequence))