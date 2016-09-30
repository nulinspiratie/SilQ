import silq

from .interface import InstrumentInterface, Channel
from .lecroy.ArbStudio1104_interface import ArbStudio1104Interface
from .chip_interface import ChipInterface

instrument_interfaces = {
    'ArbStudio1104': ArbStudio1104Interface,
    'MockArbStudio': ArbStudio1104Interface,
    'Chip': ChipInterface
}

def get_instrument_interface(instrument):
    # TODO Need to find correct name of instrument
    instrument_name = instrument.__class__.__name__
    if instrument_name == 'RemoteInstrument':
        instrument_name = instrument._instrument_class.__name__

    instrument_interface_class = instrument_interfaces[instrument_name]

    instrument_interface = instrument_interface_class(instrument)
    return instrument_interface
