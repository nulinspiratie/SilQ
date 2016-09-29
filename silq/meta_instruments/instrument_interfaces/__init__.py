from .InstrumentInterface import InstrumentInterface, Channel
from .lecroy.ArbStudio1104 import ArbStudio1104_Interface

instrument_interfaces = {
    'ArbStudio1104': ArbStudio1104_Interface
}

def get_instrument_interface(instrument):
    # TODO Need to find correct name of instrument
    instrument_name = instrument.__class__

    instrument_interface_class = instrument_interfaces[instrument_name]

    instrument_interface = instrument_interface_class(instrument)
    return instrument_interface
