import silq
from .interface import InstrumentInterface, Channel



def get_instrument_interface(instrument):
    from .lecroy.ArbStudio1104_interface import ArbStudio1104Interface
    from .chip_interface import ChipInterface
    from .spincore.PulseBlasterESRPRO_interface import \
        PulseBlasterESRPROInterface

    instrument_interfaces = {
        'ArbStudio1104': ArbStudio1104Interface,
        'MockArbStudio': ArbStudio1104Interface,
        'PulseBlasterESRPRO': PulseBlasterESRPROInterface,
        'MockPulseBlaster': PulseBlasterESRPROInterface,
        'Chip': ChipInterface
    }

    # TODO Need to find correct name of instrument
    instrument_class = instrument.__class__.__name__
    if instrument_class == 'RemoteInstrument':
        instrument_class = instrument._instrument_class.__name__

    instrument_interface_class = instrument_interfaces[instrument_class]

    instrument_interface = instrument_interface_class(instrument_name=instrument.name,
                                                      server_name=instrument._server_name)
    return instrument_interface
