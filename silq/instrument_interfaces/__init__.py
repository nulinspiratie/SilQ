from .interface import InstrumentInterface, Channel


def get_instrument_interface(instrument):
    from .lecroy.ArbStudio1104_interface import ArbStudio1104Interface
    from .chip_interface import ChipInterface
    from .spincore.PulseBlasterESRPRO_interface import \
        PulseBlasterESRPROInterface
    from .AlazarTech.ATS_interface import ATSInterface
    from .keysight import E8267DInterface

    instrument_interfaces = {
        'ArbStudio1104': ArbStudio1104Interface,
        'MockArbStudio': ArbStudio1104Interface,
        'PulseBlasterESRPRO': PulseBlasterESRPROInterface,
        'MockPulseBlaster': PulseBlasterESRPROInterface,
        'Chip': ChipInterface,
        'ATS9440': ATSInterface,
        'MockATS': ATSInterface,
        'Keysight_E8267D': E8267DInterface
    }

    instrument_class = instrument.__class__.__name__
    instrument_interface_class = instrument_interfaces[instrument_class]

    server_name = getattr(instrument, '_server_name', None)

    instrument_interface = instrument_interface_class(
        instrument_name=instrument.name,
        server_name=server_name)
    return instrument_interface
