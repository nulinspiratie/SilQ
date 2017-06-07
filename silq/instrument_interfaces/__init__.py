from .interface import InstrumentInterface, Channel


def get_instrument_interface(instrument):
    instrument_interfaces = {
        'ArbStudio1104': {'module': '.lecroy.ArbStudio1104_interface',
                          'class': 'ArbStudio1104Interface'},
        'MockArbStudio': {'module': '.lecroy.ArbStudio1104_interface',
                          'class': 'ArbStudio1104Interface'},
        'PulseBlasterESRPRO': {'module': '.spincore.PulseBlasterESRPRO_interface',
                               'class': 'PulseBlasterESRPROInterface'},
        'MockPulseBlaster': {'module': '.spincore.PulseBlasterESRPRO_interface',
                             'class': 'PulseBlasterESRPROInterface'},
        'Chip': {'module': '.chip_interface',
                 'class': 'ChipInterface'},
        'ATS9440': {'module': '.AlazarTech.ATS_interface',
                    'class': 'ATSInterface'},
        'MockATS': {'module': '.AlazarTech.ATS_interface',
                    'class': 'ATSInterface'},
        'Keysight_E8267D': {'module': '.keysight',
                            'class': 'E8267DInterface'},
        'Keysight_M3201A': {'module': '.keysight',
                            'class': 'M3201AInterface'},
        'M3300A_DIG': {'module': '.keysight',
                       'class': 'M3300A_DIG_Interface'}
    }

    instrument_class = instrument.__class__.__name__
    import_dict = instrument_interfaces[instrument_class]
    exec(f'from {import_dict["module"]} import {import_dict["class"]}')
    instrument_interface_class = eval(import_dict['class'])
    server_name = getattr(instrument, '_server_name', None)

    instrument_interface = instrument_interface_class(
        instrument_name=instrument.name,
        server_name=server_name)
    return instrument_interface
