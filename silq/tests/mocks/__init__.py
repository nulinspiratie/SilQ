from silq.instrument_interfaces import instrument_interfaces

def add_mock_instruments():
    """Attach existing interfaces to mock instruments.

    Can be called via ``silq.get_instrument_interface``
    """
    instrument_interfaces['ArbStudio1104'] = {
        'module': '.lecroy.ArbStudio1104_interface',
        'class': 'ArbStudio1104Interface'}
    instrument_interfaces['MockArbStudio'] = {
        'module': '.lecroy.ArbStudio1104_interface',
        'class': 'ArbStudio1104Interface'}
    instrument_interfaces['PulseBlasterDDS'] = {
        'module': '.spincore.PulseBlasterDDS_interface',
        'class': 'PulseBlasterDDSInterface'}
    instrument_interfaces['PulseBlasterESRPRO'] = {
        'module': '.spincore.PulseBlasterESRPRO_interface',
        'class': 'PulseBlasterESRPROInterface'}
    instrument_interfaces['MockPulseBlaster'] = {
        'module': '.spincore.PulseBlasterESRPRO_interface',
        'class': 'PulseBlasterESRPROInterface'}

def add_mock_interfaces():
    """Create new mock instrument and mock interface

    Can be called via ``silq.get_instrument_interface``
    """
    instrument_interfaces['MockTriggerInstrument'] = {
        'module': 'silq.tests.mocks.mock_interfaces',
        'class': 'MockTriggerInterface'
    }
    instrument_interfaces['MockAWGInstrument'] = {
        'module': 'silq.tests.mocks.mock_interfaces',
        'class': 'MockAWGInterface'
    }
    instrument_interfaces['MockDigitizerInstrument'] = {
        'module': 'silq.tests.mocks.mock_interfaces',
        'class': 'MockDigitizerInterface'
    }
    instrument_interfaces['MockMicrowaveInstrument'] = {
        'module': 'silq.tests.mocks.mock_interfaces',
        'class': 'MockMicrowaveInterface'
    }

