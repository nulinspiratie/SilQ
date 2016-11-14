# Load instruments
from qcodes.instrument_drivers.AlazarTech.ATS9440 import ATS9440
from qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers import \
    Triggered_AcquisitionController, Continuous_AcquisitionController, \
    SteeredInitialization_AcquisitionController
from qcodes.instrument_drivers.keysight.E8267D import Keysight_E8267D
from qcodes.instrument_drivers.lecroy.ArbStudio1104 import ArbStudio1104
from qcodes.instrument_drivers.spincore.PulseBlasterESRPRO import PulseBlasterESRPRO
from qcodes.instrument_drivers.stanford_research.SIM900 import SIM900
from silq.meta_instruments.chip import Chip
from silq.parameters.general_parameters import ScaledParameter

interfaces = {}


### SIM900
SIM900 = SIM900('SIM900', 'GPIB0::4::INSTR',server_name='' if USE_MP else None)
# Each DC voltage source has format (name, slot number, divider, max raw voltage)
DC_sources = [('TG',1,8,18), ('LB',2,4,8), ('RB',3,4,8), ('TGAC',4,5,4),
         ('SRC',5,1,1), ('DS',7,4,3.2), ('DF',6,4,3.2)]
SIM900_scaled_parameters = []
for ch_name, ch, ratio,max_voltage in DC_sources:
    SIM900.define_slot(channel=ch, name=ch_name+'_raw', max_voltage=max_voltage)
    if USE_MP:
        SIM900.update()
    param_raw = SIM900.parameters[ch_name+'_raw']
    param = ScaledParameter(param_raw, name=ch_name, label=ch_name, ratio=ratio)
    SIM900_scaled_parameters.append(param)

    exec('{ch_name}_raw = param_raw'.format(ch_name=ch_name))
    exec('{ch_name} = param'.format(ch_name=ch_name))


### ArbStudio
dll_path = os.path.join(os.getcwd(),'C:\lecroy_driver\\Library\\ArbStudioSDK.dll')
arbstudio = ArbStudio1104('arbstudio',
                          dll_path=dll_path,
                          server_name='' if USE_MP else None)
for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
    arbstudio.parameters[ch + '_sampling_rate_prescaler'](250)
interfaces['arbstudio'] = get_instrument_interface(arbstudio)


### PulseBlaster
pulseblaster = PulseBlasterESRPRO('pulseblaster',
                            api_path='spinapi.py',
                            server_name='' if USE_MP else None)
pulseblaster.core_clock(500)
interfaces['pulseblaster'] = get_instrument_interface(pulseblaster)


### Chip
chip = Chip(name='chip', server_name='' if USE_MP else None)
interfaces['chip'] = get_instrument_interface(chip)


### ATS
if USE_MP:
    from qcodes.instrument.server import InstrumentServerManager
    InstrumentServerManager('Alazar_server', {'target_instrument':pulseblaster})
ATS = ATS9440('ATS', server_name='Alazar_server' if USE_MP else None)
triggered_controller = Triggered_AcquisitionController(
    name='triggered_controller',
    alazar_name='ATS',
    server_name='Alazar_server' if USE_MP else None)
continuous_controller = Continuous_AcquisitionController(
    name='continuous_controller',
    alazar_name='ATS',
    server_name='Alazar_server' if USE_MP else None)
steered_controller = SteeredInitialization_AcquisitionController(
    name='steered_initialization_controller',
    target_instrument=pulseblaster,
    alazar_name='ATS',
    server_name='Alazar_server' if USE_MP else None)
steered_controller.silent(False)
steered_controller.record_initialization_traces(True)

interfaces['ATS'] = get_instrument_interface(ATS)
interfaces['ATS'].add_acquisition_controller('triggered_controller')
interfaces['ATS'].add_acquisition_controller('continuous_controller')
interfaces['ATS'].add_acquisition_controller('steered_initialization_controller')
interfaces['ATS'].default_acquisition_controller('Triggered')


### MW source
keysight = Keysight_E8267D('keysight','TCPIP0::192.168.0.5::inst0::INSTR',
                           server_name='' if USE_MP else None)
interfaces['keysight'] = get_instrument_interface(keysight)
interfaces['keysight'].modulation_channel('ext2')
interfaces['keysight'].envelope_padding(0.2)
keysight.power(10)