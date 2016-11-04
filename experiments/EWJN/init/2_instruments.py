# Load instruments
from qcodes.instrument_drivers.AlazarTech.ATS9440 import ATS9440
from qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers import \
    Triggered_AcquisitionController, Continuous_AcquisitionController
from qcodes.instrument_drivers.keysight.E8267D import Keysight_E8267D
from qcodes.instrument_drivers.lecroy.ArbStudio1104 import ArbStudio1104
from qcodes.instrument_drivers.spincore.PulseBlasterESRPRO import PulseBlasterESRPRO
from qcodes.instrument_drivers.stanford_research.SIM900 import SIM900
from silq.meta_instruments.chip import Chip
from silq.parameters.general_parameters import ScaledParameter

interfaces = {}

### SIM900
SIM900 = SIM900('SIM900', 'GPIB0::4::INSTR',server_name='')
# Each DC voltage source has format (name, slot number, divider, max raw voltage)
DC_sources = [('TG',1,8,18), ('LB',2,4,8), ('RB',3,4,8), ('TGAC',4,5,4),
         ('SRC',5,1,1), ('DS',7,4,3.2), ('DF',6,4,3.2)]
for ch_name, ch, ratio,max_voltage in DC_sources:
    SIM900.define_slot(channel=ch, name=ch_name+'_raw', max_voltage=max_voltage)
    SIM900.update()
    param_raw = SIM900.parameters[ch_name+'_raw']
    param = ScaledParameter(param_raw, name=ch_name, label=ch_name, ratio=ratio)

    exec('{ch_name}_raw = param_raw'.format(ch_name=ch_name))
    exec('{ch_name} = param'.format(ch_name=ch_name))


### ArbStudio
dll_path = os.path.join(os.getcwd(),'C:\lecroy_driver\\Library\\ArbStudioSDK.dll')
arbstudio = ArbStudio1104('arbstudio',
                          dll_path=dll_path,
                          server_name='')
for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
    arbstudio.parameters[ch + '_sampling_rate_prescaler'](250)
interfaces['arbstudio'] = get_instrument_interface(arbstudio)

### PulseBlaster
pulseblaster = PulseBlasterESRPRO('pulseblaster',
                            api_path='spinapi.py',
                            server_name='')
pulseblaster.core_clock(500)
interfaces['pulseblaster'] = get_instrument_interface(pulseblaster)
# interfaces['pulseblaster'].ignore_first_trigger(True)

### Chip
chip = Chip(name='chip', server_name='')
interfaces['chip'] = get_instrument_interface(chip)


### ATS
ATS = ATS9440('ATS', server_name='Alazar_server')
triggered_controller = Triggered_AcquisitionController(
    name='triggered_controller',
    alazar_name='ATS',
    server_name='Alazar_server')
continuous_controller = Continuous_AcquisitionController(
    name='continuous_controller',
    alazar_name='ATS',
    server_name='Alazar_server')
interfaces['ATS'] = get_instrument_interface(ATS)
interfaces['ATS'].add_acquisition_controller('triggered_controller')
interfaces['ATS'].add_acquisition_controller('continuous_controller')


### MW source
keysight = Keysight_E8267D('keysight','TCPIP0::192.168.0.5::inst0::INSTR',
                           server_name='')
interfaces['keysight'] = get_instrument_interface(keysight)
interfaces['keysight'].modulation_channel('ext2')
