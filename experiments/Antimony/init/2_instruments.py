###################
### Instruments ###
###################

from qcodes.instrument_drivers.AlazarTech.ATS9440 import ATS9440
from qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers import \
    Triggered_AcquisitionController, Continuous_AcquisitionController, \
    SteeredInitialization_AcquisitionController
from qcodes.instrument_drivers.lecroy.ArbStudio1104 import ArbStudio1104
from qcodes.instrument_drivers.spincore.PulseBlasterESRPRO import PulseBlasterESRPRO
from qcodes.instrument_drivers.stanford_research.SIM900 import SIM900
from silq.meta_instruments.chip import Chip
from silq.parameters.general_parameters import ScaledParameter
from silq.meta_instruments.layout import Layout

station = qc.Station()
interfaces = {}

##############
### SIM900 ###
##############
SIM900 = SIM900('SIM900', 'GPIB0::4::INSTR')
station.add_component(SIM900)
# Each DC voltage source has format (name, slot number, divider, max raw voltage)
DC_sources = [('SRC', 1, 1, 1), ('LB', 2, 8, 2), ('RB', 3, 8, 2), ('TG', 4, 8, 2.25),
              ('TGAC', 5, 8, 1.25), ('DF', 6, 8, 1.25), ('DS', 7, 8, 1.25)]
gates = ['SRC','LB', 'RB', 'TG', 'TGAC', 'DF', 'DS']
SIM900_scaled_parameters = []
for ch_name, ch, ratio,max_voltage in DC_sources:
    SIM900.define_slot(channel=ch, name=ch_name+'_raw', max_voltage=max_voltage*ratio)
    param_raw = SIM900.parameters[ch_name+'_raw']
    param = ScaledParameter(param_raw, name=ch_name, label=ch_name, ratio=ratio)
    station.add_component(param)
    SIM900_scaled_parameters.append(param)

    exec('{ch_name}_raw = param_raw'.format(ch_name=ch_name))
    exec('{ch_name} = param'.format(ch_name=ch_name))


#################
### Arbstudio ###
#################
dll_path = os.path.join(os.getcwd(),'C:\lecroy\\Library\\ArbStudioSDK.dll')
arbstudio = ArbStudio1104('arbstudio', dll_path=dll_path)
station.add_component(arbstudio)
for ch in ['ch1', 'ch2', 'ch4']:
    arbstudio.parameters[ch + '_sampling_rate_prescaler'](250)
# ch3 is used for sideband modulation
arbstudio.ch3_sampling_rate_prescaler(1)
interfaces['arbstudio'] = get_instrument_interface(arbstudio)
arbstudio_interface = interfaces['arbstudio']

####################
### PulseBlaster ###
####################
pulseblaster = PulseBlasterESRPRO('pulseblaster', api_path='spinapi.py')
station.add_component(pulseblaster)
pulseblaster.core_clock(500)
interfaces['pulseblaster'] = get_instrument_interface(pulseblaster)
pulseblaster_interface = interfaces['pulseblaster']

############
### Chip ###
############
chip = Chip(name='chip', channels=['TGAC', 'DF', 'antenna'])
station.add_component(chip)
interfaces['chip'] = get_instrument_interface(chip)
chip_interface = interfaces['chip']

###########
### ATS ###
###########
ATS = ATS9440('ATS')
station.add_component(ATS)
triggered_controller = Triggered_AcquisitionController(
    name='triggered_controller',
    alazar_name='ATS')
station.add_component(triggered_controller)
continuous_controller = Continuous_AcquisitionController(
    name='continuous_controller',
    alazar_name='ATS')
station.add_component(continuous_controller)
# steered_controller = SteeredInitialization_AcquisitionController(
#     name='steered_initialization_controller',
#     target_instrument=pulseblaster,
#     alazar_name='ATS')
# steered_controller.record_initialization_traces(True)

interfaces['ATS'] = get_instrument_interface(ATS)
interfaces['ATS'].add_acquisition_controller('triggered_controller')
interfaces['ATS'].add_acquisition_controller('continuous_controller')
# interfaces['ATS'].add_acquisition_controller('steered_initialization_controller')
interfaces['ATS'].default_acquisition_controller('Triggered')
ATS_interface = interfaces['ATS']

##############
### Layout ###
##############
layout = Layout(name='layout',
                instrument_interfaces=list(interfaces.values()))
station.add_component(layout)
layout.primary_instrument('pulseblaster')
layout.acquisition_instrument('ATS')
layout.load_connections()
