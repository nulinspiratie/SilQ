###################
### Instruments ###
###################

from qcodes.instrument_drivers.AlazarTech.ATS9440 import ATS9440
from qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers import \
    Triggered_AcquisitionController, Continuous_AcquisitionController, \
    SteeredInitialization_AcquisitionController
from qcodes.instrument_drivers.lecroy.ArbStudio1104 import ArbStudio1104
from qcodes.instrument_drivers.spincore.PulseBlasterESRPRO import PulseBlasterESRPRO
from qcodes.instrument_drivers.stanford_research.SIM900 import SIM900, \
    get_voltages, ramp_voltages, voltage_parameters
from qcodes.instrument_drivers.Keysight.E8267D import Keysight_E8267D
from silq.meta_instruments.chip import Chip
from silq.parameters.general_parameters import ScaledParameter
from silq.meta_instruments.layout import Layout

interfaces = {}


##############
### SIM900 ###
##############
SIM900_DC = SIM900('SIM900_DC', 'GPIB0::4::INSTR')
station.add_component(SIM900_DC)
SIM900_SRC = SIM900('SIM900_SRC', 'GPIB0::2::INSTR')
station.add_component(SIM900_SRC)
# Each DC voltage source has format (name, slot number, divider, max raw voltage)
DC_sources = {'SRC':  {'ch': 1, 'ratio': 1, 'max': 1, 'ins': SIM900_SRC},
              'LB':   {'ch': 1, 'ratio': 8, 'max': 1.6, 'ins':SIM900_DC},
              'RB':   {'ch': 2, 'ratio': 8, 'max': 1.6, 'ins': SIM900_DC},
              'TG':   {'ch': 3, 'ratio': 8, 'max': 2.25, 'ins': SIM900_DC},
              'TGAC': {'ch': 4, 'ratio': 8, 'max': 1, 'ins': SIM900_DC},
              'LDF':  {'ch': 5, 'ratio': 8, 'max': 1, 'ins': SIM900_DC},
              'RDF':  {'ch': 6, 'ratio': 8, 'max': 1, 'ins': SIM900_DC},
              'LDS':  {'ch': 7, 'ratio': 8, 'max': 1, 'ins': SIM900_DC},
              'RDS':  {'ch': 8, 'ratio': 8, 'max': 1, 'ins': SIM900_DC}}
gates = ['SRC','LB', 'RB', 'TG', 'TGAC', 'LDF', 'RDF', 'LDS', 'RDS']
for ch_name, info in DC_sources.items():
    instrument = info['ins']
    instrument.define_slot(channel=info['ch'], name=f'{ch_name}_raw',
                       max_voltage=info['max'] * info['ratio'])
    param_raw = instrument.parameters[f'{ch_name}_raw']
    param = ScaledParameter(param_raw, name=ch_name, label=ch_name,
                            scale=info['ratio'])
    station.add_component(param)
    voltage_parameters.append(param)

    exec(f'{ch_name}_raw = param_raw')
    exec(f'{ch_name} = param')

RDF_raw.set_step(0.02)
LDF_raw.set_step(0.02)
TGAC_raw.set_step(0.02)

sim_gui.voltage_parameters = voltage_parameters


#################
### Arbstudio ###
#################
dll_path = os.path.join(os.getcwd(),'C:\lecroy\\Library\\ArbStudioSDK.dll')
arbstudio = ArbStudio1104('arbstudio', dll_path=dll_path)
station.add_component(arbstudio)
arbstudio.optimize = ['divisors']
for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
    arbstudio.parameters[ch + '_sampling_rate_prescaler'](40)
# # ch3 is used for sideband modulation
# arbstudio.ch3_sampling_rate_prescaler(1)
interfaces['arbstudio'] = get_instrument_interface(arbstudio)
arbstudio_interface = interfaces['arbstudio']
arbstudio_interface.final_delay(0.3)


####################
### PulseBlaster ###
####################
pulseblaster = PulseBlasterESRPRO('pulseblaster', api_path='spinapi.py')
station.add_component(pulseblaster)
pulseblaster.core_clock(500)
interfaces['pulseblaster'] = get_instrument_interface(pulseblaster)
pulseblaster_interface = interfaces['pulseblaster']


#################
### MW source ###
#################
keysight = Keysight_E8267D('keysight','TCPIP0::192.168.7.67::inst0::INSTR')
interfaces['keysight'] = get_instrument_interface(keysight)
keysight_interface = interfaces['keysight']
interfaces['keysight'].modulation_channel('ext2')
interfaces['keysight'].envelope_padding(0.2)
keysight.power(10)


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
ATS.config(sample_rate=500000)
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
