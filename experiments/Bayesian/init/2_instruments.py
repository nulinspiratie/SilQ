###################
### Instruments ###
###################

from qcodes.instrument_drivers.keysight.M3300A import M3300A_DIG as DIGDriver
from qcodes.instrument_drivers.keysight.SD_common.SD_acquisition_controller import Triggered_Controller
from qcodes.instrument_drivers.keysight.M3201A import Keysight_M3201A as AWGDriver
from qcodes.instrument_drivers.spincore.PulseBlasterESRPRO import PulseBlasterESRPRO
import qcodes.instrument_drivers.national_instruments.PXIe_4322 as AODriver
from silq.meta_instruments.chip import Chip
from silq.parameters.general_parameters import ScaledParameter
from silq.meta_instruments.layout import Layout

from silq.instrument_interfaces.keysight.M3300A_DIG_interface import M3300A_DIG_Interface

interfaces = {}

##############
### NI AWG ###
##############
AOI = AODriver.PXIe_4322('4322','Dev2')
station.add_component(AOI)
voltage_parameters = []
# ch_names = ['TG', 'LB', 'RB', 'SD','DF', 'DS', 'TGAC', 'NUL']
DC_sources = [('TG', 0, 5.0, 1), ('LB', 1, 8, 2), ('RB', 2, 8, 2), ('SD', 3, 8, 2.25),
              ('DF', 4, 3.0, 1.25), ('DS', 5, 4.0, 1.25), ('TGAC', 6, 3.0, 1.25)]
for ch_name, ch, ratio, max_voltage in DC_sources:
    param_name = 'voltage_channel_'
    param_raw = AOI.parameters[param_name  + str(ch)]
    param = ScaledParameter(param_raw, name=ch_name, label=ch_name, scale=ratio)
    station.add_component(param)
    voltage_parameters.append(param)

    exec('{ch_name}_raw = param_raw'.format(ch_name=ch_name))
    exec('{ch_name} = param'.format(ch_name=ch_name))

# Each DC voltage source has format (name, slot number, divider, max raw voltage)
# DC_sources = [('SRC', 1, 1, 1), ('LB', 2, 8, 2), ('RB', 3, 8, 2), ('TG', 4, 8, 2.25),
#               ('TGAC', 5, 8, 1.25), ('DF', 6, 8, 1.25), ('DS', 7, 8, 1.25)]
# gates = ['SRC','LB', 'RB', 'TG', 'TGAC', 'DF', 'DS']
# for ch_name, ch, ratio,max_voltage in DC_sources:
#     SIM900.define_slot(channel=ch, name=ch_name+'_raw', max_voltage=max_voltage*ratio)
#     param_raw = SIM900.parameters[ch_name+'_raw']
#     param = ScaledParameter(param_raw, name=ch_name, label=ch_name, scale=ratio)
#     station.add_component(param)
#     voltage_parameters.append(param)
#
#     exec('{ch_name}_raw = param_raw'.format(ch_name=ch_name))
#     exec('{ch_name} = param'.format(ch_name=ch_name))


####################
### Keyisght AWG ###
####################
AWG_instrument = AWGDriver('AWG')

station.add_component(AWG_instrument)
interfaces['awg'] = get_instrument_interface(AWG_instrument)
AWG_interface = interfaces['awg']


####################
### PulseBlaster ###
# ####################
# pulseblaster = PulseBlasterESRPRO('pulseblaster', api_path='spinapi.py')
# station.add_component(pulseblaster)
# pulseblaster.core_clock(500)
# interfaces['pulseblaster'] = get_instrument_interface(pulseblaster)
# pulseblaster_interface = interfaces['pulseblaster']


############
### Chip ###
############
chip = Chip(name='Berdina', channels=['TGAC', 'DF', 'antenna'])
station.add_component(chip)
interfaces['Berdina'] = get_instrument_interface(chip)
berdina_interface = interfaces['Berdina']


###########
### ATS ###
###########

DIG_instrument = DIGDriver('M3300A_DIG')
ctrl = Triggered_Controller('M3300A_Controller', 'M3300A_DIG')
DIG_interface = M3300A_DIG_Interface('M3300A_DIG', acquisition_controller_names=['M3300A_Controller'])
DIG_interface.acquisition_controller('Triggered')
DIG_interface.initialize_driver()
station.add_component(DIG_instrument)
# station.add_component(ctrl)

interfaces['DIG'] = DIG_interface
# interfaces['DIG'].add_acquisition_controller('triggered_controller')
# interfaces['DIG'].add_acquisition_controller('continuous_controller')
# interfaces['DIG'].add_acquisition_controller('steered_initialization_controller')
interfaces['DIG'].default_acquisition_controller('Triggered')

##############
### Layout ###
##############
layout = Layout(name='layout',
                instrument_interfaces=list(interfaces.values()))


station.add_component(layout)
layout.primary_instrument('AWG')
# Specify the acquisition instrument
layout.acquisition_instrument('M3300A_DIG')
layout.acquisition_outputs([('AWG.ch0', 'output')])
layout.load_connections()

# Update InteractivePlot layout
InteractivePlot.layout = layout