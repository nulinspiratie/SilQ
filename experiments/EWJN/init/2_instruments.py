# Load instruments
from qcodes.instrument_drivers.lecroy.ArbStudio1104 import ArbStudio1104
from qcodes.instrument_drivers.spincore.PulseBlasterESRPRO import PulseBlaster
from qcodes.instrument_drivers.stanford_research.SIM900 import SIM900
from qcodes.instrument_drivers.AlazarTech.ATS9440 import ATS9440
from qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers import \
    Basic_AcquisitionController

from silq.meta_instruments.layout import Layout
from silq.parameters.general_parameters import ScaledParameter


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
arbstudio = ArbStudio1104('ArbStudio',
                          dll_path=dll_path,
                          server_name='')


### PulseBlaster
pulseblaster = PulseBlaster('PulseBlaster',
                            api_path='spinapi.py',
                            server_name='')


### ATS
ATS = ATS9440('ATS', server_name='Alazar_server')
ATS_controller = Basic_AcquisitionController(name='ATS_controller',
                                             alazar_name='ATS',
                                             server_name='Alazar_server')


### Interfaces
pulse_instruments = [arbstudio, pulseblaster, ATS]
interfaces = {instrument.name: get_instrument_interface(instrument)
                                      for instrument in pulse_instruments}
interfaces['ATS'].add_acquisition_controller(ATS_controller)


### Layout and connectivity
layout = Layout(name='layout',
                instrument_interfaces=list(interfaces.values()),
                server_name='layout_server')

layout.primary_instrument('pulseblaster')
layout.acquisition_instrument('ATS')

# Pulseblaster output connections
layout.add_connection(output_arg='pulseblaster.ch1',
                      input_arg='arbstudio.trig_in',
                      trigger=True)
layout.add_connection(output_arg='pulseblaster.ch2',
                      input_arg='ATS.trig_in',
                      trigger=True)

# Arbstudio output connections
c1 = layout.add_connection(output_arg='arbstudio.ch1',
                           input_arg='chip.TGAC',
                           pulse_modifiers={'amplitude_scale': 1})
c2 = layout.add_connection(output_arg='arbstudio.ch2',
                           input_arg='chip.DF',
                           pulse_modifiers={'amplitude_scale': -1.5})
c3 = layout.add_connection(output_arg='arbstudio.ch3',
                           input_arg='ATS.chC',
                           pulse_modifiers={'amplitude_scale': 1})
layout.combine_connections(c1, c2, c3, default=True)

# Chip output connection
layout.add_connection(output_arg='chip.output',
                      input_arg='ATS.chA')

# Specify acquisition channels
layout.acquisition_outputs([('chip.output', 'output'),
                            ('arbstudio.ch3', 'pulses')])