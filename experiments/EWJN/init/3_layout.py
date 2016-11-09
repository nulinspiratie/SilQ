from silq.meta_instruments.layout import Layout

### Layout and connectivity
layout = Layout(name='layout',
                instrument_interfaces=list(interfaces.values()),
                server_name='layout_server' if USE_MP else None)

layout.primary_instrument('pulseblaster')
layout.acquisition_instrument('ATS')

# Pulseblaster output connections
cPB1 = layout.add_connection(output_arg='pulseblaster.ch1',
                             input_arg='arbstudio.trig_in',
                             trigger=True)
cPB2 = layout.add_connection(output_arg='pulseblaster.ch2',
                             input_arg='keysight.trig_in',
                             trigger=True)
cPB3 = layout.add_connection(output_arg='pulseblaster.ch3',
                             input_arg='ATS.chB')
cPB4 = layout.add_connection(output_arg='pulseblaster.ch4',
                             input_arg='ATS.trig_in',
                             trigger=True)

# Arbstudio output connections
cArb1 = layout.add_connection(output_arg='arbstudio.ch1',
                              input_arg='chip.TGAC',
                              pulse_modifiers={'amplitude_scale': 1})
cArb2 = layout.add_connection(output_arg='arbstudio.ch2',
                              input_arg='chip.DF',
                              pulse_modifiers={'amplitude_scale': -1.5})
cArb3 = layout.add_connection(output_arg='arbstudio.ch3',
                              input_arg='ATS.chC',
                              pulse_modifiers={'amplitude_scale': 1})
layout.combine_connections(cArb1, cArb2, cArb3, default=True)
cArb4 = layout.add_connection(output_arg='arbstudio.ch4',
                              input_arg='keysight.ext2')

cESR = layout.add_connection(output_arg='keysight.RF_out',
                             input_arg='chip.ESR')
# Chip output connection
layout.add_connection(output_arg='chip.output',
                      input_arg='ATS.chA')

# Specify acquisition channels
layout.acquisition_outputs([('chip.output', 'output'),
                            ('arbstudio.ch3', 'pulses')])