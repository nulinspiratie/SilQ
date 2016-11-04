from silq.meta_instruments.layout import Layout

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
layout.add_connection(output_arg='pulseblaster.ch4',
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