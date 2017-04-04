from silq.meta_instruments.layout import Layout

### Layout and connectivity
layout = Layout(name='layout',
                instrument_interfaces=list(interfaces.values()),
                server_name='layout_server' if USE_MP else None)

layout.primary_instrument('pulseblaster')
layout.acquisition_instrument('ATS')

# Pulseblaster output connections
cPB4 = layout.add_connection(output_arg='pulseblaster.ch1',
                             input_arg='ATS.trig_in',
                             trigger=True)
cPB2 = layout.add_connection(output_arg='pulseblaster.ch2',
                             input_arg='arbstudio.trig_in',
                             trigger=True)

# Arbstudio output connections
cArb1 = layout.add_connection(output_arg='arbstudio.ch1',
                              input_arg='chip.DF', scale=1/80)
cArb2 = layout.add_connection(output_arg='arbstudio.ch2',
                              input_arg='chip.PL', scale=1/80)
cArb3 = layout.add_connection(output_arg='arbstudio.ch3',
                              input_arg='ATS.chC', scale=1/80)

# Without scales for single channels, the compensation factor for DF is -1.5
# With scales this becomes -1.5 * 1/20 / 1/25 = -15/8
layout.combine_connections(cArb1, cArb2, cArb3, default=True,
                           scale=[1, -0.55, 1])

# Chip output connection
layout.add_connection(output_arg='chip.output',
                      input_arg='ATS.chA')

# Specify acquisition channels
layout.acquisition_outputs([('chip.output', 'output'),
                            ('arbstudio.ch3', 'pulses')])