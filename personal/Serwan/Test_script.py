

if __name__ == "__main__":
    import silq
    silq.initialize("EWJN", ignore=['layout', 'parameters', 'plotting'])

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
    c3 = layout.add_connection(output_arg='arbstudio.ch3',
                               input_arg='ATS.chC', default=True)

    # Specify acquisition channels
    layout.acquisition_outputs([('arbstudio.ch3', 'pulses')])

    pulses = []
    pulses += [TriggerPulse(t_start=0, connection_requirements={
        'input_arg': 'arbstudio.trig_in'})]

    pulse_sequence = PulseSequence(pulses)
    layout.target_pulse_sequence(pulse_sequence)
    layout.setup()