

if __name__ == "__main__":
    import silq

    silq.initialize("EWJN", ignore=['layout', 'parameters', 'plotting'])
    from silq.pulses import *

    for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
        arbstudio.parameters[ch + '_sampling_rate_prescaler'](256)

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

    ramp_pulse1 = DCRampPulse(amplitude_start=-1, amplitude_stop=1,
                              duration=10, acquire=True)
    ramp_pulse2 = DCRampPulse(amplitude_start=-1, amplitude_stop=1,
                              duration=20, acquire=True)
    ramp_pulse3 = DCRampPulse(amplitude_start=-1, amplitude_stop=1,
                              duration=30, acquire=True)
    final_pulse = DCPulse(name='final', amplitude=0,
                          duration=2)
    pulses = [ramp_pulse1, ramp_pulse2, ramp_pulse3, final_pulse]

    pulse_sequence = PulseSequence()
    pulse_sequence.add(pulses)

    layout.target_pulse_sequence(pulse_sequence)
    layout.setup()
    # layout.start()
    # result = layout.acquisition()