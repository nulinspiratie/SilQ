

if __name__ == "__main__":
    USE_MP = False
    import silq

    silq.initialize("EWJN")

    ELR_parameter.setup(samples=50)
    ELR_parameter()
    traces_read = ELR_parameter.trace_segments['output']['read']
    _, _, threshold_voltage = analysis.find_high_low(traces_read, plot=True)

    steered_initialization = SteeredInitialization(
        name='steered_initialization',
        t_no_blip=20, t_max_wait=500, t_buffer=40)
    load_pulse = DCPulse(name='load', amplitude=1.5,
                         duration=5, acquire=True)
    read_pulse = DCPulse(name='read', amplitude=0,
                         duration=50, acquire=True)
    final_pulse = DCPulse(name='final', amplitude=0,
                          duration=2)
    pulses = [steered_initialization, load_pulse, read_pulse, final_pulse]
    pulse_sequence = PulseSequence(pulses=pulses)

    layout.target_pulse_sequence(pulse_sequence)

    layout.setup(samples=10,
                 readout_threshold_voltage=threshold_voltage)
    result = layout.do_acquisition(return_dict=True)