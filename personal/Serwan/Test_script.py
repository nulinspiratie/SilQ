

if __name__ == "__main__":
    USE_MP = False
    import silq

    silq.initialize("EWJN")

    frequency_center = 25.175e9

    EPR_parameter.setup(samples=50)
    EPR_parameter()
    traces_read = EPR_parameter.trace_segments['output']['read']
    _, _, threshold_voltage = analysis.find_high_low(traces_read)
    assert threshold_voltage is not None, "Couldn't find accurate threshold"
    print('Threshold voltage found at {:.2f} V'.format(threshold_voltage))


    T1_parameter.pulse_sequence['steered_initialization'].enabled = False

    T1_parameter.pulse_sequence['plunge'].amplitude = 1.8

    T1_parameter.pulse_sequence['adiabatic_sweep'].enabled = True
    T1_parameter.pulse_sequence[
        'adiabatic_sweep'].frequency_center = frequency_center
    T1_parameter.pulse_sequence['adiabatic_sweep'].frequency_deviation = 10e6
    T1_parameter.pulse_sequence['adiabatic_sweep'].t_start = 1
    T1_parameter.pulse_sequence['adiabatic_sweep'].duration = 0.2

    T1_parameter.pulse_sequence['read'].duration = 20

    T1_parameter.setup(threshold_voltage, samples=3)
    T1_parameter(10)
    T1_parameter.print_results = True

    up_proportion, number_traces_loaded = T1_parameter()
    print('\n\n*** Starting summary')
    from pympler import tracker
    memory_tracker = tracker.SummaryTracker()

    up_proportion, number_traces_loaded = T1_parameter()
    print('\n\n*** Printing diff')
    memory_tracker.print_diff()


    T1_parameter.setup(threshold_voltage, samples=1000)
    T1_parameter(10)
    T1_parameter.print_results = True

    up_proportion, number_traces_loaded = T1_parameter()
    print('\n\n*** Printing diff')
    memory_tracker.print_diff()
