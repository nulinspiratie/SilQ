

if __name__ == "__main__":
    USE_MP = False
    import silq

    silq.initialize("EWJN")

    frequency_center = 25.175e9

    EPR_parameter.setup(samples=50)
    EPR_parameter()
    traces_read = EPR_parameter.trace_segments['output']['read']
    _, _, readout_threshold_voltage = analysis.find_high_low(traces_read)
    assert readout_threshold_voltage is not None, "Couldn't find accurate threshold"
    print(
        'Threshold voltage found at {:.2f} V'.format(readout_threshold_voltage))



    adiabatic_sweep_parameter.pulse_sequence['empty'].enabled = False

    adiabatic_sweep_parameter.pulse_sequence['plunge'].amplitude = 1.8
    adiabatic_sweep_parameter.pulse_sequence['plunge'].duration = 5

    adiabatic_sweep_parameter.pulse_sequence['steered_initialization'].enabled = True
    adiabatic_sweep_parameter.pulse_sequence['steered_initialization'].t_buffer = 30
    adiabatic_sweep_parameter.pulse_sequence['steered_initialization'].t_no_blip = 90
    adiabatic_sweep_parameter.pulse_sequence['steered_initialization'].t_max_wait = 400

    adiabatic_sweep_parameter.pulse_sequence['adiabatic_sweep'].frequency_deviation=10e6
    adiabatic_sweep_parameter.pulse_sequence['adiabatic_sweep'].duration = 0.2
    adiabatic_sweep_parameter.pulse_sequence['adiabatic_sweep'].enabled = True
    adiabatic_sweep_parameter.pulse_sequence['adiabatic_sweep'].t_start = 4

    adiabatic_sweep_parameter.pulse_sequence['read'].duration=120

    adiabatic_sweep_parameter.setup(readout_threshold_voltage=readout_threshold_voltage)
    adiabatic_sweep_parameter(frequency_center)
    adiabatic_sweep_parameter.pulse_sequence

    TGAC_vals = list(np.linspace(0.33, 0.35, 8))
    DF_DS_vals = list(np.linspace(0.476, 0.485, 8))

    adiabatic_sweep_parameter.setup(samples=15,
                                    readout_threshold_voltage=readout_threshold_voltage,
                                    save_traces=True,
                                    data_manager=data_manager_raw)
    data = qc.Loop(TGAC[TGAC_vals]
                   ).loop(DF_DS[DF_DS_vals]
                          ).each(adiabatic_sweep_parameter
                                 ).run(name='adiabatic_calibration',
                                       progress_interval=True, background=False)