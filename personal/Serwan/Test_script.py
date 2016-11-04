

if __name__ == "__main__":
    import silq
    # silq.initialize("EWJN", ignore=['layout', 'parameters', 'plotting'])
    silq.initialize("EWJN")

    gamma_e = 28.02495266 * 1e9  # Hz
    B_0 = 1.2  # T

    f_res = gamma_e * B_0
    f_span = 10e6
    f_start = f_res - f_span / 2
    f_stop = f_res + f_span / 2

    amplitude = 10


    pulses = [FrequencyRampPulse(frequency_start=f_start,
                                 frequency_stop=f_stop,
                                 frequency_final=f_start,
                                 t_start=10,
                                 duration=0.2,
                                 amplitude=amplitude
                                 )
              ]
    pulse_sequence = PulseSequence(pulses)

    layout.target_pulse_sequence(pulse_sequence)