from typing import Tuple
__all__ = ['pulse_to_waveform_sequence']

def pulse_to_waveform_sequence(duration: float,
                               frequency: float,
                               sampling_rate: float,
                               threshold: float,
                               n_min: int = 1,
                               n_max: int = 1000,
                               sample_points_multiple: int = 1) -> \
        Tuple[int, float, int]:
    """Method for generating a periodic signal with an AWG device. 
    
    Given a frequency and duration of the
    desired signal, a general AWG can produce that signal by repeating one waveform (waveform_1) for a number of times
    (cycles) and ending with a second waveform (waveform_2). This is a practical way of generating periodic signals that
    are long in duration without using a lot of the available RAM of the AWG.

    Because of the finite sampling rate and restrictions on amount of sample points (which must generally be a multiple
    of a certain number), there will be an error in the period of the generated signal. This error goes down with the
    number of periods (n) that one cycle of the repeated waveform contains.

    This function calculates the minimum number n for which the error threshold is satisfied. Therefore minimizing the
    total amount of sample points that need to be stored by the AWG.

    Args:
        duration: duration of the signal in seconds
        frequency: frequency of the signal in Hz
        sampling_rate: the sampling rate of the waveform
        threshold: threshold in relative period error
        n_min: minimum number of signal periods that the waveform must contain
        n_max: maximum number of signal periods that a waveform can contain
        sample_points_multiple: the number of samples must be a multiple of

    Returns:
        Tuple:
        * **n (int)**:        number of signal periods that are in one cycle of the repeating waveform
        * **error (float)**:  relative error in the signal period
        * **samples (int)**:  number of samples in one cycle of the repeating waveform

    """
    period = 1 / frequency
    cycles = duration // period
    n_max = int(min(n_max, cycles))

    period_sample = 1 / sampling_rate

    extra_sample = False
    n = 0
    error = 0

    for n in range(n_min, n_max + 1):
        error = (n * period) % (period_sample * sample_points_multiple)
        error_extra_sample = (period_sample * sample_points_multiple) - error
        if error_extra_sample < error:
            extra_sample = True
            error = error_extra_sample
        else:
            extra_sample = False
        error = error / n / period
        if error < threshold:
            break
        else:
            continue

    samples = (n * period) // (period_sample * sample_points_multiple) * sample_points_multiple
    if extra_sample:
        samples += sample_points_multiple

    return n, error, samples
