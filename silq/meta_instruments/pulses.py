

class Pulse:
    def __init__(self, t_start, t_stop=None, duration=None):
        self.t_start = t_start
        if t_stop is not None:
            self.t_stop = t_stop
            self.duration = t_stop - t_start
        elif duration is not None:
            self.duration = duration
            self.t_stop = t_start + duration
        else:
            raise Exception("Must provide either t_stop or duration")


class SinePulse(Pulse):
    def __init__(self, frequency, amplitude, **kwargs):
        super().__init__(**kwargs)

        self.frequency = frequency
        self.amplitude = amplitude

        class DCPulse(Pulse):
            def __init__(self, amplitude, **kwargs):
                super().__init__(kwargs)

                self.frequency = frequency
                self.amplitude = amplitude

    def __repr__(self):
        return 'SinePulse(f={:.2f} MHz, A={}, t_start={}, t_stop={})'.format(
            self.frequency/1e6, self.amplitude, self.t_start, self.t_stop
        )
