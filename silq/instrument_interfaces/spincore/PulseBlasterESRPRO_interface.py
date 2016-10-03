from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, PulseImplementation


class PulseBlasterESRPROInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self.output_channels = {
            'ch{}'.format(k): Channel(self, name='ch{}'.format(k), id=k,
                                      output_trigger=True)
            for k in [1, 2, 3, 4]}
        self.channels = {**self.output_channels}

        self.pulse_implementations = [
            TriggerPulse.create_implementation(
                TriggerPulseImplementation,
                pulse_conditions=[]
            )
        ]

    def setup(self):
        remaining_pulses = self._pulse_sequence.pulses
        while remaining_pulses:
            t_start_list = [pulse.t_start for pulse in remaining_pulses]
            t_start_min = min(t_start_list)
            active_pulses = [pulse for pulse in remaining_pulses if pulse.t_start == t_start_min]


            remaining_pulses = [pulse for pulse in remaining_pulses if pulse.t_start != t_start_min]





class TriggerPulseImplementation(PulseImplementation):
    trigger_duration = 100
    def __init__(self, pulse_class, **kwargs):
        super().__init__(pulse_class, **kwargs)

    def implement_pulse(self, trigger_pulse):
        pass