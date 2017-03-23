from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import SinePulse, PulseImplementation


class M3201AInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            'ch{}'.format(k):
                Channel(instrument_name=self.instrument_name(),
                        name='ch{}'.format(k), id=k,
                        output=True) for k in range(4)}

        self._channels = {
            **self._output_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True),
            'trig_out': Channel(instrument_name=self.instrument_name(),
                                name='trig_out', output_TTL=(0, 3.3))}
        # TODO: lookup output TTL for M3201A
        self.pulse_implementations = [
            SinePulseImplementation(
                pulse_requirements=[('frequency', {'min': 1e6, 'max': 125e6})]
            )
        ]
        # TODO: how to implement the PXI triggers?

    # TODO: is this device specific? Does it return [0,1,2] or [1,2,3]?
    def _get_active_channels(self):
        active_channels = [pulse.connection.output['channel'].name
                           for pulse in self.pulse_sequence()]
        # Transform into set to ensure that elements are unique
        active_channels = list(set(active_channels))
        return active_channels

    # TODO: is this the behaviour we want?
    def stop(self):
        # stop all AWG channels and sets FG channels to 'No Signal'
        self.instrument.off()

    def setup(self):
        pass

    def start(self):
        mask = 0
        for c in self.active_channels():
            mask |= 1 << c
        self.instrument.awg_start_multiple(mask)

    def get_final_additional_pulses(self, **kwargs):
        return []

    def write_raw(self, cmd):
        pass

    def ask_raw(self, cmd):
        pass


# TODO: figure out how this hole pulseimplementation stuff works, seems overcomplicated atm
class SinePulseImplementation(PulseImplementation, SinePulse):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=SinePulse, **kwargs)

    def implement(self):
        pass
