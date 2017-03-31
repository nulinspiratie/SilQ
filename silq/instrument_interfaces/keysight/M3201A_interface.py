from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import SinePulse, PulseImplementation, TriggerPulse


class M3201AInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            'ch{}'.format(k):
                Channel(instrument_name=self.instrument_name(),
                        name='ch{}'.format(k), id=k,
                        output=True) for k in range(4)}

        self._pxi_channels = {
            'pxi{}'.format(k):
                Channel(instrument_name=self.instrument_name(),
                        name='pxi{}'.format(k), id=4000+k,
                        input_trigger=True, output=True, input=True) for k in range(8)}

        self._channels = {
            **self._output_channels,
            **self._pxi_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True, input_TTL=(0, 5.0)),
            'trig_out': Channel(instrument_name=self.instrument_name(),
                                name='trig_out', output_TTL=(0, 3.3))}

        # TODO: how does the power parameter work? How can I set requirements on the amplitude?
        self.pulse_implementations = [
            SinePulseImplementation(
                pulse_requirements=[('frequency', {'min': 0, 'max': 200e6}),
                                    ('power', {'max': 1.5})]
            )
        ]

    # TODO: is this device specific? Does it return [0,1,2] or [1,2,3]?
    def _get_active_channels(self):
        active_channels = [pulse.connection.output['channel'].name
                           for pulse in self.pulse_sequence()]
        # Transform into set to ensure that elements are unique
        active_channels = list(set(active_channels))
        return active_channels

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

    def target_pulse(self, pulse, interface, is_primary=False, **kwargs):
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, is_primary=is_primary, **kwargs)

        # Check if there are already trigger pulses
        trigger_pulses = interface.input_pulse_sequence().get_pulses(
            t_start=pulse.t_start, trigger=True
        )

        if not (is_primary or trigger_pulses or targeted_pulse.t_start == 0):
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=targeted_pulse.t_start,
                             duration=interface.trigger_in_duration() * 1e-3,
                             connection_requirements={
                                 'input_instrument':
                                     interface.instrument_name(),
                                 'trigger': True}
                             )
            )
        return targeted_pulse

    def implement(self):
        """
        This function takes the targeted pulse (i.e. an interface specific pulseimplementation) and converts
        it to a set of pulse-independent instructions/information that can be handled by interface.setup().

        For example:
            SinePulseImplementation()
                initializes a generic (interface independent) pulseimplementation for the sinepulse
            SinePulseImplementation.target_pulse()
                target the generic (interface independent) pulseimplementation to this specific interface, also adds
                trigger requirements, which are communicated back to the layout. The trigger requirements are also
                interface specific. The pulse in now called a 'targeted pulse'.
            SinePulseImplementation.implement()
                takes the targeted pulse and returns information/instructions that are independent of the pulse type
                (DCPulse, SinePulse, ...) and can be handled by interface.setup() to send instructions to the driver.
        """
        pass
