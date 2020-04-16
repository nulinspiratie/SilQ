from typing import Any, Dict
from time import sleep

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import TriggerPulse, PulseImplementation

class TriggerFPGAInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name=instrument_name, **kwargs)

        # Setup channels
        self._output_channels = {
            f'pxi{k}': Channel(instrument_name=self.instrument_name(),
                               name=f'pxi{k}', id=k)
            for k in range(8)}
        self._output_channels['trig_out'] = Channel(instrument_name=self.instrument_name(),
                                                    name='trig_out',
                                                    output_TTL=(0, 2))
        self._channels = self._output_channels

        self.pulse_implementations = [TriggerPulseImplementation(
            pulse_requirements=[('t_start', {'min': 0, 'max': 0})]
        )]

        self.add_parameter('final_delay',
                           unit='s',
                           initial_value=None,
                           set_cmd=None)
        self.add_parameter('trigger_duration',
                           unit='s',
                           initial_value=100e-9,
                           set_cmd=None)
        self.repeat = True

    def setup(self,
              repeat: bool = True, **kwargs):
        assert self.is_primary(), "FPGA trigger source can only function as " \
                                  "primary instrument"
        self.repeat = repeat

        pxi_ports = {pulse.connection.output['channel'].name
                       for pulse in self.pulse_sequence}
        assert len(pxi_ports) == 1, "Only one PXI output can be configured " \
                                    "for triggering"
        pxi_channel = next(iter(pxi_ports))
        pxi_number = int(pxi_channel[-1])
        self.instrument.pxi_port(pxi_number)

        if self.repeat:
            trigger_interval = self.pulse_sequence.duration
            if self.final_delay() is not None:
                trigger_interval += self.final_delay()
            else:
                trigger_interval += self.pulse_sequence.final_delay
        else:
            trigger_interval = .4  # Crazy high number to ensure we stop it on time

        self.instrument.trigger_interval(trigger_interval)
        self.instrument.trigger_duration(self.trigger_duration())

        # targeted_pulse_sequence is the pulse sequence that is currently setup
        self.targeted_pulse_sequence = self.pulse_sequence
        self.targeted_input_pulse_sequence = self.input_pulse_sequence

    def start(self):
        self.instrument.start()
        if not self.repeat:
            sleep(0.6)
            self.stop()

    def stop(self):
        self.instrument.stop()


class TriggerPulseImplementation(PulseImplementation):
    pulse_class = TriggerPulse
