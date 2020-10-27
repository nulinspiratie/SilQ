from typing import Any, Dict, List
from copy import copy
from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import Pulse, DCPulse, SinePulse, FrequencyRampPulse, \
    TriggerPulse, PulseImplementation, MarkerPulse


class PCDDSInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name=instrument_name, **kwargs)

        # Setup channels
        self._output_channels = {
            f'ch{k}': Channel(self.instrument_name(),
                              name=f'ch{k}', id=k)
            for k in range(4)}

        self._input_channels = {
            f'pxi{k}': Channel(self.instrument_name(),
                               name=f'pxi{k}', input=True)
            for k in range(4)}
        self._input_channels['trig_in'] =  Channel(self.instrument_name(),
                                                   name=f'trig_in', input=True)

        self._channels = {**self._output_channels, **self._input_channels}

        self.pulse_implementations = [
            DCPulseImplementation(),
            SinePulseImplementation(),
            FrequencyRampPulseImplementation(),
            MarkerPulseImplementation()
        ]

        self.add_parameter('use_trig_in',
                           initial_value=True,
                           set_cmd=None,
                           docstring="Whether to use trig_in for triggering."
                                     "All DDS channels listen simultaneosly to"
                                     "trig_in, while the pxi channels can "
                                     "trigger individual dds channels")

        self.add_parameter('trigger_in_duration',
                           initial_value=.1e-6,
                           set_cmd=None,
                           docstring="Duration for a trigger input")

    @property
    def active_channel_ids(self):
        """Sorted list of active channel id's"""
        # First create a set to ensure unique elements
        active_channel_ids = {pulse.connection.output['channel'].id
                              for pulse in self.pulse_sequence}
        return sorted(active_channel_ids)

    @property
    def active_instrument_channels(self):
        return self.instrument.channels[self.active_channel_ids]

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Get list of pulses required by instrument (trigger pulses)

        A trigger pulse is returned for each pulse start and stop time.
        """
        assert self.use_trig_in(), "Interface not yet programmed for pxi triggering"
        # Get list of unique pulse start and stop times
        t_list = self.pulse_sequence.t_list
        trigger_pulses = [TriggerPulse(t_start=t,
                                       duration=self.trigger_in_duration(),
                                       connection_requirements={
                                           'input_instrument': self.instrument.name,
                                           'input_channel': self._input_channels['trig_in']
                                       })
                          for t in t_list if t != self.pulse_sequence.duration]
        return trigger_pulses

    def setup(self, **kwargs):
        for channel in self.instrument.channels:
            channel.instruction_sequence().clear()
        self.instrument.channels.output_enable(False)
        self.instrument.channels.pcdds_enable(True)

        assert self.use_trig_in(), "Interface not yet programmed for pxi triggering"

        # First pulses are 0V DC pulses
        # t_start and duration must be set but are irrelevant
        DC_0V_pulse = self.get_pulse_implementation(DCPulse('initial_0V',
                                                            t_start=0,
                                                            duration=0,
                                                            amplitude=0))
        current_pulses = {channel.name: DC_0V_pulse
                          for channel in self.active_instrument_channels}
        # for channel in self.active_instrument_channels:
            # current_pulses[channel.name]: copy(DC_0V_pulse)
            # current_pulses[channel.name].t_start = 0
            # next_pulse = next(self.pulse_sequence.get_pulses(output_channel=channel.name))
            # current_pulses[channel.name].t_stop = next_pulse.t_start

        for channel in self.active_instrument_channels:
            current_pulse = current_pulses[channel.name]
            pulse_implementation = current_pulse.implementation.implement()
            pulse_implementation['pulse_idx'] = 0
            pulse_implementation['next_pulse'] = 1
            channel.write_instr(pulse_implementation)

        total_instructions = len(self.pulse_sequence.t_list)
        for pulse_idx, t in enumerate(self.pulse_sequence.t_list):
            if t == self.pulse_sequence.duration:
                continue
            pulse_idx += 1  # We start with 1 since we have initial 0V pulse
            for channel in self.active_instrument_channels:
                active_pulse = self.pulse_sequence.get_pulse(t_start=t,
                                                             output_channel=channel.name)
                if active_pulse is not None:  # New pulse starts
                    current_pulses[channel.name] = active_pulse
                elif t >= current_pulses[channel.name].t_stop:
                    current_pulses[channel.name] = DC_0V_pulse

                pulse_implementation = current_pulses[channel.name].implementation.implement()
                pulse_implementation['pulse_idx'] = pulse_idx
                if pulse_idx + 1 < total_instructions:
                    pulse_implementation['next_pulse'] = pulse_idx + 1
                else:
                    # Loop back to second pulse (ignore first 0V pulse)
                    pulse_implementation['next_pulse'] = 1
                channel.write_instr(pulse_implementation)

    def start(self):
        self.active_instrument_channels.set_next_pulse(pulse=0, update=True)
        self.active_instrument_channels.output_enable(True)

    def stop(self):
        self.instrument.channels.output_enable(False)


class DCPulseImplementation(PulseImplementation):
    pulse_class = DCPulse

    def implement(self, *args, **kwargs):
        return {'instr': 'dc',
                'amp': self.pulse.amplitude}


class MarkerPulseImplementation(PulseImplementation):
    pulse_class = MarkerPulse

    def implement(self, *args, **kwargs):
        return {'instr': 'dc',
                'amp': self.pulse.amplitude}


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse

    def implement(self, *args, **kwargs):
        # TODO distinguish between abolute / relative phase
        phase = self.pulse.phase

        return {'instr': 'sine',
                'freq': self.pulse.frequency,
                'amp': self.pulse.amplitude,
                'offset': self.pulse.offset,
                'phase': phase
                }


class FrequencyRampPulseImplementation(PulseImplementation):
    pulse_class = FrequencyRampPulse

    def implement(self, *args, **kwargs):
        accumulation = (self.pulse.frequency_stop -
                        self.pulse.frequency_start) / self.pulse.duration
        return {'instr': 'chirp',
                'freq': self.pulse.frequency_start,
                'amp': self.pulse.amplitude,
                'offset': self.pulse.offset,
                'phase': getattr(self.pulse, 'phase', 0),
                'accum': accumulation
                }