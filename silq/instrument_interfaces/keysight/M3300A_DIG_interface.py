from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses.pulse_types import TriggerPulse

from qcodes.utils import validators as vals
from qcodes import ManualParameter
from qcodes.instrument_drivers.keysight.M3300A import M3300A_DIG as dig_driver


class M3300A_DIG_Interface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)
        self._pulse_sequence.allow_untargeted_pulses = True

        # for kw in kwargs.keys():
        #     if (kw in M3300A_interface_kwargs):
        #         print(kw)
        # A prelim implementation for a triggered connection, can only handle
        # one trigger per pulse sequence
        self.acq_mode = 'OneShot'
        self.add_parameter('acquisition',
                           label='Acquisition parameter',
                           get_cmd=self.acquire,
                           docstring='Parameter to use for acquisition in loop'
                           )
        # Initialize channels
        self._input_channels = {
            'ch{}'.format(k): Channel(instrument_name=self.instrument_name(),
                                      name='ch{}'.format(k), input=True)
            for k in range(8)
            }

        self._channels = {
            **self._input_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input=True),
        }

        # Names of acquisition channels [chA, chB, etc.]
        self.add_parameter(name='acquisition_channels',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Anything())

        # Obtain a list of all valid M3300A Digitizer parameters
        self._parameters_names = sorted(list(
            self.parameters.keys()))
        # Set up the driver to a known default state
        self.initialize_driver()

    def initialize_driver(self):
        for k in range(8):
            self.instrument.parameters['impedance_{}'.format(k)].set(1)  # 50 Ohm impedance
            self.instrument.parameters['coupling_{}'.format(k)].set(0)  # DC Coupled
            self.instrument.parameters['full_scale_{}'.format(k)].set(3.0)  # 3.0 Volts

        self.sample_freq = 100e6;
        # Configure the trigger type
        if self.acq_mode == 'OneShot':
            for k in range(8):
                # Trigger on rising edge of some channel
                # TODO: Read connections to figure out where to trigger from
                self.instrument.parameters['trigger_edge_{}'.format(k)].set(1)
                self.instrument.parameters['trigger_threshold_{}'.format(k)].set(0.5)
                # Select channel on which to trigger each DAQ
                self.instrument.parameters['analog_trigger_mask_{}'.format(k)].set(1 << 2)
                self.instrument.parameters['n_cycles_{}'.format(k)].set(-1)
                self.instrument.parameters['DAQ_trigger_delay_{}'.format(k)].set(0)
                self.instrument.parameters['DAQ_trigger_mode_{}'.format(k)].set(3)

    def get_final_additional_pulses(self, **kwargs):
        if not self._pulse_sequence.get_pulses(acquire=True):
            # No pulses need to be acquired
            return []
        elif self.acq_mode == 'OneShot':
            # Add a single trigger pulse when starting acquisition
            t_start = min(pulse.t_start for pulse in
                          self._pulse_sequence.get_pulses(acquire=True))
            t_stop = max(pulse.t_stop for pulse in
                         self._pulse_sequence.get_pulses(acquire=True))
            t_final = max(pulse.t_stop for pulse in
                          self._pulse_sequence.get_pulses())

            T = t_stop - t_start

            # Capture maximum number of samples on all channels
            for k in range(8):
                self.instrument.parameters['points_per_cycle_{}'.format(k)].set(int(T * self.sample_freq))
                self.instrument.parameters['n_points_{}'.format(k)].set(int(T * self.sample_freq))
                # Set an acquisition timeout to be 10% after the last pulse finishes.
                self.instrument.parameters['timeout_{}'.format(k)].set(int(t_final * self.sample_freq * 1.1))

            acquisition_pulse = \
                TriggerPulse(t_start=t_start, duration=1e-5,
                             connection_requirements={
                                 'input_instrument': self.instrument_name(),
                                 'trigger': True})
            return [acquisition_pulse]

    def setup(self, **kwargs):
        pass
        # for param in self._used_params:

    def start(self):
        self.instrument.daq_stop_multiple(2 ** 8 - 1)
        self.instrument.daq_flush_multiple(2 ** 8 - 1)
        self.instrument.daq_start_multiple(2 ** 8 - 1)

    def acquire(self):
        data = {}
        # The start of acquisition
        t_0 = min(pulse.t_start for pulse in
                      self._pulse_sequence.get_pulses(acquire=True))
        # Split data into pulse traces
        for ch in range(8):
            data[ch] = {}
            ch_data = self.instrument.daq_read(ch)
            for pulse in self._pulse_sequence.get_pulses(acquire=True):
                ts = (pulse.t_start - t_0, pulse.t_stop - t_0)
                sample_range = [int(t * self.sample_freq) for t in ts]
                # Extract acquired data from the channel data
                data[ch][pulse.name] = ch_data[sample_range[0]:sample_range[1]]

        # For instrument safety, stop all acquisition after we are done
        self.instrument.daq_stop_multiple(2 ** 8 - 1)
        return data

    def stop(self):
        pass

