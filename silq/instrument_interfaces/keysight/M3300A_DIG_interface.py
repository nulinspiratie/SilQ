from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection

from qcodes.utils import validators as vals
from qcodes.instrument_drivers.keysight.M3300A import M3300A_DIG as dig_driver

class M3300A_DIG_Interface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)
        self._pulse_sequence.allow_untargeted_pulses = True

        for kw in kwargs.keys():
            if (kw in M3300A_interface_kwargs):
                print(kw)
        # A prelim implementation for a triggered connection, can only handle
        # one trigger per pulse sequence
        self.acq_mode = 'OneShot'
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

        # Obtain a list of all valid M3300A Digitizer parameters
        self._parameters_names = sorted(list(
            self.parameters.keys()))
        # Set up the driver to a known default state
        self.initialize_driver()

    def initialize_driver(self):
        for k in range(8):
            self.parameters['impedance_'.format(k)].set(1)    # 50 Ohm impedance
            self.parameters['coupling_'.format(k)].set(0)     # DC Coupled
            self.parameters['full_scale_'.format(k)].set(3.0) # 3.0 Volts
        
        self.sample_freq = 100e6;
        # Configure the trigger type
        if self.acq_mode == 'OneShot':
            for k in range(8):
                # Trigger on rising edge of some channel
                # TODO: Read connections to figure out where to trigger from
                self.parameters['trigger_mode_{}'.format(k)].set(3)
                self.parameters['trigger_threshold_{}'.format(k)].set(0.65)
                self.parameters['trigger_mode_{}'.format(k)].set(3)
                # Select channel on which to trigger each DAQ
                self.parameters['analog_trigger_mask_{}'.format(k)].set(0)
                self.parameters['DAQ_trigger_delay_{}'.format(k)].set(0)
                self.parameters['DAQ_trigger_mode_{}'.format(k)].set(3)


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
                self.parameters['n_points_{}'.format(k)].set(T*self.sample_freq)
                # Set an acquisition timeout to be 10% after the last pulse finishes.
                self.parameters['timeout_{}'.format(k)].set(t_final*1.1)

            acquisition_pulse = \
                TriggerPulse(t_start=t_start,
                             connection_requirements={
                                 'input_instrument': self.instrument_name(),
                                 'trigger': True})
            return [acquisition_pulse]

    def setup(self, **kwargs):
        for param in self._used_params:
            
    def start(self):
        self.daq_flush_multiple(2**9-1)
        self.daq_start_multiple(2**9-1)

    def stop(self):
        pass

