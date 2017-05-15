from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses.pulse_types import TriggerPulse

from qcodes.utils import validators as vals
from qcodes import ManualParameter
from qcodes.instrument_drivers.keysight.SD_common.SD_acquisition_controller import *
from qcodes.instrument_drivers.keysight.M3300A import M3300A_DIG as dig_driver


class M3300A_DIG_Interface(InstrumentInterface):
    def __init__(self, instrument_name, acquisition_controller_names=[], **kwargs):
        super().__init__(instrument_name, **kwargs)
        self._pulse_sequence.allow_untargeted_pulses = True

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

        # Organize acquisition controllers
        self.acquisition_controllers = {}
        for acquisition_controller_name in acquisition_controller_names:
            self.add_acquisition_controller(acquisition_controller_name)

        self._configuration_settings = {}
        self._acquisition_settings = {}

        # Obtain a list of all valid Keysight acquisition settings
        self._acquisition_settings_names = sorted(list(
            dig_driver.parameters.keys()))
        self._settings_names = sorted(self._acquisition_settings_names)

        self.add_parameter(name='acquisition_settings',
                           get_cmd=lambda: self._acquisition_settings)

        self.add_parameter(name="acquisition",
                           parameter_class=KeysightAcquisitionParameter)

        self.add_parameter(name='default_acquisition_controller',
                           parameter_class=ManualParameter,
                           initial_value='None',
                           vals=vals.Enum(None,
                               'None', *self.acquisition_controllers.keys()))

        self.add_parameter(name='acquisition_controller',
                           parameter_class=ManualParameter,
                           vals=vals.Enum(
                               'None', *self.acquisition_controllers.keys()))

        self.add_parameter(name='average_mode',
                           parameter_class=ManualParameter,
                           initial_value='trace',
                           vals=vals.Enum('none', 'trace', 'point'))

        # Names of acquisition channels [chA, chB, etc.]
        self.add_parameter(name='acquisition_channels',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Anything())

        self.add_parameter(name='samples',
                           parameter_class=ManualParameter,
                           initial_value=1)

        self.add_parameter(name='trigger_channel',
                           parameter_class=ManualParameter,
                           initial_value='trig_in',
                           vals=vals.Enum('trig_in', 'disable',
                                          *self._acquisition_channels.keys()))

        self.add_parameter(name='trigger_slope',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('positive', 'negative', 'both'))

        self.add_parameter(name='trigger_threshold',
                           unit='V',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

        # Obtain a list of all valid M3300A Digitizer parameters
        self._parameter_names = sorted(list(
            self.parameters.keys()))
        # Set up the driver to a known default state
        self.default_driver()

    def default_driver(self):
        self.sample_freq = 100e6;
        for k in range(8):
            self.instrument.parameters['impedance_{}'.format(k)].set(1)  # 50 Ohm impedance
            self.instrument.parameters['coupling_{}'.format(k)].set(0)  # DC Coupled
            self.instrument.parameters['full_scale_{}'.format(k)].set(3.0)  # 3.0 Volts

            # Configure the trigger behaviour
            # Trigger on rising edge of some channel
            # TODO: Read connections to figure out where to trigger from
            self.instrument.parameters['trigger_edge_{}'.format(k)].set(1)
            self.instrument.parameters['trigger_threshold_{}'.format(k)].set(0.5)
            # Select channel on which to trigger each DAQ
            self.instrument.parameters['analog_trigger_mask_{}'.format(k)].set(1 << 2)
            self.instrument.parameters['n_cycles_{}'.format(k)].set(-1)
            self.instrument.parameters['DAQ_trigger_delay_{}'.format(k)].set(0)
            self.instrument.parameters['DAQ_trigger_mode_{}'.format(k)].set(3)

    def configure_driver(self):
        """ Configures the underlying driver using interface parameters

            Args:
                None
            Return: 
                None
            Parameters: 
                average_mode
        """
        average_mode = self.average_mode()
        controller = self.acquisition_controller
        if controller() == 'Triggered_Controller':
            pass

        if (average_mode == 'none'):
            pass
        elif (average_mode == 'point'):
            pass
        elif (average_mode == 'trace'):
            pass

    def get_final_additional_pulses(self, **kwargs):
        if not self._pulse_sequence.get_pulses(acquire=True):
            # No pulses need to be acquired
            return []
        elif self.average_mode() == 'none':
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
                self.instrument.parameters['n_points_{}'.format(k)].set(int(T * self.sample_freq))
                # Set an acquisition timeout to be 10% after the last pulse finishes.
                self.instrument.parameters['timeout_{}'.format(k)].set(int(t_final * 1.1))

            acquisition_pulse = \
                TriggerPulse(t_start=t_start,
                             connection_requirements={
                                 'input_instrument': self.instrument_name(),
                                 'trigger': True})
            return [acquisition_pulse]

    def setup(self, **kwargs):
        pass
        # for param in self._used_params:

    def start(self):
        self.instrument.daq_flush_multiple(2 ** 9 - 1)
        self.instrument.daq_start_multiple(2 ** 9 - 1)

    def acquire(self):
        data = {}
        # Split data into pulse traces
        for pulse in self._pulse_sequence.get_pulses(acquire=True):
            data[pulse.name] = {}
            ts = (pulse.t_start, pulse.t_stop)
            sample_range = [int(t * self.sample_freq) for t in ts]
            for ch in range(8):
                ch_data = self.daq_read(ch)
                # Extract acquired data from the channel data
                data[pulse.name][ch] = ch_data[sample_range]
        return data

    def stop(self):
        pass

