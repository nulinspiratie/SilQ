from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses.pulse_types import TriggerPulse

from qcodes.utils import validators as vals
from qcodes.instrument_drivers.keysight.SD_common.SD_acquisition_controller import *

class M3300A_DIG_Interface(InstrumentInterface):
    def __init__(self, instrument_name, acquisition_controller_names=[], **kwargs):
        super().__init__(instrument_name, **kwargs)
        self.pulse_sequence.allow_untargeted_pulses = True
        self.pulse_sequence.allow_pulse_overlap = True

        # Initialize channels
        self._acquisition_channels  = {
            'ch{}'.format(k): Channel(instrument_name=self.instrument_name(),
                                      name='ch{}'.format(k), id=k, input=True)
            for k in range(8)
            }

        self._channels = {
            **self._acquisition_channels ,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input=True),
        }

        # Organize acquisition controllers
        self.acquisition_controllers = {}
        for acquisition_controller_name in acquisition_controller_names:
            self.add_acquisition_controller(acquisition_controller_name)

        self.add_parameter(name='default_acquisition_controller',
                           parameter_class=ManualParameter,
                           initial_value='None',
                           vals=vals.Enum(None,
                               'None', *self.acquisition_controllers.keys()))

        self.add_parameter(name='acquisition_controller',
                           parameter_class=ManualParameter,
                           vals=vals.Enum(
                               'None', *self.acquisition_controllers.keys()))

        # Names of acquisition channels [chA, chB, etc.]
        self.add_parameter(name='acquisition_channels',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Anything())

        # Add ManualParameters which will be distributed to the active acquisition
        # controller during the setup routine
        self.add_parameter(
            'sample_rate',
            vals=Numbers(),
            parameter_class=ManualParameter
        )

        self.add_parameter(
            'samples',
            vals=Numbers(),
            parameter_class=ManualParameter
        )

        self.add_parameter(
            'channel_selection',
            vals=Anything(),
            parameter_class=ManualParameter
        )

        self.add_parameter(
            'trigger_channel',
            vals=Enum(0, 1, 2, 3, 4, 5, 6, 7),
            parameter_class=ManualParameter
        )

        self.add_parameter(
            'trigger_edge',
            vals=Enum('rising', 'falling', 'both'),
            parameter_class=ManualParameter
        )

        self.add_parameter(
            'trigger_threshold',
            parameter_class=ManualParameter
        )

        # Set up the driver to a known default state
        self.initialize_driver()

    @property
    def _acquisition_controller(self):
        return self.acquisition_controllers.get(
            self.acquisition_controller(), None)

    # Make all parameters of the interface transparent to the acquisition controller

    def acquisition(self):
        """
        Perform acquisition
        """
        data = {}
        acq_data = self._acquisition_controller.acquire()
        acq_data = self._acquisition_controller.post_acquire(acq_data)

        acquisition_average_mode = self._acquisition_controller.average_mode()

        # The start of acquisition
        t_0 = min(pulse.t_start for pulse in
                  self.input_pulse_sequence.get_pulses(acquire=True))

        # Split data into pulse traces
        for pulse in self.input_pulse_sequence.get_pulses(acquire=True):
            data[pulse.name] = {}
            for ch in self.channel_selection():
                ch_data = acq_data[ch]
                ch_name = 'ch{}'.format(ch)
                ts = (pulse.t_start - t_0, pulse.t_stop - t_0)
                sample_range = [int(t * self.sample_rate()) for t in ts]

                # Extract pulse data from the channel data
                if acquisition_average_mode == 'none':
                    data[pulse.name][ch_name] = ch_data[:, sample_range[0]:sample_range[1]]
                    # Further average the pulse data
                    if pulse.average == 'trace':
                        data[pulse.name][ch_name] = np.mean(data[pulse.name][ch_name], axis=0)
                    elif pulse.average == 'point':
                        data[pulse.name][ch_name] = np.mean(data[pulse.name][ch_name])

                elif acquisition_average_mode == 'trace':
                    data[pulse.name][ch_name] = ch_data[sample_range[0]:sample_range[1]]
                    # Further average the pulse data
                    if pulse.average == 'point':
                        data[pulse.name][ch_name] = np.mean(data[pulse.name][ch_name])

        # For instrument safety, stop all acquisition after we are done
        self.instrument.daq_stop_multiple(self._acquisition_controller._ch_array_to_mask( \
            self._acquisition_controller.channel_selection))

        return data

    def add_acquisition_controller(self, acquisition_controller_name,
                                   cls_name=None):
        """
        Adds an acquisition controller to the available controllers.
        If another acquisition controller exists of the same class, it will
        be overwritten.
        Args:
            acquisition_controller_name: instrument name of controller.
                Must be on same server as interface and Keysight digitizer
            cls_name: Optional name of class, which is used as controller key.
                If no cls_name is provided, it is found from the instrument
                class name

        Returns:
            None
        """
        acquisition_controller = self.find_instrument(
            acquisition_controller_name)
        if cls_name is None:
            cls_name = acquisition_controller.__class__.__name__
        # Remove _Controller from cls_name

        cls_name = cls_name.replace('_Controller', '')

        self.acquisition_controllers[cls_name] = acquisition_controller

    def initialize_driver(self):
        """
            Puts driver into a known initial state. Further configuration will
            be done in the configure_driver and get_final_additional_pulses
            functions.
        """
        for k in range(8):
            self.instrument.parameters['impedance_{}'.format(k)].set(1) # 50 Ohm impedance
            self.instrument.parameters['coupling_{}'.format(k)].set(0)  # DC Coupled
            self.instrument.parameters['full_scale_{}'.format(k)].set(3.0)  # 3.0 Volts

    def get_final_additional_pulses(self, **kwargs):
        if not self.input_pulse_sequence.get_pulses(acquire=True):
            # No pulses need to be acquired
            return []
        else:
            # Add a single trigger pulse when starting acquisition
            t_start = min(pulse.t_start for pulse in
                          self.input_pulse_sequence.get_pulses(acquire=True))

            acquisition_pulse = \
                TriggerPulse(t_start=t_start, duration=1e-5, acquire=True,
                             average='none',
                             connection_requirements={
                                 'input_instrument': self.instrument_name(),
                                 'trigger': True
                             })
            return [acquisition_pulse]

    def setup(self, **kwargs):
        controller = self._acquisition_controller
        self.samples(kwargs.pop('samples', 1))

        # Find all unique pulse_connections to choose which channels to acquire on
        channel_selection = {pulse.connection.input['channel'].id
                                  for pulse in self.input_pulse_sequence.get_pulses(acquire=True)}
        # import pdb; pdb.set_trace()
        self.channel_selection(sorted(list(channel_selection)))
        controller.channel_selection(self.channel_selection())
        # Acquire on all channels
        # controller.channel_selection = [x for x in range(8)]

        # Check what averaging mode is needed by each pulse
        if any(self.input_pulse_sequence.get_pulses(average='none')):
            controller.average_mode('none')
        else:
            controller.average_mode('trace')

        if controller() == 'Triggered':
            # Get trigger connection to determine how to trigger the controller
            trigger_pulse = self.input_pulse_sequence.get_pulse(trigger=True)
            trigger_connection = trigger_pulse.connection
            self.trigger_threshold(trigger_pulse.get_voltage(trigger_pulse.t_start) / 2)
            self.trigger_edge('rising')

            t_start = min(pulse.t_start for pulse in
                          self.input_pulse_sequence.get_pulses(acquire=True))
            t_stop = max(pulse.t_stop for pulse in
                         self.input_pulse_sequence.get_pulses(acquire=True))
            t_final = max(self.input_pulse_sequence.t_stop_list)

            T = t_stop - t_start

            controller.sample_rate(int(round((self.sample_rate()))))
            controller.traces_per_acquisition(int(round(self.samples())))

            controller.trigger_channel(trigger_connection.input['channel'].id)
            controller.trigger_threshold(self.trigger_threshold())
            # Map the string value of trigger edge to a device integer
            controller.trigger_edge(self.trigger_edge())

            # Capture maximum number of samples on all channels
            controller.samples_per_record(int(T * self.sample_rate()))

            # Set an acquisition timeout to be 10% after the last pulse finishes.
            # NOTE: time is defined in milliseconds
            controller.read_timeout(int(t_final * 1.1 * 1e3))

    def start(self):
        self._acquisition_controller.pre_start_capture()
        self._acquisition_controller.start()


    def acquire(self):
        data = {}
        acq_data = self._acquisition_controller.acquire()
        acq_data = self._acquisition_controller.post_acquire(acq_data)

        acquisition_average_mode = self._acquisition_controller.average_mode()

        # The start of acquisition
        t_0 = min(pulse.t_start for pulse in
                      self.input_pulse_sequence.get_pulses(acquire=True))

        # Split data into pulse traces
        for ch in self.channel_selection():
            data[ch] = {}
            ch_data = acq_data[ch]
            for pulse in self.input_pulse_sequence.get_pulses(acquire=True):
                ts = (pulse.t_start - t_0, pulse.t_stop - t_0)
                sample_range = [int(t * self.sample_rate()) for t in ts]

                # Extract pulse data from the channel data
                if acquisition_average_mode == 'none':
                    data[ch][pulse.name] = ch_data[:, sample_range[0]:sample_range[1]]
                    # Further average the pulse data
                    if pulse.average == 'trace':
                        data[ch][pulse.name] = np.mean(data[ch][pulse.name], axis=0)
                    elif pulse.average == 'point':
                        data[ch][pulse.name] = np.mean(data[ch][pulse.name])

                elif acquisition_average_mode == 'trace':
                    data[ch][pulse.name] = ch_data[sample_range[0]:sample_range[1]]
                    # Further average the pulse data
                    if pulse.average == 'point':
                        data[ch][pulse.name] = np.mean(data[ch][pulse.name])

        # For instrument safety, stop all acquisition after we are done
        self.instrument.daq_stop_multiple(self._acquisition_controller._ch_array_to_mask( \
                    self._acquisition_controller.channel_selection))
    
        return data

    def stop(self):
        self.instrument.daq_stop_multiple(self._acquisition_controller._ch_array_to_mask( \
            self._acquisition_controller.channel_selection))

