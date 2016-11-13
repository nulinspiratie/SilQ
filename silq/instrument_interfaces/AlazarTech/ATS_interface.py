import numpy as np
import inspect

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from qcodes.instrument_drivers.AlazarTech.ATS import AlazarTech_ATS

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.tools import get_instrument_class
from silq.pulses import MeasurementPulse, SteeredInitialization, TriggerPulse,\
    MarkerPulse, PulseImplementation


class ATSInterface(InstrumentInterface):
    def __init__(self, instrument_name, acquisition_controller_names=[],
                 **kwargs):
        super().__init__(instrument_name, **kwargs)
        # Override untargeted pulse adding (measurement pulses can be added)
        self._pulse_sequence.allow_untargeted_pulses = True

        # Define channels
        self._acquisition_channels = {
            'ch'+idx: Channel(instrument_name=self.name,
                               name='ch'+idx,
                               id=idx, input=True)
            for idx in ['A', 'B', 'C', 'D']}
        self._trigger_in_channel = Channel(instrument_name=self.name,
                                           name='trig_in',
                                           input_trigger=True)
        self._aux_channels = {'aux'+idx: Channel(instrument_name=self.name,
                                                 name='aux'+idx,
                                                 input_TTL=True,
                                                 output_TTL=(0,5))
                             for idx in ['1', '2']}
        self._channels = {**self._acquisition_channels,
                         **self._aux_channels,
                         'trig_in': self._trigger_in_channel}

        # Organize acquisition controllers
        self.acquisition_controllers = {}
        for acquisition_controller_name in acquisition_controller_names:
            self.add_acquisition_controller(acquisition_controller_name)

        # Active acquisition controller is chosen during setup
        self.acquisition_controller = None

        self._configuration_settings = {}
        self._acquisition_settings = {}

        # Obtain a list of all valid ATS configuration settings
        self._configuration_settings_names = sorted(list(
            inspect.signature(AlazarTech_ATS.config).parameters.keys()))
        # Obtain a list of all valid ATS acquisition settings
        self._acquisition_settings_names = sorted(list(
            inspect.signature(AlazarTech_ATS.acquire).parameters.keys()))
        self._settings_names = sorted(self._acquisition_settings_names +
                                      self._configuration_settings_names)

        self.add_parameter(name='configuration_settings',
                           get_cmd=lambda: self._configuration_settings)
        self.add_parameter(name='acquisition_settings',
                           get_cmd=lambda: self._acquisition_settings)

        # Names and shapes must have initial value, even through they will be
        # overwritten in set_acquisition_settings. If we don't do this, the
        # remoteInstrument will not recognize that it returns multiple values.
        self.add_parameter(name="acquisition",
                           names=['channel_signal'],
                           get_cmd=self._acquisition,
                           shapes=((),),
                           snapshot_value=False)

        self.add_parameter(name='acquisition_mode',
                           parameter_class=ManualParameter,
                           initial_value='trigger',
                           vals=vals.Enum('trigger', 'continuous'))

        self.add_parameter(name='active_acquisition_controller',
                           get_cmd=lambda: (
                               self.acquisition_controller.name if
                               self.acquisition_controller is not None
                               else 'None'),
                           vals=vals.Enum('None',
                                          *self.acquisition_controllers.keys()))

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
                           vals=vals.Enum('positive', 'negative'))
        self.add_parameter(name='trigger_threshold',
                           units='V',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

    def add_acquisition_controller(self, acquisition_controller_name,
                                   cls_name=None):
        """
        Adds an acquisition controller to the available controllers.
        If another acquisition controller exists of the same class, it will
        be overwritten.
        Args:
            acquisition_controller_name: instrument name of controller.
                Must be on same server as interface and ATS
            cls_name: Optional name of class, which is used as controller key.
                If no cls_name is provided, it is found from the instrument
                class name

        Returns:
            None
        """
        acquisition_controller = self.find_instrument(
            acquisition_controller_name)
        if cls_name is None:
            cls_name = get_instrument_class(acquisition_controller)
        # Remove _AcquisitionController from cls_name
        cls_name = cls_name.replace('_AcquisitionController', '')

        self.acquisition_controllers[cls_name] = acquisition_controller

    def get_final_additional_pulses(self):
        if not self._pulse_sequence.get_pulses(acquire=True):
            return []
        elif self.trigger_mode() == 'trigger':
            # Add a single trigger pulse when starting acquisition
            t_start = min(pulse.t_start for pulse in
                          self._pulse_sequence.get_pulses(acquire=True))
            acquisition_pulse = \
                TriggerPulse(t_start=t_start,
                             connection_requirements={
                                 'input_instrument': self.instrument_name(),
                                 'trigger': True})
        elif self.trigger_mode() == 'continuous':
            # TODO add possibility of continuous acquisition having correct
            # timing even if it should acquire for the entire duration of the
            #  pulse sequence.
            t_start = min(pulse.t_start for pulse in
                          self._pulse_sequence.get_pulses(acquire=True))
            t_stop = max(pulse.t_stop for pulse in
                         self._pulse_sequence.get_pulses(acquire=True))
            if t_start != 0 or t_stop != self._pulse_sequence.duration:
                # Add a marker high for readout stage
                # Get steered initialization pulse
                initialization = self._pulse_sequence.get_pulse(initialize=True)

                acquisition_pulse = MarkerPulse(
                    t_start=t_start, t_stop=t_stop,
                    connection_requirements={
                        'connection': initialization.trigger_connection})
        return [acquisition_pulse]

    def setup(self, samples=None, average_mode=None, connections=None,
              readout_threshold_voltage=None,
              **kwargs):
        self._configuration_settings.clear()
        self._acquisition_settings.clear()

        # Determine the acquisition controller to use
        if self.acquisition_mode() == 'trigger':
            self.acquisition_controller_name = 'Triggered'
        elif self.acquisition_mode() == 'continuous':
            if self._pulse_sequence.get_pulse(
                    pulse_class=SteeredInitialization) is not None:
                # Use steered initialization
                self.acquisition_controller_name = 'SteeredInitialization'
            else:
                self.acquisition_controller_name = 'Continuous'
        else:
            raise Exception('Acquisition mode {} not implemented'.format(
                self.acquisition_mode()))

        self.acquisition_controller = \
            self.acquisition_controllers[self.acquisition_controller_name]

        if samples is not None:
            self.samples(samples)

        if average_mode is not None:
            self.average_mode(average_mode)
            self.acquisition_controller.average_mode(average_mode)

        self.setup_trigger()
        self.setup_ATS()
        self.setup_acquisition_controller(
            readout_threshold_voltage=readout_threshold_voltage)

        # Update acquisition metadata
        self.acquisition.names = self.acquisition_controller.acquisition.names
        self.acquisition.labels = self.acquisition_controller.acquisition.labels
        self.acquisition.units = self.acquisition_controller.acquisition.units
        self.acquisition.shapes = self.acquisition_controller.acquisition.shapes

    def setup_trigger(self):
        if self.acquisition_mode() == 'trigger':
            if self.trigger_channel() == 'trig_in':
                self.update_settings(external_trigger_range=5)
                trigger_range = 5
            else:
                trigger_channel = self._acquisition_channels[
                    self.trigger_channel()]
                trigger_id = trigger_channel.id
                trigger_range = self.setting('channel_range' + trigger_id)

            trigger_pulses = self._input_pulse_sequence.get_pulses(
                input_channel=self.trigger_channel())
            if trigger_pulses:
                trigger_pulse = min(trigger_pulses, key=lambda p: p.t_start)
                pre_voltage, post_voltage = \
                    self._input_pulse_sequence.get_transition_voltages(
                        pulse=trigger_pulse)
                assert post_voltage != pre_voltage, \
                    'Could not determine trigger voltage transition'

                self.trigger_slope('positive' if post_voltage > pre_voltage
                                   else 'negative')
                self.trigger_threshold((pre_voltage + post_voltage) / 2)
                # Trigger level is between 0 (-trigger_range)
                # and 255 (+trigger_range)
                trigger_level = int(128 + 127 * (self.trigger_threshold() /
                                                 trigger_range))

                self.update_settings(trigger_operation='J',
                                     trigger_engine1='J',
                                     trigger_source1=self.trigger_channel(),
                                     trigger_slope1=self.trigger_slope(),
                                     trigger_level1=trigger_level,
                                     external_trigger_coupling='DC',
                                     trigger_delay=0)
            else:
                print('Cannot setup ATS trigger because there are no '
                      'acquisition pulses')
        else:
            raise Exception('Acquisition mode {} not implemented'.format(
                self.acquisition_mode()
            ))

    def setup_ATS(self):
        # Setup ATS configuration

        self.update_settings(channel_range=2,
                             coupling='DC')
        self.instrument.config(**self._configuration_settings)

    def setup_acquisition_controller(self, readout_threshold_voltage=None):
        # Get duration of acquisition. Use flag acquire=True because
        # otherwise initialization Pulses would be taken into account as well
        t_start = min(pulse.t_start for pulse in
                      self._pulse_sequence.get_pulses(acquire=True))
        t_stop = max(pulse.t_stop for pulse in
                     self._pulse_sequence.get_pulses(acquire=True))
        acquisition_duration = t_stop - t_start

        sample_rate = self.setting('sample_rate')
        samples_per_trace = sample_rate * acquisition_duration * 1e-3
        # samples_per_record must be a multiple of 16
        samples_per_trace = int(16 * np.ceil(float(samples_per_trace) / 16))
        if self.acquisition_controller_name == 'Triggered':
            # TODO Allow variable records_per_buffer
            records_per_buffer = 1
            buffers_per_acquisition = self.samples()
            self.update_settings(samples_per_record=samples_per_trace,
                                 records_per_buffer=records_per_buffer,
                                 buffers_per_acquisition=buffers_per_acquisition)

        elif self.acquisition_controller_name == 'Continuous':
            # records_per_buffer and buffers_per_acquisition are fixed
            self._acquisition_settings.pop('records_per_buffer', None)
            self._acquisition_settings.pop('buffers_per_acquisition', None)

            # TODO better way to decide on allocated buffers
            allocated_buffers = 20
            self.update_settings(allocated_buffers=allocated_buffers)

            self.acquisition_controller.samples_per_trace(samples_per_trace)
            self.acquisition_controller.traces_per_acquisition(self.samples())

        elif self.acquisition_controller_name == 'SteeredInitialization':
            # records_per_buffer and buffers_per_acquisition are fixed
            self._acquisition_settings.pop('records_per_buffer', None)
            self._acquisition_settings.pop('buffers_per_acquisition', None)

            # Get steered initialization pulse
            initialization = self._pulse_sequence.get_pulse(initialize=True)

            # TODO better way to decide on allocated buffers
            allocated_buffers = 20

            samples_per_buffer = sample_rate * initialization.t_buffer * 1e-3
            # samples_per_record must be a multiple of 16
            samples_per_buffer = int(16 * np.ceil(float(samples_per_buffer) / 16))
            self.update_settings(samples_per_record=samples_per_buffer,
                                 allocated_buffers=allocated_buffers)

            initialization.implement(
                interface=self,
                readout_threshold_voltage=readout_threshold_voltage)
            for channel in [initialization.trigger_channel,
                            initialization.readout_channel]:
                assert channel.name in self.acquisition_channels(), \
                    "Channel {} must be in acquisition channels".format(channel)

            self.acquisition_controller.samples_per_trace(samples_per_trace)
            self.acquisition_controller.traces_per_acquisition(self.samples())
        else:
            raise Exception("Cannot setup {} acquisition controller".format(
                self.acquisition_controller_name))

        # Set acquisition channels setting
        # Channel_selection must be a sorted string of acquisition channel ids
        channel_ids = ''.join(sorted(
            [self._channels[ch].id for ch in self.acquisition_channels()]))
        buffer_timeout = max(20000, 3.1 * self._pulse_sequence.duration)
        self.update_settings(channel_selection=channel_ids,
                             buffer_timeout=buffer_timeout)  # ms

        # Update settings in acquisition controller
        self.acquisition_controller.set_acquisition_settings(
            **self._acquisition_settings)
        self.acquisition_controller.average_mode(self.average_mode())
        self.acquisition_controller.setup()

    def start(self):
        pass

    def stop(self):
        pass

    def _acquisition(self):
        return self.acquisition_controller.acquisition()

    def setting(self, setting):
        """
        Obtain a setting for the ATS.
        It first checks if the setting is an actual ATS kwarg, and raises an
        error otherwise.
        It then checks if it is a configuration or acquisition setting.
        If the setting is specified in self.configuration/acquisition_setting,
        it returns that value, else it returns the value set in the ATS

        Args:
            setting: configuration or acquisition setting to look for

        Returns:
            Value of the setting
        """
        assert setting in self._settings_names, \
            "Kwarg {} is not a valid ATS acquisition setting".format(setting)
        if setting in self._configuration_settings.keys():
            return self._configuration_settings[setting]
        elif setting in self._acquisition_settings.keys():
            return self._acquisition_settings[setting]
        else:
            # Must get latest value, since it may not be updated in ATS
            return self.instrument.parameters[setting]()

    def set_configuration_settings(self, **settings):
        """
        Sets the configuration settings for the ATS through its controller.
        It additionally checks if the settings are all actual ATS configuration
        settings, and raises an error otherwise.

        Args:
            settings: configuration settings for the acquisition controller

        Returns:
            None
        """
        assert all([setting in self._configuration_settings_names
                    for setting in settings]), \
            "Settings are not all valid ATS configuration settings"
        self._configuration_settings = settings

    def set_acquisition_settings(self, **settings):
        """
        Sets the acquisition settings for the ATS through its controller.
        It additionally checks if the settings are all actual ATS acquisition
        settings, and raises an error otherwise.

        Args:
            settings: acquisition settings for the acquisition controller

        Returns:
            None
        """
        assert all([setting in self._acquisition_settings_names
                    for setting in settings]), \
            "Settings are not all valid ATS acquisition settings"
        self._acquisition_settings = settings

    def update_settings(self, **settings):
        settings_valid = all(map(
            lambda setting: setting in self._settings_names, settings.keys()))
        assert settings_valid, \
            'Not all settings are valid ATS settings. Settings: {}\n' \
            'Valid ATS settings: {}'.format(settings, self._settings_names)

        configuration_settings = {k: v for k, v in settings.items()
                                  if k in self._configuration_settings_names}
        self._configuration_settings.update(**configuration_settings)

        acquisition_settings = {k: v for k, v in settings.items()
                                  if k in self._acquisition_settings_names}
        self._acquisition_settings.update(**acquisition_settings)


class SteeredInitializationImplementation(SteeredInitialization,
                                          PulseImplementation):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=SteeredInitialization,
                                     **kwargs)

    def target_pulse(self, pulse, interface, connections, **kwargs):
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Add readout connection to targeted pulse
        readout_connection = [connection for connection in connections if
                              connection.satisfies_conditions(
                                  output_arg='chip.output')]
        assert len(readout_connection) == 1, \
            "No unique readout connection: {}".format(readout_connection)
        targeted_pulse.readout_connection = readout_connection[0]

        # Add trigger connection to targeted pulse
        trigger_connection = [connection for connection in connections if
                              connection.satisfies_conditions(
                                  input_channel=['chA', 'chB', 'chC', 'chD'],
                                  trigger=True)]
        assert len(trigger_connection) == 1, \
            "No unique triggerconnection: {}".format(trigger_connection)
        targeted_pulse.trigger_connection = trigger_connection[0]
        return targeted_pulse

    def implement(self, interface, readout_threshold_voltage):
        acquisition_controller = interface.acquisition_controller
        acquisition_controller.t_max_wait(self.t_max_wait)
        acquisition_controller.t_no_blip(self.t_no_blip)

        # Setup readout channel and threshold voltage
        self.readout_channel = self.readout_connection.input['channel']
        acquisition_controller.readout_channel(self.readout_channel.id)
        acquisition_controller.readout_threshold_voltage(
            readout_threshold_voltage)

        # Setup trigger channel and threshold voltage
        self.trigger_channel = self.trigger_connection.input['channel']
        TTL_voltages = self.trigger_connection.output['channel'].output_TTL
        trigger_threshold_voltage = (TTL_voltages[0] + TTL_voltages[1]) / 2
        acquisition_controller.trigger_channel(self.trigger_channel.id)
        acquisition_controller.trigger_threshold_voltage(
            trigger_threshold_voltage)
