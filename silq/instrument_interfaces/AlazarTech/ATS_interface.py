import numpy as np
import inspect
import logging
from functools import partial

from qcodes.instrument.parameter import ManualParameter, StandardParameter
from qcodes.utils import validators as vals
from qcodes.instrument_drivers.AlazarTech.ATS import AlazarTech_ATS, \
    ATSAcquisitionParameter

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import MeasurementPulse, SteeredInitialization, TriggerPulse,\
    MarkerPulse, TriggerWaitPulse, PulseImplementation


class ATSInterface(InstrumentInterface):
    def __init__(self, instrument_name, acquisition_controller_names=[],
                 **kwargs):
        super().__init__(instrument_name, **kwargs)
        # Override untargeted pulse adding (measurement pulses can be added)
        self.pulse_sequence.allow_untargeted_pulses = True

        # Define channels
        self._acquisition_channels = {
            'ch'+idx: Channel(instrument_name=self.instrument_name(),
                               name='ch'+idx,
                               id=idx, input=True)
            for idx in ['A', 'B', 'C', 'D']}
        self._aux_channels = {'aux'+idx: Channel(
            instrument_name=self.instrument_name(),
            name='aux'+idx,
            input_TTL=True,
            output_TTL=(0,5))
                             for idx in ['1', '2']}
        self._channels = {
            **self._acquisition_channels,
            **self._aux_channels,
            'trig_in':  Channel(instrument_name=self.instrument_name(),
                                name='trig_in', input_trigger=True),
            'software_trig_out': Channel(instrument_name=self.instrument_name(),
                                         name='software_trig_out')}

        self.pulse_implementations = [
            SteeredInitializationImplementation(
                pulse_requirements=[]
            ),
            TriggerWaitPulseImplementation(
                pulse_requirements=[]
            )
        ]

        # Organize acquisition controllers
        self.acquisition_controllers = {}
        for acquisition_controller_name in acquisition_controller_names:
            self.add_acquisition_controller(acquisition_controller_name)

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

        self.add_parameter(name="acquisition",
                           parameter_class=ATSAcquisitionParameter)

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
                           unit='V',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter(name='sample_rate',
                           unit='samples/sec',
                           parameter_class=StandardParameter,
                           get_cmd=partial(self.setting, 'sample_rate'),
                           set_cmd=lambda x:self.update_settings(sample_rate=x),
                           vals=vals.Numbers())

    @property
    def _acquisition_controller(self):
        return self.acquisition_controllers.get(
            self.acquisition_controller(), None)

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
            cls_name = acquisition_controller.__class__.__name__
        # Remove _AcquisitionController from cls_name
        cls_name = cls_name.replace('_AcquisitionController', '')

        self.acquisition_controllers[cls_name] = acquisition_controller

        # Update possible values for (default) acquisition controller
        self.default_acquisition_controller._vals = vals.Enum(
            'None', *self.acquisition_controllers.keys())
        self.acquisition_controller._vals = vals.Enum(
            'None', *self.acquisition_controllers.keys())

    def get_additional_pulses(self, interface, **kwargs):
        if not self.pulse_sequence.get_pulses(acquire=True):
            # No pulses need to be acquired
            return []
        elif self.acquisition_controller() == 'Triggered':
            # Add a single trigger pulse when starting acquisition
            t_start = min(pulse.t_start for pulse in
                          self.pulse_sequence.get_pulses(acquire=True))
            return [
                TriggerPulse(t_start=t_start,
                             connection_requirements={
                                 'input_instrument': self.instrument_name(),
                                 'trigger': True})]
        elif self.acquisition_controller() == 'Continuous':
            raise NotImplementedError("Continuous mode not implemented")
        elif self.acquisition_controller() == 'SteeredInitialization':
            # TODO add possibility of continuous acquisition having correct
            # timing even if it should acquire for the entire duration of the
            #  pulse sequence.
            t_start = min(pulse.t_start for pulse in
                          self.pulse_sequence.get_pulses(acquire=True))
            t_stop = max(pulse.t_stop for pulse in
                         self.pulse_sequence.get_pulses(acquire=True))
            # Add a marker high for readout stage
            # Get steered initialization pulse
            initialization = self.pulse_sequence.get_pulse(initialize=True)

            acquisition_pulse = MarkerPulse(
                t_start=t_start, t_stop=t_stop,
                connection_requirements={
                    'connection': initialization.trigger_connection})
            trigger_wait_pulse = TriggerWaitPulse(
                t_start=self.pulse_sequence.duration,
                connection_requirements={
                    'output_arg': 'ATS.software_trig_out'})

            return [acquisition_pulse, trigger_wait_pulse]

    def initialize(self):
        """
        This method gets called at the start of targeting a pulse sequence
        Returns:
            None
        """
        super().initialize()
        self.acquisition_controller(self.default_acquisition_controller())

    def setup(self, samples=None, connections=None, **kwargs):
        self._configuration_settings.clear()
        self._acquisition_settings.clear()

        if samples is not None:
            self.samples(samples)

        self.setup_trigger()
        self.setup_ATS()
        self.setup_acquisition_controller()

        if self.acquisition_controller() == 'SteeredInitialization':
            # Add instruction for target instrument setup and to skip start
            target_instrument = self._acquisition_controller.target_instrument()
            return {'skip_start': target_instrument}

    def setup_trigger(self):
        if self.acquisition_controller() == 'Triggered':
            if self.trigger_channel() == 'trig_in':
                self.update_settings(external_trigger_range=5)
                trigger_range = 5
            else:
                trigger_channel = self._acquisition_channels[
                    self.trigger_channel()]
                trigger_id = trigger_channel.id
                trigger_range = self.setting('channel_range' + trigger_id)

            trigger_pulses = self.input_pulse_sequence.get_pulses(
                input_channel=self.trigger_channel())
            if trigger_pulses:
                trigger_pulse = min(trigger_pulses, key=lambda p: p.t_start)
                pre_voltage, post_voltage = \
                    self.input_pulse_sequence.get_transition_voltages(
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
            pass

    def setup_ATS(self):
        # Setup ATS configuration

        self.update_settings(channel_range=2,
                             coupling='DC')
        self.instrument.config(**self._configuration_settings)

    def setup_acquisition_controller(self):
        # Get duration of acquisition. Use flag acquire=True because
        # otherwise initialization Pulses would be taken into account as well
        t_start = min(pulse.t_start for pulse in
                      self.pulse_sequence.get_pulses(acquire=True))
        t_stop = max(pulse.t_stop for pulse in
                     self.pulse_sequence.get_pulses(acquire=True))
        acquisition_duration = t_stop - t_start

        samples_per_trace = self.sample_rate() * acquisition_duration * 1e-3
        if self.acquisition_controller() == 'Triggered':
            # samples_per_record must be a multiple of 16
            samples_per_record = int(16 * np.ceil(float(samples_per_trace) / 16))
            # TODO Allow variable records_per_buffer
            records_per_buffer = 1
            buffers_per_acquisition = self.samples()

            if buffers_per_acquisition > 1:
                allocated_buffers = 2
            else:
                allocated_buffers = 1

            self.update_settings(samples_per_record=samples_per_record,
                                 records_per_buffer=records_per_buffer,
                                 buffers_per_acquisition=buffers_per_acquisition,
                                 allocated_buffers=allocated_buffers)
        elif self.acquisition_controller() == 'Continuous':
            # records_per_buffer and buffers_per_acquisition are fixed
            self._acquisition_settings.pop('records_per_buffer', None)
            self._acquisition_settings.pop('buffers_per_acquisition', None)

            # TODO better way to decide on allocated buffers
            allocated_buffers = 20
            self.update_settings(allocated_buffers=allocated_buffers)

            # samples_per_trace must be a multiple of samples_per_record
            samples_per_trace = int(16 * np.ceil(float(samples_per_trace) / 16))
            self._acquisition_controller.samples_per_trace(samples_per_trace)
            self._acquisition_controller.traces_per_acquisition(self.samples())
        elif self.acquisition_controller() == 'SteeredInitialization':
            # records_per_buffer and buffers_per_acquisition are fixed
            self._acquisition_settings.pop('records_per_buffer', None)
            self._acquisition_settings.pop('buffers_per_acquisition', None)

            # Get steered initialization pulse
            initialization = self.pulse_sequence.get_pulse(initialize=True)

            # TODO better way to decide on allocated buffers
            allocated_buffers = 80

            samples_per_buffer = self.sample_rate() * \
                                 initialization.t_buffer * 1e-3
            # samples_per_record must be a multiple of 16
            samples_per_buffer = int(16 * np.ceil(float(samples_per_buffer) / 16))
            self.update_settings(samples_per_record=samples_per_buffer,
                                 allocated_buffers=allocated_buffers)

            # Setup acquisition controller settings through initialization pulse
            initialization.implement()
            for channel in [initialization.trigger_channel,
                            initialization.readout_channel]:
                assert channel.name in self.acquisition_channels(), \
                    "Channel {} must be in acquisition channels".format(channel)

            # samples_per_trace must be a multiple of samples_per_buffer
            samples_per_trace = int(samples_per_buffer * np.ceil(
                float(samples_per_trace) / samples_per_buffer))
            self._acquisition_controller.samples_per_trace(samples_per_trace)
            self._acquisition_controller.traces_per_acquisition(self.samples())
        else:
            raise Exception("Cannot setup {} acquisition controller".format(
                self.acquisition_controller()))

        # Set acquisition channels setting
        # Channel_selection must be a sorted string of acquisition channel ids
        channel_ids = ''.join(sorted(
            [self._channels[ch].id for ch in self.acquisition_channels()]))
        if len(channel_ids) == 3:
            # TODO add 'silent' mode
            # logging.warning("ATS cannot be configured with three acquisition "
            #                 "channels {}, setting to ABCD".format(channel_ids))
            channel_ids = 'ABCD'
        buffer_timeout = int(max(20000, 3.1 * self.pulse_sequence.duration))
        self.update_settings(channel_selection=channel_ids,
                             buffer_timeout=buffer_timeout)  # ms

        # Update settings in acquisition controller
        self._acquisition_controller.set_acquisition_settings(
            **self._acquisition_settings)
        self._acquisition_controller.average_mode('none')
        self._acquisition_controller.setup()

    def start(self):
        pass

    def stop(self):
        pass

    def acquisition(self):
        traces = self._acquisition_controller.acquisition()
        traces_dict = {
            ch: trace for ch, trace in zip(self.acquisition_channels(), traces)}
        pulse_traces = self.segment_traces(traces_dict)
        return pulse_traces

    def segment_traces(self, traces):
        pulse_traces = {}
        t_start_initial = min(p.t_start for p in
                              self.pulse_sequence.get_pulses(acquire=True))
        for pulse in self.pulse_sequence.get_pulses(acquire=True):
            delta_t_start = pulse.t_start - t_start_initial
            start_idx = int(round(delta_t_start / 1e3 * self.sample_rate()))
            pts = int(round(pulse.duration / 1e3 * self.sample_rate()))

            pulse_traces[pulse.full_name] = {}
            for ch, trace in traces.items():
                pulse_trace = trace[:, start_idx:start_idx + pts]
                if pulse.average == 'point':
                    pulse_traces[pulse.full_name][ch] = np.mean(pulse_trace)
                elif pulse.average == 'trace':
                    pulse_traces[pulse.full_name][ch] = np.mean(pulse_trace, 0)
                elif 'point_segment' in pulse.average:
                    segments = int(pulse.average.split(':')[1])

                    segments_idx = [int(round(pts * idx / segments))
                                    for idx in np.arange(segments + 1)]

                    pulse_traces[pulse.full_name][ch] = np.zeros(segments)
                    for k in range(segments):
                        pulse_traces[pulse.full_name][ch][k] = np.mean(
                            pulse_trace[:, segments_idx[k]:segments_idx[k + 1]])
                elif 'trace_segment' in pulse.average:
                    segments = int(pulse.average.split(':')[1])

                    segments_idx = [int(round(pts * idx / segments))
                                    for idx in np.arange(segments + 1)]

                    pulse_traces[pulse.full_name][ch] = np.zeros(segments)
                    for k in range(segments):
                        pulse_traces[pulse.full_name][ch][k] = \
                            pulse_trace[:, segments_idx[k]:segments_idx[k + 1]]
                elif pulse.average == 'none':
                    pulse_traces[pulse.full_name][ch] = pulse_trace
                else:
                    raise SyntaxError(f'Unknown average mode {pulse.average}')
        return pulse_traces

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


class SteeredInitializationImplementation(PulseImplementation):
    pulse_class = SteeredInitialization

    def target_pulse(self, pulse, interface, connections, **kwargs):
        targeted_pulse = super().target_pulse(pulse, interface, **kwargs)

        # Add readout connection to targeted pulse
        readout_connection = [connection for connection in connections if
                              connection.satisfies_conditions(
                                  output_arg='chip.output')]
        assert len(readout_connection) == 1, \
            f"No unique readout connection: {readout_connection}"
        targeted_pulse.implementation.readout_connection = readout_connection[0]

        # Add trigger connection to targeted pulse
        trigger_connection = [connection for connection in connections if
                              connection.satisfies_conditions(
                                  input_channel=['chA', 'chB', 'chC', 'chD'],
                                  trigger=True)]
        assert len(trigger_connection) == 1, \
            f"No unique trigger connection: {trigger_connection}"
        targeted_pulse.implementation.trigger_connection = trigger_connection[0]

        # Force ATS acquisition mode to be continuous
        interface.acquisition_controller('SteeredInitialization')
        return targeted_pulse

    def implement(self, interface):
        acquisition_controller = interface._acquisition_controller
        acquisition_controller.t_max_wait(self.pulse.t_max_wait)
        acquisition_controller.t_no_blip(self.pulse.t_no_blip)

        # Setup readout channel and threshold voltage
        self.readout_channel = self.readout_connection.input['channel']
        acquisition_controller.readout_channel(self.readout_channel.id)
        acquisition_controller.readout_threshold_voltage(
            self.pulse.readout_threshold_voltage)

        # Setup trigger channel and threshold voltage
        self.trigger_output_channel = self.trigger_connection.output['channel']
        TTL_voltages = self.trigger_output_channel.output_TTL
        trigger_threshold_voltage = (TTL_voltages[0] + TTL_voltages[1]) / 2

        self.trigger_channel = self.trigger_connection.input['channel']
        acquisition_controller.trigger_channel(self.trigger_channel.id)
        acquisition_controller.trigger_threshold_voltage(
            trigger_threshold_voltage)


class TriggerWaitPulseImplementation(PulseImplementation):
    pulse_class = TriggerWaitPulse

    def target_pulse(self, pulse, interface, **kwargs):
        # Verify that acquisition controller is SteeredInitialization
        if not interface.acquisition_controller() == 'SteeredInitialization':
            raise RuntimeError('ATS interface acquisition controller is not '
                               'SteeredInitialization')
        return super().target_pulse(pulse, interface, **kwargs)

    def implement(self, **kwargs):
        pass

