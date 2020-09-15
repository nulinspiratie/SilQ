import numpy as np
import inspect
import logging
from functools import partial
from typing import List, Union, Dict
from time import sleep

from qcodes.utils import validators as vals
from qcodes.instrument_drivers.AlazarTech.ATS import AlazarTech_ATS, \
    ATSAcquisitionParameter
from qcodes.station import Station
from qcodes.instrument.base import Instrument

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import Pulse, SteeredInitialization, TriggerPulse,\
    MarkerPulse, TriggerWaitPulse, PulseImplementation


logger = logging.getLogger(__name__)

class ATSInterface(InstrumentInterface):
    """Interface for the AlazarTech ATS.

    When a `PulseSequence` is targeted in the `Layout`, the
    pulses are directed to the appropriate interface. Each interface is
    responsible for translating all pulses directed to it into instrument
    commands. During the actual measurement, the instrument's operations will
    correspond to that required by the pulse sequence.

    Args:
        instrument_name: Name of ATS instrument.
        acquisition_controller_names: Instrument names of all ATS
            acquisition controllers. Interface will find the associated
            acquisition controllers.
        default_settings: Default settings to use for the ATS.
        **kwargs: Additional kwargs passed to Instrument.

    Notes:
        * Only been tested on ATS9440, might give issues with other models,
          in particular those having 2 channels instead of 4
        * For a given instrument, its associated interface can be found using
          `get_instrument_interface`

    Todo:
        * Choose continuous acquisition controller if pulse sequence only
          consists of a measurement pulse, as this doesn't require a trigger
          from another instrument
    """
    def __init__(self,
                 instrument_name: str,
                 acquisition_controller_names: List[str] = [],
                 default_settings={},
                 **kwargs):
        # TODO: Change acquisition_controller_names to acquisition_controllers
        super().__init__(instrument_name, **kwargs)
        # Override untargeted pulse adding (measurement pulses can be added)
        self.pulse_sequence.allow_untargeted_pulses = True

        default_settings = {'channel_range': 2, 'coupling': 'DC', **default_settings}

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
                                name='trig_in', id='trig_in', input_trigger=True),
            'software_trig_out': Channel(instrument_name=self.instrument_name(),
                                         name='software_trig_out')}

        self.pulse_implementations = [
            SteeredInitializationImplementation(
                pulse_requirements=[]),
            TriggerWaitPulseImplementation(
                pulse_requirements=[])]

        # Organize acquisition controllers
        self.acquisition_controllers = {}
        for acquisition_controller_name in acquisition_controller_names:
            self.add_acquisition_controller(acquisition_controller_name)

        # Obtain a list of all valid ATS configuration settings
        self._configuration_settings_names = sorted(list(
            inspect.signature(AlazarTech_ATS.config).parameters.keys()))
        # Obtain a list of all valid ATS acquisition settings
        self._acquisition_settings_names = sorted(list(
            inspect.signature(AlazarTech_ATS.acquire).parameters.keys()))
        self._settings_names = sorted(self._acquisition_settings_names +
                                      self._configuration_settings_names)

        self.add_parameter(name='default_settings',
                           get_cmd=None, set_cmd=None,
                           initial_value=default_settings,
                           docstring='Default settings to use when setting up '
                                     'ATS for a pulse sequence')
        initial_configuration_settings = {k: v for k, v in default_settings.items()
                                          if k in self._configuration_settings_names}
        self.add_parameter(name='configuration_settings',
                           get_cmd=None,
                           set_cmd=None,
                           vals=vals.Dict(allowed_keys=self._configuration_settings_names),
                           initial_value=initial_configuration_settings)
        initial_acquisition_settings = {k: v for k, v in default_settings.items()
                                        if k in self._acquisition_settings_names}
        self.add_parameter(name='acquisition_settings',
                           get_cmd=None,
                           set_cmd=None,
                           vals=vals.Dict(allowed_keys=self._acquisition_settings_names),
                           initial_value=initial_acquisition_settings)

        self.add_parameter(name="acquisition",
                           parameter_class=ATSAcquisitionParameter)

        self.add_parameter(name='default_acquisition_controller',
                           set_cmd=None,
                           initial_value='None',
                           vals=vals.Enum(None,
                               'None', *self.acquisition_controllers.keys()))

        self.add_parameter(name='acquisition_controller',
                           set_cmd=None,
                           vals=vals.Enum(
                               'None', *self.acquisition_controllers.keys()))

        # Names of acquisition channels [chA, chB, etc.]
        self.add_parameter(name='acquisition_channels',
                           set_cmd=None,
                           initial_value=[],
                           vals=vals.Anything(),
                           docstring='Names of acquisition channels '
                                     '[chA, chB, etc.]. Set by the layout')

        self.add_parameter(name='samples',
                           set_cmd=None,
                           initial_value=1,
                           docstring='Number of times to acquire the pulse '
                                     'sequence.')

        self.add_parameter('points_per_trace',
                           get_cmd=lambda: self._acquisition_controller.samples_per_trace(),
                           docstring='Number of points in a trace.')

        self.add_parameter(name='trigger_channel',
                           set_cmd=None,
                           initial_value='trig_in',
                           vals=vals.Enum('trig_in', 'disable',
                                          *self._acquisition_channels.keys()))
        self.add_parameter(name='trigger_slope',
                           set_cmd=None,
                           vals=vals.Enum('positive', 'negative'))
        self.add_parameter(name='trigger_threshold',
                           unit='V',
                           set_cmd=None,
                           vals=vals.Numbers())
        self.add_parameter(name='sample_rate',
                           unit='samples/sec',
                           get_cmd=partial(self.setting, 'sample_rate'),
                           set_cmd=lambda x:self.default_settings().update(sample_rate=x),
                           vals=vals.Numbers(),
                           docstring='Acquisition sampling rate (Hz)')

        self.add_parameter('capture_full_trace',
                           initial_value=False,
                           vals=vals.Bool(),
                           set_cmd=None,
                           docstring='Capture from t=0 to end of pulse '
                                     'sequence. False by default, in which '
                                     'case start and stop times correspond to '
                                     'min(t_start) and max(t_stop) of all '
                                     'pulses with the flag acquire=True, '
                                     'respectively.')

        self.traces = {}
        self.pulse_traces = {}


    @property
    def _acquisition_controller(self):
        """Active acquisition controller"""
        return self.acquisition_controllers.get(
            self.acquisition_controller(), None)

    def add_acquisition_controller(self,
                                   acquisition_controller_name: str,
                                   cls_name: Union[str, None] = None):
        """Add an acquisition controller to the available controllers.

        If another acquisition controller exists of the same class, it will
        be overwritten.

        Args:
            acquisition_controller_name: instrument name of controller.
                Must be on same server as interface and ATS
            cls_name: Optional name of class, which is used as controller key.
                If no cls_name is provided, it is found from the instrument
                class name

        """
        # TODO: change acquisition_controller_name to acquisition_controller
        acquisition_controller = self.find_instrument(
            acquisition_controller_name)
        if cls_name is None:
            cls_name = acquisition_controller.__class__.__name__
        # Remove _AcquisitionController from cls_name
        cls_name = cls_name.replace('_AcquisitionController', '')

        self.acquisition_controllers[cls_name] = acquisition_controller

        # Update possible values for (default) acquisition controller
        self.default_acquisition_controller.vals = vals.Enum(
            'None', *self.acquisition_controllers.keys())
        self.acquisition_controller.vals = vals.Enum(
            'None', *self.acquisition_controllers.keys())

    def get_additional_pulses(self, connections) -> list:
        """Additional pulses required for instrument, e.g. trigger pulses.

        Args:
            connections: List of all connections in the layout

        Returns:
            * Empty list if there are no acquisition pulses.
            * A single trigger pulse at start of acquisition if using triggered
              acquisition controller.
            * AcquisitionPulse and TriggerWaitPulse if using the steered
              initialization controller

        Raises:
            NotImplementedError
                Using continous acquisition controller
        """
        if not self.pulse_sequence.get_pulses(acquire=True):
            # No pulses need to be acquired
            return []
        elif self.acquisition_controller() == 'Triggered':
            # Add a single trigger pulse when starting acquisition
            if not self.capture_full_trace():
                t_start = min(pulse.t_start for pulse in
                              self.pulse_sequence.get_pulses(acquire=True))
            else:
                t_start = 0
            return [TriggerPulse(t_start=t_start,
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
        else:
            raise Exception("No controller found")

    def initialize(self):
        """Initializes ATS interface by setting acquisition controller.

        Called at the start of targeting a pulse sequence.
        """
        super().initialize()
        self.acquisition_controller(self.default_acquisition_controller())

    def setup(self,
              samples: Union[int, None] = None,
              connections: list = None,
              **kwargs) -> Union[dict, None]:
        """ Sets up ATS and its controller after targeting a pulse sequence.

        Args:
            samples: Number of acquisition samples.
                If None, it will use the previously set value.
            **kwargs: Unused setup kwargs passed from Layout

        Returns:
            If using ``SteeredInitialization_AcquisitionController``,
            a ``skip_start`` flag is passed with the target instrument, which
            signals to the layout that that instrument should not be started.
            Instead, it is triggered from the steered initialization controller.

        """
        self.configuration_settings({
            k: v for k,v in self.default_settings().items()
            if k in self._configuration_settings_names})
        self.acquisition_settings({
            k: v for k, v in self.default_settings().items()
            if k in self._acquisition_settings_names})

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
        """Configure settings related to triggering of the ATS

        Only configures anything if TriggeredAcquisitionController is used.

        Raises:
            AssertionError
                TriggeredAcquisitionController is used, and no voltage
                transition can be determined.
            """
        # TODO: Correctly handle case where there are no trigger pulses
        if self.acquisition_controller() == 'Triggered':
            if self.trigger_channel() == 'trig_in':
                self.update_settings(external_trigger_range=5)
                trigger_range = 5
            else:
                trigger_range = self.setting('channel_range')
                if isinstance(trigger_range, list):
                    # Different channel ranges, choose one corresponding to
                    # trigger channel
                    trigger_channel = self._acquisition_channels[self.trigger_channel()]
                    trigger_idx = 'ABCD'.index(trigger_channel.id)
                    trigger_range = trigger_range[trigger_idx]

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
                trigger_channel = self._channels[self.trigger_channel()]

                self.update_settings(trigger_operation='J',
                                     trigger_engine1='J',
                                     trigger_source1=trigger_channel.id,
                                     trigger_slope1=self.trigger_slope(),
                                     trigger_level1=trigger_level,
                                     external_trigger_coupling='DC',
                                     trigger_delay=0)
            else:
                print('Cannot setup ATS trigger because there are no acquisition '
                      f'pulses on self.trigger_channel {self.trigger_channel()}')
        else:
            pass

    def setup_ATS(self):
        """ Configure ATS using ``ATS.config`` """
        self.instrument.config(**self.configuration_settings())

    def setup_acquisition_controller(self):
        """ Setup acquisition controller

        Notes:
            - ``Triggered_AcquisitionController``
              The following settings are fixed at the moment, but there could be
              siturations where these are not optimal, e.g. fast measurements.

              - Allocated buffers is maximally 2.
              - Records per buffer is fixed to 1.

            - ``Continuous_AcquisitionController``:

              - Allocated buffers is fixed to 20

            - ``SteeredInitialization_AcquisitionController``:

              - Allocated buffers is fixed to 80

        Raises:
            RuntimeError if acquisition controller is not
                ``Continuous_AcquisitionController``,
                ``SteeredInitialization_AcquisitionController``.
        """
        # Get duration of acquisition. Use flag acquire=True because
        # otherwise initialization Pulses would be taken into account as well
        if not self.capture_full_trace():
            t_start = min(pulse.t_start for pulse in
                          self.pulse_sequence.get_pulses(acquire=True))
            t_stop = max(pulse.t_stop for pulse in
                         self.pulse_sequence.get_pulses(acquire=True))
        else:  # Capture from t = 0 to end of pulse sequence.
            t_start = 0
            t_stop = self.pulse_sequence.duration

        # Subtracting 1e-12 due to machine precision rounding
        # Always subtract, since we later round samples_per_record upwards to
        # the nearest multiple of 16
        acquisition_duration = t_stop - t_start - 1e-12

        samples_per_trace = self.sample_rate() * acquisition_duration
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
            self.acquisition_settings().pop('records_per_buffer', None)
            self.acquisition_settings().pop('buffers_per_acquisition', None)

            # TODO better way to decide on allocated buffers
            allocated_buffers = 20
            self.update_settings(allocated_buffers=allocated_buffers)

            # samples_per_trace must be a multiple of samples_per_record
            samples_per_trace = int(16 * np.ceil(float(samples_per_trace) / 16))
            self._acquisition_controller.samples_per_trace(samples_per_trace)
            self._acquisition_controller.traces_per_acquisition(self.samples())
        elif self.acquisition_controller() == 'SteeredInitialization':
            # records_per_buffer and buffers_per_acquisition are fixed
            self.acquisition_settings().pop('records_per_buffer', None)
            self.acquisition_settings().pop('buffers_per_acquisition', None)

            # Get steered initialization pulse
            initialization = self.pulse_sequence.get_pulse(initialize=True)

            # TODO better way to decide on allocated buffers
            allocated_buffers = 80

            samples_per_buffer = self.sample_rate() * \
                                 initialization.t_buffer
            # samples_per_record must be a multiple of 16
            samples_per_buffer = int(16 * np.ceil(float(samples_per_buffer) / 16))
            self.update_settings(samples_per_record=samples_per_buffer,
                                 allocated_buffers=allocated_buffers)

            # Setup acquisition controller settings through initialization pulse
            initialization.implement()
            for channel in [initialization.trigger_channel,
                            initialization.readout_channel]:
                assert channel.name in self.acquisition_channels(), \
                    f"Channel {channel} must be in acquisition channels"

            # samples_per_trace must be a multiple of samples_per_buffer
            samples_per_trace = int(samples_per_buffer * np.ceil(
                float(samples_per_trace) / samples_per_buffer))
            self._acquisition_controller.samples_per_trace(samples_per_trace)
            self._acquisition_controller.traces_per_acquisition(self.samples())
        else:
            raise RuntimeError(f"No setup programmed for "
                               f"{self.acquisition_controller()}")

        # Set acquisition channels setting
        # Channel_selection must be a sorted string of acquisition channel ids
        channel_ids = ''.join(sorted(
            [self._channels[ch].id for ch in self.acquisition_channels()]))
        if len(channel_ids) == 3:
            # TODO add 'silent' mode
            # logging.warning("ATS cannot be configured with three acquisition "
            #                 "channels {}, setting to ABCD".format(channel_ids))
            channel_ids = 'ABCD'

        buffer_timeout = int(max(20000, 3.1 * self.pulse_sequence.duration * 1e3))
        self.update_settings(channel_selection=channel_ids,
                             buffer_timeout=buffer_timeout)  # ms

        # Update settings in acquisition controller
        self._acquisition_controller.set_acquisition_settings(
            **self.acquisition_settings())
        self._acquisition_controller.average_mode('none')
        self._acquisition_controller.setup()

    def start(self):
        """Ignored method called from `Layout.start`"""
        pass

    def stop(self):
        """ Ignored method called from `Layout.stop`"""
        pass

    def acquisition(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Perform an acquisition.

        Should only be called after the interface has been setup and all other
        instruments have been started (via `Layout.start`).

        Returns:
            Acquisition traces that have been segmented for each pulse.
            Returned dictionary format is:
            ``{pulse.full_name: {channel_id: pulse_channel_trace}}``.

        """
        traces = self._acquisition_controller.acquisition()
        # Convert list of channel traces to a {ch_id: trace} dict
        self.traces = {ch: trace for ch, trace in zip(self.acquisition_channels(), traces)}
        self.pulse_traces = self.segment_traces(self.traces)
        return self.pulse_traces

    def segment_traces(self, traces: Dict[str, np.ndarray]):
        """ Segment traces by acquisition pulses.

        For each pulse with ``acquire`` set to True (which should be all pulses
        passed along to the ATSInterface), the relevant portion of each channel
        trace is segmented and returned in a new dict

        Args:
            traces: ``{channel_id: channel_traces}`` dict

        Returns:
            Dict[str, Dict[str, np.ndarray]:
            Dict format is
            ``{pulse.full_name: {channel_id: pulse_channel_trace}}``.

        """
        pulse_traces = {}

        if self.capture_full_trace():
            t_start_initial = 0
        else:
            t_start_initial = min(p.t_start for p in
                                  self.pulse_sequence.get_pulses(acquire=True))
        for pulse in self.pulse_sequence.get_pulses(acquire=True):
            delta_t_start = pulse.t_start - t_start_initial
            start_idx = int(round(delta_t_start * self.sample_rate()))
            pts = int(round(pulse.duration * self.sample_rate()))

            pulse_traces[pulse.full_name] = {}
            for ch, trace in traces.items():
                pulse_trace = trace[:, start_idx:start_idx + pts]
                if pulse.average == 'point':
                    pulse_traces[pulse.full_name][ch] = np.mean(pulse_trace)
                elif pulse.average == 'trace':
                    pulse_traces[pulse.full_name][ch] = np.mean(pulse_trace, 0)
                elif 'point_segment' in pulse.average:
                    # Extract number of segments to split trace into
                    segments = int(pulse.average.split(':')[1])

                    # average over number of samples, returns 1D trace
                    mean_arr = np.mean(pulse_trace, axis=0)

                    # Split 1D trace into segments
                    segmented_array = np.array_split(mean_arr, segments)
                    pulse_traces[pulse.full_name][ch] = [np.mean(arr) for arr in segmented_array]
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
        """Obtain a setting for the ATS.

        It then checks if it is a configuration or acquisition setting.
        If the setting is specified in self.configuration/acquisition_setting,
        it returns that value, else it returns the value set in the ATS

        Args:
            setting: configuration or acquisition setting to look for.

        Returns:
            Value of the setting

        Raises:
            AssertionError
                Setting is not an ATS configuration or acquisition setting.
        """
        assert setting in self._settings_names, \
            f"Kwarg {setting} is not a valid ATS acquisition setting"
        if setting in self.configuration_settings():
            return self.configuration_settings()[setting]
        elif setting in self.acquisition_settings():
            return self.acquisition_settings()[setting]
        else:
            # Must get latest value, since it may not be updated in ATS
            return self.instrument.parameters[setting]()

    def set_configuration_settings(self, **settings):
        """ Sets the configuration settings for the ATS through its controller.

        All existing configuration settings are cleared.
        The controller's configuration settings are not actually updated here,
        but will be done when calling `ATSInterface.setup`.

        Args:
            **settings: ATS configuration settings to be set

        Raises:
            AssertionError
                Setting is not an ATS configuration setting
        """
        assert all([setting in self._configuration_settings_names
                    for setting in settings]), \
            "Settings are not all valid ATS configuration settings"
        self._configuration_settings = settings

    def set_acquisition_settings(self, **settings):
        """ Sets the acquisition settings for the ATS through its controller.

        All existing acquisition settings are cleared.
        The controller's acquisition settings are not actually updated here,
        but will be done when calling `ATSInterface.setup`.

        Args:
            **settings: ATS acquisition settings to be set

        Raises:
            AssertionError
                Setting is not an ATS acquisition setting
        """
        assert all([setting in self._acquisition_settings_names
                    for setting in settings]), \
            "Settings are not all valid ATS acquisition settings"
        self._acquisition_settings = settings

    def update_settings(self, **settings):
        """ Update configuration and acquisition settings

        The acquisition controller's settings are not actually updated here,
        this will be done when calling `ATSInterface.setup`.

        Args:
            **settings: ATS configuration and acquisition settings to be set.

        Raises:
            AssertionError
                Some settings are not configuration nor acquisition settings.
        """
        settings_valid = all(map(
            lambda setting: setting in self._settings_names, settings.keys()))
        assert settings_valid, \
            f'Not all settings are valid ATS settings. Settings: {settings}\n' \
            f'Valid ATS settings: {self._settings_names}'

        configuration_settings = {k: v for k, v in settings.items()
                                  if k in self._configuration_settings_names}
        self.configuration_settings().update(**configuration_settings)

        acquisition_settings = {k: v for k, v in settings.items()
                                  if k in self._acquisition_settings_names}
        self.acquisition_settings().update(**acquisition_settings)


class SteeredInitializationImplementation(PulseImplementation):
    pulse_class = SteeredInitialization

    def target_pulse(self,
                     pulse: Pulse,
                     interface,
                     connections: list,
                     **kwargs) -> Pulse:
        """ Target steered initialization pulse to an interface.

        The implementation will further have a ``readout_connection`` and
        ``trigger_connection``.

        Args:
            pulse: Steered initialization pulse to be targeted.
            interface: Interface to target pulse to.
            connections: List of output connections
            **kwargs:

        Returns:
            targeted pulse

        Raises:
            AssertionError
                Not exactly one readout connection found
            AssertionError
                Not exactly one trigger connection found
        """
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

    def implement(self, interface: InstrumentInterface):
        """ Implements pulse """
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
