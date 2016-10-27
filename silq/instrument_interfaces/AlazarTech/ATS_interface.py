import numpy as np
import inspect

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from qcodes.instrument_drivers.AlazarTech.ATS import AlazarTech_ATS

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.tools import get_instrument_class
from silq.pulses import MeasurementPulse, TriggerPulse, PulseImplementation


class ATSInterface(InstrumentInterface):
    def __init__(self, instrument_name, acquisition_controller_names=[],
                 **kwargs):
        super().__init__(instrument_name, **kwargs)

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

        self.configuration_settings = {}
        self.acquisition_settings = {}

        # Obtain a list of all valid ATS configuration settings
        self._configuration_settings_names = sorted(list(
            inspect.signature(AlazarTech_ATS.config).parameters.keys()))
        # Obtain a list of all valid ATS acquisition settings
        self._acquisition_settings_names = sorted(list(
            inspect.signature(AlazarTech_ATS.acquire).parameters.keys()))
        self._settings_names = sorted(self._acquisition_settings_names +
                                      self._configuration_settings_names)

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

        self.add_parameter(name='acquisition_channels',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Anything())

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
        self.acquisition_controllers[cls_name] = acquisition_controller

    def get_final_additional_pulses(self):
        if not self._pulse_sequence:
            return []
        else:
            t_start_min = min(pulse.t_start for pulse in self._pulse_sequence)
            acquisition_trigger_pulse = \
                TriggerPulse(t_start=t_start_min,
                             connection_requirements={
                                 'input_instrument': self.instrument_name(),
                                 'trigger': True},
                             acquire=True
                             )
        return [acquisition_trigger_pulse]

    def setup(self):
        self.configuration_settings.clear()
        self.acquisition_settings.clear()

        if self.acquisition_mode() == 'trigger':
            self.acquisition_controller = self.acquisition_controllers[
                'Basic_AcquisitionController']
        else:
            raise Exception('Acquisition mode {} not implemented'.format(
                self.acquisition_mode()))

        # Set acquisition channels setting
        # Channel_selection must be a sorted string of acquisition channel ids
        channel_ids = ''.join(sorted(
            [self._channels[ch].id for ch in self.acquisition_channels()]))
        self.update_settings(channel_selection=channel_ids)

        # TODO Check/target pulse sequence and

        self.setup_trigger()
        self.setup_ATS()
        self.setup_acquisition_controller()

    def setup_trigger(self):
        if self.acquisition_mode() == 'trigger':
            if self.trigger_channel() == 'trig_in':
                self.update_settings(external_trigger_range=5)
                trigger_range = 5
            else:
                trigger_range = self.setting('channel_range' +
                                             self.trigger_channel())

            acquisition_pulses = self._pulse_sequence.get_pulses(
                acquire=True, input_channel=self.trigger_channel())
            # print('All pulses: {}'.format(self._pulse_sequence))
            # print('acquire pulses: {}'.format(self._pulse_sequence.get_pulses(
            #     acquire=True)))
            # print('channel pulses: {}'.format(self._pulse_sequence.get_pulses(
            #     input_channel=self.trigger_channel())))
            # print('ATS acquisition pulses: {}'.format(acquisition_pulses))
            if acquisition_pulses:
                start_pulse = min(acquisition_pulses, key=lambda p: p.t_start)
                pre_voltage, post_voltage = \
                    self._pulse_sequence.get_transition_voltages(
                        pulse=start_pulse)
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
        self.instrument.config(**self.configuration_settings)

    def setup_acquisition_controller(self):
        self.acquisition_controller.set_acquisition_settings(
            **self.acquisition_settings)
        self.acquisition_controller.average_mode(self.average_mode())
        self.acquisition_controller.setup()

    def start(self):
        pass

    def stop(self):
        pass

    def _acquisition(self):
        raise NotImplementedError(
            'This method should be implemented in a subclass')

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
        if setting in self.configuration_settings.keys():
            return self.configuration_settings[setting]
        elif self.acquisition_settings.keys():
            return self.acquisition_settings[setting]
        else:
            # Must get latest value, since it may not be updated in ATS
            return self.instrument.parameters[setting].get_latest()

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
        self.configuration_settings = settings

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
        self.acquisition_settings = settings

    def update_settings(self, **settings):
        settings_valid = all(map(
            lambda setting: setting in self._settings_names, settings.keys()))
        assert settings_valid, \
            'Not all settings are valid ATS settings. Settings: {}\n' \
            'Valid ATS settings: {}'.format(settings, self._settings_names)

        configuration_settings = {k: v for k, v in settings.items()
                                  if k in self._configuration_settings_names}
        self.configuration_settings.update(**configuration_settings)

        acquisition_settings = {k: v for k, v in settings.items()
                                  if k in self._acquisition_settings_names}
        self.acquisition_settings.update(**acquisition_settings)
