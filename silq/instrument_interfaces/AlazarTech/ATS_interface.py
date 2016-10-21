import numpy as np
import inspect

from silq.instrument_interfaces import InstrumentInterface

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from qcodes.instrument_drivers.AlazarTech.ATS import AcquisitionController


class ATSInterface(InstrumentInterface):
    def __init__(self, name, instrument_name, acquisition_controller_names,
                 **kwargs):
        InstrumentInterface.__init__(name, instrument_name, **kwargs)

        # Create a dictionary of the acquisition controller classes along with
        # the acquisition controllers
        self.acquisition_controllers = {}
        for acquisition_controller_name in acquisition_controller_names:
            acquisition_controller = self.find_instrument(
                acquisition_controller_name)
            cls = acquisition_controller._instrument_class.__name__
            self.acquisition_controllers[cls] = acquisition_controller

        self.acquisition_settings = {}
        # Obtain a list of all valid ATS acquisition settings
        self._acquisition_settings_names = \
            inspect.signature(self.instrument.acquire).parameters.keys()

        # Names and shapes must have initial value, even through they will be
        # overwritten in set_acquisition_settings. If we don't do this, the
        # remoteInstrument will not recognize that it returns multiple values.
        self.add_parameter(name="acquisition",
                           names=['channel_signal'],
                           get_cmd=self._acquisition,
                           shapes=((),),
                           snapshot_value=False)

        self.add_parameter(name='active_acquisition_controller',
                           vals=vals.Enum('auto',
                                          *self.acquisition_controllers.keys()))

    def setup(self):
        # TODO change when necessary
        # acquisition_controller
        pass

    def _acquisition(self):
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def acquisition_setting(self, setting):
        """
        Obtain an acquisition setting for the ATS.
        It first checks if the setting is an actual ATS acquisition kwarg, and
        raises an error otherwise.
        It then checks if the setting is in ATS_controller._acquisition_settings
        If not, it will retrieve the ATS latest parameter value

        Args:
            setting: acquisition kwarg to look for

        Returns:
            Value of the acquisition setting
        """
        assert setting in self._acquisition_settings_names, \
            "Kwarg {} is not a valid ATS acquisition setting".format(setting)
        if setting in self.settings.keys():
            return self.settings[setting]
        else:
            # Must get latest value, since it may not be updated in ATS
            return self.instrument.parameters[setting].get_latest()

    def update_acquisition_settings(self, **settings):
        settings_valid = all(map(
            lambda setting: setting in self._acquisition_settings_names,
            settings.keys()))
        assert settings_valid, \
            'Not all settings are valid ATS acquisition settings'
        self.acquisition_settings.update(**settings)