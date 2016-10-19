import numpy as np
import inspect

from silq.instrument_interfaces import InstrumentInterface

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from qcodes.instrument_drivers.AlazarTech.ATS import AcquisitionController


class ATSInterface(InstrumentInterface, AcquisitionController):

    def __init__(self, name, instrument_name, **kwargs):
        InstrumentInterface.__init__(name, instrument_name, **kwargs)

        self.acquisition_settings = {}
        # Obtain a list of all valid ATS acquisition kwargs
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
            Value of the acquisition kwarg
        """
        assert setting in self._settings_names, \
            "Kwarg {} is not a valid ATS acquisition setting".format(setting)
        if setting in self.settings.keys():
            return self.settings[setting]
        else:
            # Must get latest value, since it may not be updated in ATS
            return self.instrument.parameters[setting].get_latest()

    def update_acquisition_settings(self, **settings):
        self.acquisition_settings.update(**settings)