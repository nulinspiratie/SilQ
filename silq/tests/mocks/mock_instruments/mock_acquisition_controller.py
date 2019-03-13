import numpy as np
from functools import partial

from . import MockInstrument

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


class MockAcquisitionController(MockInstrument):
    def __init__(self, name, alazar_name, **kwargs):
        super().__init__(name, **kwargs)

        self._alazar = self.find_instrument(alazar_name)

        self._acquisition_settings = {}

        functions = ['config', 'acquire']
        for function in functions:
            self.add_function(function,
                              call_cmd=partial(self.print_function,
                                               function=function),
                              args=[vals.Anything()])

        self.add_parameter('average_mode',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Anything())

        self.add_parameter(name='acquisition_settings',
                           get_cmd=lambda: self._acquisition_settings)

        self.add_parameter(name='acquisition',
                           names=['channel_signal'],
                           shapes=((),),
                           get_cmd=self._acquisition)

    def setup(self):
        for attr in ['channel_selection', 'samples_per_record',
                     'records_per_buffer', 'buffers_per_acquisition']:
            setattr(self, attr, self.get_acquisition_setting(attr))
        self.acquisition.names = tuple(['ch{}_signal'.format(ch) for ch in
                                        self.channel_selection])
        self.number_of_channels = len(self.channel_selection)

        self.acquisition.labels = self.acquisition.names
        self.acquisition.units = ['V'] * self.number_of_channels

        if self.average_mode() == 'point':
            self.acquisition.shapes = tuple([()] * self.number_of_channels)
        elif self.average_mode() == 'trace':
            shape = (self.samples_per_record,)
            self.acquisition.shapes = tuple([shape] * self.number_of_channels)
        else:
            shape = (self.records_per_buffer * self.buffers_per_acquisition,
                     self.samples_per_record)
            self.acquisition.shapes = tuple([shape] * self.number_of_channels)

    def _acquisition(self):
        if self.average_mode() == 'point':
            return [k for k in range(self.number_of_channels)]
        elif self.average_mode() == 'trace':
            return [k * np.ones(self.samples_per_record)
                    for k in range(self.number_of_channels)]
        else:
            return [k * np.ones((self.records_per_buffer *
                                 self.buffers_per_acquisition,
                                 self.samples_per_record))
                    for k in range(self.number_of_channels)]

    def _get_alazar(self):
        """
        returns a reference to the alazar instrument. A call to self._alazar is
        quicker, so use that if in need for speed
        :return: reference to the Alazar instrument
        """
        return self._alazar

    def get_acquisition_setting(self, setting):
        """
        Obtain an acquisition setting for the ATS.
        It checks if the setting is in ATS_controller._acquisition_settings
        If not, it will retrieve the ATS latest parameter value

        Args:
            setting: acquisition setting to look for

        Returns:
            Value of the acquisition setting
        """
        if setting in self._acquisition_settings.keys():
            return self._acquisition_settings[setting]
        else:
            # Must get latest value, since it may not be updated in ATS
            return self._alazar.parameters[setting].get_latest()

    def set_acquisition_settings(self, **settings):
        self._acquisition_settings = settings

    def update_acquisition_settings(self, **kwargs):
        self._acquisition_settings.update(**kwargs)
