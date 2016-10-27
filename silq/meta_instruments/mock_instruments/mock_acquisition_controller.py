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
                           get_cmd=lambda: self._acquisition_settings)

    def setup(self):
        channel_selection = self.get_acquisition_setting('channel_selection')
        samples_per_record = self.get_acquisition_setting('samples_per_record')
        records_per_buffer = self.get_acquisition_setting('records_per_buffer')
        buffers_per_acquisition = self.get_acquisition_setting('buffers_per_acquisition')
        self.acquisition.names = tuple(['Channel_{}_signal'.format(ch) for ch in
                                        self.get_acquisition_setting('channel_selection')])

        self.acquisition.labels = self.acquisition.names
        self.acquisition.units = ['V'*len(channel_selection)]

        if self.average_mode() == 'point':
            self.acquisition.shapes = tuple([()]*len(channel_selection))
        elif self.average_mode() == 'trace':
            shape = (samples_per_record,)
            self.acquisition.shapes = tuple([shape] * len(channel_selection))
        else:
            shape = (records_per_buffer * buffers_per_acquisition, samples_per_record)
            self.acquisition.shapes = tuple([shape] * len(channel_selection))

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
            return self.alazar.parameters[setting].get_latest()

    def set_acquisition_settings(self, **settings):
        self._acquisition_settings = settings

    def update_acquisition_settings(self, **kwargs):
        self._acquisition_settings.update(**kwargs)

