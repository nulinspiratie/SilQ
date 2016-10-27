from functools import partial
import inspect

from . import MockInstrument

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals



class MockATS(MockInstrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        channels = ['A', 'B', 'C', 'D']

        # Obtain a list of all valid ATS configuration settings
        self._configuration_settings_names = list(
            inspect.signature(self.config).parameters.keys())
        # Obtain a list of all valid ATS acquisition settings
        self._acquisition_settings_names = list(
            inspect.signature(self.acquire).parameters.keys())
        self._settings_names = self._acquisition_settings_names + \
            self._configuration_settings_names

        self._configuration_settings = {}
        self._acquisition_settings = {}
        self.add_parameter(name='configuration_settings',
                           get_cmd=lambda: self._configuration_settings)
        self.add_parameter(name='acquisition_settings',
                           get_cmd=lambda: self._acquisition_settings)

        for param in ['clock_source', 'sample_rate', 'clock_edge', 'decimation',
                      'trigger_operation', 'external_trigger_coupling',
                      'external_trigger_range', 'trigger_delay',
                      'timeout_ticks', 'mode', 'sampled_per_record',
                      'records_per_buffer', 'bufers_per_acquisition',
                      'channel_selection', 'transfer_offset',
                      'external_startcapture', 'enable_record_headers',
                      'alloc_buffers', 'fifo_only_streaming',
                      'interleave_samples', 'get_processed_data',
                      'allocated_buffers', 'buffer_timeout']:
            self.add_parameter(name=param,
                               get_cmd=partial(self.get_setting, param),
                               # set_cmd=partial(self.set_setting, param),
                               vals=vals.Anything())

        for idx in ['1', '2']:
            for param in ['trigger_engine', 'trigger_source',
                          'trigger_slope', 'trigger_level']:
                self.add_parameter(name=param+idx,
                                   get_cmd=partial(self.get_setting, param+idx),
                                   # set_cmd=partial(self.set_setting, param+idx),
                                   vals=vals.Anything())


        for idx, ch in enumerate(channels):
            for param in ['coupling', 'channel_range', 'impedance']:
                self.add_parameter(name=param+ch,
                                   get_cmd=partial(self.get_setting, param,
                                                   idx=idx),
                                   # set_cmd=partial(self.set_setting, param),
                                   vals=vals.Anything())


    def config(self, clock_source=None, sample_rate=None, clock_edge=None,
               decimation=None, coupling=None, channel_range=None,
               impedance=None, bwlimit=None, trigger_operation=None,
               trigger_engine1=None, trigger_source1=None,
               trigger_slope1=None, trigger_level1=None,
               trigger_engine2=None, trigger_source2=None,
               trigger_slope2=None, trigger_level2=None,
               external_trigger_coupling=None, external_trigger_range=None,
               trigger_delay=None, timeout_ticks=None):
        for setting in self._configuration_settings_names:
            if locals()[setting] is not None:
                self._configuration_settings[setting] = locals()[setting]

        self.print_function(function='config', **self._configuration_settings)

    def acquire(self, mode=None, samples_per_record=None,
                records_per_buffer=None, buffers_per_acquisition=None,
                channel_selection=None, transfer_offset=None,
                external_startcapture=None, enable_record_headers=None,
                alloc_buffers=None, fifo_only_streaming=None,
                interleave_samples=None, get_processed_data=None,
                allocated_buffers=None, buffer_timeout=None,
                acquisition_controller=None):
        for setting in self._acquisition_settings_names:
            if locals()[setting] is not None:
                self._acquisition_settings[setting] = locals()[setting]
        self.print_function(function='config',
                            **self._acquisition_settings)

    def get_setting(self, setting, idx=None):
        if setting in self._acquisition_settings_names:
            val = self._acquisition_settings[setting]
        else:
            val = self._configuration_settings[setting]

        if idx is not None and isinstance(val, list):
            val = val[idx]
        return val

    def set_setting(self, setting, val):
        if setting in self._acquisition_settings_names:
            self._acquisition_settings[setting] = val
        else:
            self._configuration_settings[setting] = val
