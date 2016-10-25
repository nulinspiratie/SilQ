from functools import partial
import inspect

from . import MockInstrument

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals



class MockATS(MockInstrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # Obtain a list of all valid ATS configuration settings
        self._configuration_settings_names = list(
            inspect.signature(self.config).parameters.keys())
        # Obtain a list of all valid ATS acquisition settings
        self._acquisition_settings_names = list(
            inspect.signature(self.acquire).parameters.keys())
        self._settings_names = self._acquisition_settings_names + \
            self._configuration_settings_names


    def config(self, clock_source=None, sample_rate=None, clock_edge=None,
               decimation=None, coupling=None, channel_range=None,
               impedance=None, bwlimit=None, trigger_operation=None,
               trigger_engine1=None, trigger_source1=None,
               trigger_slope1=None, trigger_level1=None,
               trigger_engine2=None, trigger_source2=None,
               trigger_slope2=None, trigger_level2=None,
               external_trigger_coupling=None, external_trigger_range=None,
               trigger_delay=None, timeout_ticks=None):
        settings = {}
        for setting in self._configuration_settings_names:
            if locals()[setting] is not None:
                settings[setting] = locals()[setting]
        self.print_function(function='config', **settings)

    def acquire(self, mode=None, samples_per_record=None,
                records_per_buffer=None, buffers_per_acquisition=None,
                channel_selection=None, transfer_offset=None,
                external_startcapture=None, enable_record_headers=None,
                alloc_buffers=None, fifo_only_streaming=None,
                interleave_samples=None, get_processed_data=None,
                allocated_buffers=None, buffer_timeout=None,
                acquisition_controller=None):
        settings = {}
        for setting in self._acquisition_settings_names:
            if locals()[setting] is not None:
                settings[setting] = locals()[setting]
        self.print_function(function='config', **settings)