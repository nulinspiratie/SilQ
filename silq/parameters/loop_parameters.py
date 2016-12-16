from time import sleep
import numpy as np
from collections import OrderedDict
import logging

import qcodes as qc
from qcodes import config
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.data import hdf5_format, io
h5fmt = hdf5_format.HDF5Format()

from silq.analysis import analysis
from silq.tools import data_tools, general_tools

properties_config = config['user'].get('properties', {})


class Loop_Parameter(Parameter):
    def __init__(self, name, measure_parameter, **kwargs):
        super().__init__(name, **kwargs)
        self.measure_parameter = measure_parameter

        self.loc_provider = qc.data.location.FormatLocation(
            fmt='#{counter}_{name}_{time}')
        self._meta_attrs.extend(['measure_parameter_name'])

    @property
    def measure_parameter_name(self):
        return self.measure_parameter.name

    @property
    def disk_io(self):
        return io.DiskIO(data_tools.get_latest_data_folder())


class Loop0D_Parameter(Loop_Parameter):
    def __init__(self, name, measure_parameter, **kwargs):
        super().__init__(name, measure_parameter=measure_parameter, **kwargs)

    def get(self):

        self.measurement = qc.Measure(self.measure_parameter)
        self.data = self.measurement.run(
            name='{}_{}'.format(self.name, self.measure_parameter_name),
            data_manager=False,
            io=self.disk_io, location=self.loc_provider)
        return self.data

class Loop1D_Parameter(Loop_Parameter):
    def __init__(self, name, set_parameter, measure_parameter, set_vals=None,
                 **kwargs):
        super().__init__(name, measure_parameter=measure_parameter, **kwargs)
        self.set_parameter = set_parameter
        self.set_vals = set_vals

        self._meta_attrs.extend(['set_parameter_name', 'set_vals'])

    @property
    def set_parameter_name(self):
        return self.set_parameter.name

    def get(self):
        # Set data saving parameters
        self.measurement = qc.Loop(self.set_parameter[self.set_vals]
                                  ).each(self.measure_parameter)
        self.data = self.measurement.run(
            name='{}_{}_{}'.format(self.name, self.set_parameter_name,
                                   self.measure_parameter_name),
            background=False, data_manager=False,
            io=self.disk_io, location=self.loc_provider)
        return self.data

    def set(self, val):
        self.set_vals = val
