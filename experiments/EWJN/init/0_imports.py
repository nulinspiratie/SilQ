# Imports
import os
import sys
import numpy as np
from functools import partial
from importlib import reload
from time import sleep, time
from matplotlib import pyplot as plt

# SilQ imports
from silq.analysis import analysis
from silq.parameters import measurement_parameters, general_parameters

# Qcodes imports
import qcodes as qc
from qcodes import Instrument
from qcodes.instrument.parameter import Parameter, ManualParameter, StandardParameter
from qcodes.data import hdf5_format
from qcodes.data.data_set import DataSet
h5fmt = hdf5_format.HDF5Format()

from qcodes.data.manager import DataManager, DataServer
from qcodes.data.data_set import new_data, DataMode
from qcodes.data.data_array import DataArray


# Data handling
qc.data.data_set.DataSet.default_io.base_location = 'E:\EWJN\data'
loc_provider = qc.data.location.FormatLocation(fmt='{date}/#{counter}_{name}_{time}')
qc.data.data_set.DataSet.location_provider=loc_provider


from silq.tools.general_tools import execfile
SilQ_folder = silq.get_SilQ_folder()
base_folder = os.path.join(SilQ_folder, r'experiments\EWJN\init')

# Load instruments
execfile(os.path.join(base_folder, '1_instruments.py'))