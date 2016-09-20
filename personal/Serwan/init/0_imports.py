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

qc.show_subprocess_widget()

# Data handling
qc.data.data_set.DataSet.default_io.base_location = r"C:\Users\serwa_000\Documents\data"
loc_provider = qc.data.location.FormatLocation(fmt='{date}/#{counter}_{name}_{time}')
qc.data.data_set.DataSet.location_provider=loc_provider