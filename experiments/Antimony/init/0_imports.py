###############
### Imports ###
###############


###############
### General ###
###############
import os
import sys
import numpy as np
from numpy import array, nan
from functools import partial
from importlib import reload
from time import sleep, time
from winsound import Beep
from matplotlib import pyplot as plt
import pyperclip

np.set_printoptions(precision=3)

from IPython.core.magic import (register_line_magic, register_cell_magic,
                                register_line_cell_magic, needs_local_scope)


##############
### Qcodes ###
##############
import qcodes as qc
from qcodes import Instrument, config, Loop, Task, load_data, MatPlot, combine
from qcodes.utils.helpers import in_notebook
from qcodes.instrument.parameter import Parameter, ManualParameter, \
    StandardParameter
# from qcodes.widgets.slack import Slack

from qcodes.data.hdf5_format import HDF5Format as h5fmt
from qcodes.data.data_set import DataSet
from qcodes.data.data_array import DataArray


############
### SilQ ###
############
from silq.parameters import measurement_parameters, general_parameters, acquisition_parameters
from silq.instrument_interfaces import get_instrument_interface
from silq.tools.general_tools import partial_from_attr, print_attr, run_code
from silq.tools.parameter_tools import create_set_vals
from silq.tools.notebook_tools import create_cell
from silq.pulses import *

# Dictionary of code with labels, these can be registered via cell magic
# %label {lbl}. They can then be run via for instance Slack
code_labels = {}
silq.tools.general_tools.code_labels = code_labels