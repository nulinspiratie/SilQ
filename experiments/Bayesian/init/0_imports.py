###############
### Imports ###
###############


###############
### General ###
###############
import os
import sys
import win32ui
import numpy as np
from numpy import array, nan
from functools import partial
from importlib import reload
from time import sleep, time
from winsound import Beep
from matplotlib import rcParams, pyplot as plt
import pyperclip
import json
import threading
from PyQt5.QtWidgets import QApplication

np.set_printoptions(precision=3)

from IPython.core.magic import (register_line_magic, register_cell_magic,
                                register_line_cell_magic, needs_local_scope)

rcParams['figure.max_open_warning'] = 80
rcParams['figure.max_open_warning'] = 80
plt.ion()

##############
### Qcodes ###
##############
import qcodes as qc
from qcodes.utils.helpers import in_notebook
from qcodes import Instrument, Loop, Task, load_data,combine
from qcodes.utils.helpers import in_notebook
from qcodes.instrument.parameter import Parameter, ManualParameter, \
    StandardParameter
if in_notebook():
    from qcodes import MatPlot
# from qcodes.widgets.slack import Slack

from qcodes.data.hdf5_format import HDF5Format as h5fmt
from qcodes.data.data_set import DataSet
from qcodes.data.data_array import DataArray

station = qc.Station()

############
### SilQ ###
############
from silq import parameters, config
from silq.instrument_interfaces import get_instrument_interface
from silq.tools.general_tools import partial_from_attr, print_attr, run_code
from silq.tools.plot_tools import InteractivePlot, CalibrationPlot, DCPlot, \
    DCSweepPlot
from silq.tools.parameter_tools import create_set_vals
from silq.tools.notebook_tools import create_cell
from silq.pulses import *

bayesian = silq.config.berdina

# Dictionary of code with labels, these can be registered via cell magic
# %label {lbl}. They can then be run via for instance Slack
code_labels = {}
silq.tools.general_tools.code_labels = code_labels