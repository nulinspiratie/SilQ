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
import time
from time import sleep
from winsound import Beep
import logging
from matplotlib import pyplot as plt
import pyperclip
import json
import threading
from PyQt5.QtWidgets import QApplication

from IPython.core.magic import (register_line_magic, register_cell_magic,
                                register_line_cell_magic, needs_local_scope)


##############
### Qcodes ###
##############
import qcodes as qc
from qcodes.utils.helpers import in_notebook
from qcodes import Instrument, Loop, Measure, Task, load_data,combine
from qcodes.utils.helpers import in_notebook
from qcodes.instrument.parameter import Parameter, ManualParameter, \
    StandardParameter
if in_notebook():
    from qcodes import MatPlot
from qcodes.widgets.slack import Slack

from qcodes.data import hdf5_format
from qcodes.data.data_set import DataSet
from qcodes.data.data_array import DataArray


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
from silq.gui.SIMGui import sim_gui


#############
### setup ###
#############

# Matplotlib
plt.rcParams['figure.max_open_warning'] = 80
plt.rcParams['figure.max_open_warning'] = 80
plt.rcParams['image.cmap'] = 'inferno'
plt.ion()

np.set_printoptions(precision=3)

station = qc.Station()

antimony = silq.config.antimony

# Dictionary of code with labels, these can be registered via cell magic
# %label {lbl}. They can then be run via for instance Slack
code_labels = {}
silq.tools.general_tools.code_labels = code_labels

h5fmt = hdf5_format.HDF5Format()