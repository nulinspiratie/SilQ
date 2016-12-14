# Imports
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


# Qcodes imports
import qcodes as qc
from qcodes import Instrument, config
from qcodes.instrument.parameter import Parameter, ManualParameter, StandardParameter


# Qcodes config
if 'pulses' not in config['user']:
    config.add('pulses', {})
pulse_config = config['user']['pulses']

if 'properties' not in config['user']:
    config.add('properties', {})
properties_config = config['user']['properties']


# SilQ imports
from silq.parameters import measurement_parameters, general_parameters
from silq.instrument_interfaces import get_instrument_interface
from silq.pulses import *
from silq.functions.general_functions import printattr

if not 'USE_MP' in globals():
    USE_MP = True