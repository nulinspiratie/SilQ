
# Imports
import os
import clr
import sys
from imp import reload
from System import Array
from time import sleep, time
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import factorial
import peakutils
# sys.path.append(os.getcwd())

import qcodes as qc

import qcodes.instrument.parameter as parameter

loc_provider = qc.data.location.FormatLocation(fmt='data/{date}/#{counter}_{name}_{time}')
qc.data.data_set.DataSet.location_provider=loc_provider

import qcodes.instrument_drivers.lecroy.ArbStudio1104 as ArbStudio_driver
import qcodes.instrument_drivers.spincore.PulseBlasterESRPRO as PulseBlaster_driver
import qcodes.instrument_drivers.stanford_research.SIM900 as SIM900_driver
import qcodes.instrument_drivers.AlazarTech.ATS9440 as ATS_driver
import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ATS_controller_driver



if __name__ == "__main__":
    ATS = ATS_driver.ATS9440('ATS', server_name='Alazar_server')