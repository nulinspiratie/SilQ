import os
import numpy as np

import qcodes as qc
from qcodes import config
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.data.data_set import new_data, DataSet
from qcodes.data.data_array import DataArray

def get_latest_data_folder():
    if 'data_folder' in config['user']:
        base_location = config['user']['data_folder']
    else:
        base_location = DataSet.default_io.base_location
    date = os.listdir(base_location)[-1]
    dir_date = os.path.join(base_location, date)
    folders_date = os.listdir(dir_date)
    latest_folder = folders_date[-1]
    data_folder = os.path.join(dir_date, latest_folder)
    return data_folder


def create_data_set(name, base_folder, subfolder=None, formatter=None):
    location_string = '{base_folder}/'
    if subfolder is not None:
        location_string += '{subfolder}/'
    location_string += '#{{counter}}_{name}'

    location = qc.data.location.FormatLocation(
        fmt=location_string.format(base_folder=base_folder, name=name,
                                   subfolder=subfolder))

    data_set = new_data(location=location,
                        name=name,
                        formatter=formatter)
    return data_set

def store_data(dataset, result):
    dataset.store(loop_indices=slice(0, result.shape[0], 1),
                  ids_values={'data_vals': result})