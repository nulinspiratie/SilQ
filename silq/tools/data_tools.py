import os
import numpy as np

import qcodes as qc
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.data.data_set import new_data, DataMode, DataSet
from qcodes.data.data_array import DataArray

def get_latest_data_folder():
    base_location = DataSet.default_io.base_location
    date = os.listdir(base_location)[-1]
    dir_date = os.path.join(base_location, date)
    folders_date = os.listdir(dir_date)
    latest_folder = folders_date[-1]
    data_folder = os.path.join(dir_date, latest_folder)
    return data_folder


def create_raw_data_set(name, data_manager, shape,
                        folder_name=None, location=None, formatter=None):
    if folder_name is None:
        folder_name = name
    data_array_set = DataArray(name='set_vals',
                               shape=(shape[0],),
                               preset_data=np.arange(shape[0]),
                               is_setpoint=True)
    index0 = DataArray(name='index0', shape=shape,
                       preset_data=np.full(shape,
                                           np.arange(shape[-1]),
                                           dtype=np.int))
    data_array_values = DataArray(name='data_vals',
                                  shape=shape,
                                  set_arrays=(
                                      data_array_set, index0))

    data_mode = DataMode.PUSH_TO_SERVER

    if hasattr(data_manager, 'base_location'):
        DataSet.default_io.base_location = data_manager.base_location

    if location is None:
        data_folder = get_latest_data_folder()
        location = qc.data.location.FormatLocation(
            fmt='{data_folder}/{name}/#{{counter}}'.format(
                data_folder=data_folder, name=folder_name))

    data_set = new_data(
        location=location,
        arrays=[data_array_set, index0, data_array_values],
        mode=data_mode,
        data_manager=data_manager, name=name,
        formatter=formatter)
    return data_set

def store_data(data_manager, result):
    loop_indices = slice(0, result.shape[0], 1)
    ids_values = {'data_vals': result}
    data_manager.write('store_data', loop_indices, ids_values)
    data_manager.write('finalize_data')