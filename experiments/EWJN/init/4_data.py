from qcodes.data import hdf5_format
from qcodes.data.data_set import DataSet, new_data, DataMode
from qcodes.data.manager import DataManager, DataServer
from qcodes.data.data_array import DataArray
h5fmt = hdf5_format.HDF5Format()

# Data handling
qc.data.data_set.DataSet.default_io.base_location = 'E:\EWJN\data'
loc_provider = qc.data.location.FormatLocation(fmt='{date}/#{counter}_{name}_{time}')
qc.data.data_set.DataSet.location_provider=loc_provider


# Add an extra data manager for trace saving
data_manager = qc.data.manager.get_data_manager()
data_manager_raw = DataManager(server_name='Raw_DataServer')
data_manager_raw.base_location = DataSet.default_io.base_location
qc.data.manager.DataManager.default = data_manager