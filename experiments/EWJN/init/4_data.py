h5fmt = hdf5_format.HDF5Format()

# Data handling
qc.data.data_set.DataSet.default_io.base_location = \
    config['user']['data_folder']
loc_provider = qc.data.location.FormatLocation(fmt='{date}/#{counter}_{name}_{time}')
qc.data.data_set.DataSet.location_provider=loc_provider