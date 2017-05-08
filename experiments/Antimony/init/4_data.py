#####################
### Data handling ###
#####################
qc.data.data_set.DataSet.default_io.base_location = config.user.properties.data_folder
loc_provider = qc.data.location.FormatLocation(fmt='{date}/#{counter}_{name}_{time}')
qc.data.data_set.DataSet.location_provider=loc_provider