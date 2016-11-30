

if __name__ == "__main__":
    USE_MP = True
    import silq

    silq.initialize("EWJN", mode='analysis')
    qc.config['core']['legacy_mp'] = True

    # Data handling
    qc.data.data_set.DataSet.default_io.base_location = r'C:\Users\serwa_000\Documents\data'
    loc_provider = qc.data.location.FormatLocation(
        fmt='{date}/#{counter}_{name}_{time}')
    qc.data.data_set.DataSet.location_provider = loc_provider

    station = qc.station.Station()
    from qcodes.tests.instrument_mocks import MockParabola

    mock_parabola = MockParabola(name='MockParabola', server_name='')

    measurement_parameters.MeasurementParameter.data_manager = data_manager_raw

    measurement_param = measurement_parameters.TestMeasurementParameter()
    # measurement_param.data_manager = data_manager_raw

    loop = qc.Loop(mock_parabola.x[-100:100:20]).each(
        measurement_param,
        mock_parabola.skewed_parabola)
    data = loop.run(name='MockParabola_run', formatter=h5fmt, background=False)