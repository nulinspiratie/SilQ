
if __name__ == "__main__":
    import silq
    silq.initialize("EWJN")

    from qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers import \
        Continuous_AcquisitionController

    continuous_controller = Continuous_AcquisitionController(
        name='continuous_controller',
        alazar_name='ATS',
        server_name='Alazar_server')

    continuous_controller.average_mode('none')
    continuous_controller.samples_per_trace(1024 * 8)
    continuous_controller.traces_per_acquisition(1)
    continuous_controller.update_acquisition_settings(mode='CS',
                                                      samples_per_record=1024,
                                                      buffer_timeout=5000,
                                                      allocated_buffers=100)
    continuous_controller.setup()

    continuous_controller.acquisition()