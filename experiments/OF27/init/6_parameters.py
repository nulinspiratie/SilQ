acquisition_parameters.AcquisitionParameter.layout = layout

dummy_parameter = ManualParameter(name='dummy', initial_value=42)
DF_DS = general_parameters.CombinedParameter(parameters=[DF, DS])
# turnon_parameter = general_parameters.CombinedParameter(parameters=[TG, LB, RB])
# PL_DF_DS = general_parameters.CombinedParameter(parameters=[PL, DF, DS])
# LB_RB = general_parameters.CombinedParameter(parameters=[LB, RB])
#
DC_parameter = acquisition_parameters.DCParameter()
# EPR_parameter = acquisition_parameters.EPRParameter()
# T1_parameter = acquisition_parameters.T1Parameter(mode='ESR')
# variable_read_parameter = acquisition_parameters.VariableReadParameter()
# adiabatic_ESR_parameter = acquisition_parameters.AdiabaticParameter(mode='ESR')
# adiabatic_NMR_parameter = acquisition_parameters.AdiabaticParameter(mode='ESR')
# rabi_ESR_parameter = acquisition_parameters.RabiParameter(mode='ESR')
# rabi_drive_ESR_parameter =  acquisition_parameters.RabiDriveParameter(mode='ESR')
# # select_ESR_parameter = measurement_parameters.SelectFrequencyParameter(
# #     acquisition_parameter=adiabatic_ESR_parameter,
# #     mode='ESR', discriminant='contrast')
# dark_counts_parameter = acquisition_parameters.DarkCountsParameter()
#
# plunge_voltage_parameter = general_parameters.ConfigPulseAttribute(
#     pulse_name='plunge', attribute='amplitude')
#
# # Add all our instruments and parameters for logging
station = qc.Station(
    SIM900, arbstudio, ATS, triggered_controller,
    layout,
    DC_parameter,
    DF_DS,
    *SIM900_scaled_parameters)
