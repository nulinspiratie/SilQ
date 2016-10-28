dummy_param = ManualParameter(name='dummy', initial_value=42)
DF_DS = general_parameters.CombinedParameter(parameters=[DF, DS])
ELR_parameter = measurement_parameters.ELR_Parameter(pulsemaster=pulsemaster)
# T1_parameter = measurement_parameters.T1_Parameter(pulsemaster=pulsemaster)
# DC_parameter = measurement_parameters.DC_Parameter(pulsemaster=pulsemaster)
# ELRLR_parameter = measurement_parameters.ELRLR_Parameter(pulsemaster=pulsemaster)
# variable_read_parameter = measurement_parameters.VariableRead_Parameter(pulsemaster=pulsemaster)

# Modify default parameter values
# parameters = [ELR_parameter, T1_parameter, ELRLR_parameter, variable_read_parameter]
# for parameter in parameters:
#     parameter.stages['load']['voltage'] = 2.5
#     parameter.stages['empty']['duration'] = 20


station = qc.Station(layout, arbstudio, pulseblaster, ATS, ATS_controller,
                     SIM900,
                     DF_DS, ELR_parameter, T1_parameter, DC_parameter,
                     ELRLR_parameter, variable_read_parameter)