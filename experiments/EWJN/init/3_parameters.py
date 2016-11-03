dummy_param = ManualParameter(name='dummy', initial_value=42)
DF_DS = general_parameters.CombinedParameter(parameters=[DF, DS])
ELR_parameter = measurement_parameters.ELR_Parameter(layout=layout)
# T1_parameter = measurement_parameters.T1_Parameter(pulsemaster=pulsemaster)
DC_parameter = measurement_parameters.DC_Parameter(layout=layout)
# ELRLR_parameter = measurement_parameters.ELRLR_Parameter(pulsemaster=pulsemaster)
#variable_read_parameter = measurement_parameters.VariableRead_Parameter(pulsemaster=pulsemaster)

# Modify default parameter values
# parameters = [ELR_parameter, T1_parameter, ELRLR_parameter, variable_read_parameter]
# for parameter in parameters:
#     parameter.stages['load']['voltage'] = 2.5
#     parameter.stages['empty']['duration'] = 20

turnon_param = general_parameters.CombinedParameter(parameters=[TG, LB, RB])
TGAC_DF_DS = general_parameters.CombinedParameter(parameters=[TGAC, DF, DS])
LB_RB = general_parameters.CombinedParameter(parameters=[LB, RB])

station = qc.Station(layout, arbstudio, pulseblaster, ATS, ATS_controller,
                     SIM900, ELR_parameter, DC_parameter)
                     # DF_DS, ELR_parameter, T1_parameter, DC_parameter,
                     # ELRLR_parameter, variable_read_parameter)