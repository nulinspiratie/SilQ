##################
### Parameters ###
##################

acquisition_parameters.AcquisitionParameter.layout = layout

dummy_parameter = ManualParameter(name='dummy', initial_value=42)

#####################
### DC parameters ###
#####################
DF_DS = general_parameters.CombinedParameter(parameters=[DF, DS])
turnon_parameter = general_parameters.CombinedParameter(parameters=[TG, LB, RB])
TGAC_DF_DS = general_parameters.CombinedParameter(parameters=[TGAC, DF, DS])
LB_RB = general_parameters.CombinedParameter(parameters=[LB, RB])

##############################
### Acquisition parameters ###
##############################
DC_parameter = acquisition_parameters.DCParameter()
# EPR_parameter = acquisition_parameters.EPRParameter()
# T1_parameter = acquisition_parameters.T1Parameter()
# # variable_read = acquisition_parameters.VariableRead_Parameter(layout=layout)
# adiabatic_ESR_parameter = acquisition_parameters.AdiabaticParameter()
# adiabatic_NMR_parameter = acquisition_parameters.AdiabaticParameter()
# rabi_ESR_parameter = acquisition_parameters.RabiParameter()
# rabi_drive_ESR_parameter =  acquisition_parameters.RabiDriveParameter()
# # select_ESR_parameter = measurement_parameters.SelectFrequencyParameter(
# #     acquisition_parameter=adiabatic_ESR_parameter,
# #     mode='ESR', discriminant='contrast')
# dark_counts_parameter = acquisition_parameters.DarkCountsParameter()
#
# plunge_voltage_parameter = general_parameters.ConfigPulseAttribute(
#     pulse_name='plunge', attribute='amplitude')

for parameter in [DC_parameter]:
    station.add_component(parameter)