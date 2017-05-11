##################
### Parameters ###
##################

parameters.AcquisitionParameter.layout = layout

dummy_parameter = ManualParameter(name='dummy', initial_value=42)

#####################
### DC parameters ###
#####################
DF_DS = parameters.CombinedParameter(parameters=[DF, DS])
turnon_parameter = parameters.CombinedParameter(parameters=[TG, LB, RB])
TGAC_DF_DS = parameters.CombinedParameter(parameters=[TGAC, DF, DS])
LB_RB = parameters.CombinedParameter(parameters=[LB, RB])

for parameter in [DF_DS]:
    station.add_component(parameter)
##############################
### Acquisition parameters ###
##############################
DC_parameter = parameters.DCParameter()
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