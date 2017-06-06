##################
### Parameters ###
##################

parameters.AcquisitionParameter.layout = layout

dummy_parameter = ManualParameter(name='dummy', initial_value=42)

#####################
### DC parameters ###
#####################
DF = parameters.CombinedParameter(parameters=[LDF, RDF], name='DF')
DS = parameters.CombinedParameter(parameters=[LDS, RDS], name='DS')
DF_DS = parameters.CombinedParameter(parameters=[LDF, RDF, LDS, RDS],
                                     name='DF_DS')
turnon_parameter = parameters.CombinedParameter(parameters=[TG, LB, RB])
TGAC_DF_DS = parameters.CombinedParameter(parameters=[TGAC, LDF, RDF, LDS,
                                                      RDS])
LB_RB = parameters.CombinedParameter(parameters=[LB, RB])

for parameter in [DF, DS, DF_DS, turnon_parameter, TGAC_DF_DS, LB_RB]:
    station.add_component(parameter)

DCSweepPlot.gate_mapping = {'DF': 'RDF'}

##############################
### Acquisition parameters ###
##############################
DC_parameter = parameters.DCParameter()
EPR_parameter = parameters.EPRParameter()
T1_parameter = parameters.T1Parameter()
variable_read_parameter = parameters.VariableReadParameter()
adiabatic_ESR_parameter = parameters.AdiabaticParameter()
DC_sweep_parameter = parameters.DCSweepParameter()
# adiabatic_NMR_parameter = parameters.AdiabaticParameter()
# rabi_ESR_parameter = parameters.RabiParameter()
# rabi_drive_ESR_parameter =  parameters.RabiDriveParameter()
# # select_ESR_parameter = parameters.SelectFrequencyParameter(
# #     acquisition_parameter=adiabatic_ESR_parameter,
# #     mode='ESR', discriminant='contrast')
# dark_counts_parameter = parameters.DarkCountsParameter()
#
# plunge_voltage_parameter = general_parameters.ConfigPulseAttribute(
#     pulse_name='plunge', attribute='amplitude')

for parameter in [DC_parameter, EPR_parameter, variable_read_parameter,
                  adiabatic_ESR_parameter, T1_parameter]:
    station.add_component(parameter)