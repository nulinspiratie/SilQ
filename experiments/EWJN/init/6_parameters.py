dummy_parameter = ManualParameter(name='dummy', initial_value=42)
DF_DS = general_parameters.CombinedParameter(parameters=[DF, DS])
DC_parameter = measurement_parameters.DC_Parameter(layout=layout)
ELR_parameter = measurement_parameters.ELR_Parameter(layout=layout)
T1_parameter = measurement_parameters.T1_Parameter(layout=layout)
variable_read_parameter = measurement_parameters.VariableRead_Parameter(layout=layout)
adiabatic_sweep_parameter = measurement_parameters.AdiabaticSweep_Parameter(layout=layout)

turnon_parameter = general_parameters.CombinedParameter(parameters=[TG, LB, RB])
TGAC_DF_DS = general_parameters.CombinedParameter(parameters=[TGAC, DF, DS])
LB_RB = general_parameters.CombinedParameter(parameters=[LB, RB])

# Add all our instruments and parameters for logging
station = qc.Station(SIM900, arbstudio, pulseblaster,
                     ATS, triggered_controller, continuous_controller,
                     keysight, layout,
                     DC_parameter, ELR_parameter, T1_parameter,
                     variable_read_parameter, adiabatic_sweep_parameter,
                     *SIM900_scaled_parameters)
