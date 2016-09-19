# Load instruments
import qcodes.instrument_drivers.lecroy.ArbStudio1104 as arbstudio_driver
import qcodes.instrument_drivers.spincore.PulseBlasterESRPRO as pulseblaster_driver
import qcodes.instrument_drivers.stanford_research.SIM900 as SIM900_driver
import qcodes.instrument_drivers.AlazarTech.ATS9440 as ATS_driver
import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ATS_controller_driver
from silq.meta_instruments.pulsemaster import PulseMaster as pulsemaster_driver

dll_path = os.path.join(os.getcwd(),'C:\lecroy_driver\\Library\\ArbStudioSDK.dll')
arbstudio = arbstudio_driver.ArbStudio1104('ArbStudio', dll_path)

pulseblaster = pulseblaster_driver.PulseBlaster('PulseBlaster', api_path='spinapi.py')

SIM900 = SIM900_driver.SIM900('SIM900', 'GPIB0::4::INSTR')
for ch_name, ch, max_voltage in [('TG',1,18), ('LB',2,3.8), ('RB',3,3.8), ('TGAC',4,3), 
                                 ('SRC',5,1), ('DS',7,3.2), ('DF',6,3.2)]:
    SIM900.define_slot(channel=ch, name=ch_name, max_voltage=max_voltage)
    SIM900.update()
    exec('{ch_name} = SIM900.parameters["{ch_name}"]'.format(ch_name=ch_name))


ATS = ATS_driver.ATS9440('ATS', server_name='Alazar_server')
ATS_controller = ATS_controller_driver.Basic_AcquisitionController(name='ATS_controller', 
                                                           alazar_name='ATS',
                                                           server_name='Alazar_server')

pulsemaster=pulsemaster_driver.PulseMaster(pulseblaster=pulseblaster, 
                                           arbstudio=arbstudio, 
                                           ATS=ATS, 
                                           ATS_controller=ATS_controller, 
                                           server_name='PulseMaster_server')

# Additional parameters
dummy_param = ManualParameter(name='dummy', initial_value=42)
DF_DS = general_parameters.CombinedParameter(parameters=[DF, DS])
ELR_parameter = measurement_parameters.ELR_Parameter(pulsemaster=pulsemaster)
T1_parameter = measurement_parameters.T1_Parameter(pulsemaster=pulsemaster)
DC_parameter = measurement_parameters.DC_Parameter(pulsemaster=pulsemaster)
ELRLR_parameter = measurement_parameters.ELRLR_Parameter(pulsemaster=pulsemaster)
variable_read_parameter = measurement_parameters.VariableRead_Parameter(pulsemaster=pulsemaster)

# Modify default parameter values
parameters = [ELR_parameter, T1_parameter, ELRLR_parameter, variable_read_parameter]
for parameter in parameters:
    parameter.stages['load']['voltage'] = 2.5
    parameter.stages['empty']['duration'] = 20


station = qc.Station(arbstudio, pulseblaster, ATS, ATS_controller, pulsemaster, SIM900,
                     DF_DS, ELR_parameter, T1_parameter, DC_parameter, ELRLR_parameter, variable_read_parameter)