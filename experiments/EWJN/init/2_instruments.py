# Load instruments
import qcodes.instrument_drivers.lecroy.ArbStudio1104 as arbstudio_driver
import qcodes.instrument_drivers.spincore.PulseBlasterESRPRO as pulseblaster_driver
import qcodes.instrument_drivers.stanford_research.SIM900 as SIM900_driver
import qcodes.instrument_drivers.AlazarTech.ATS9440 as ATS_driver
import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ATS_controller_driver
from silq.meta_instruments.pulsemaster import PulseMaster as pulsemaster_driver
from silq.parameters.general_parameters import ScaledParameter

dll_path = os.path.join(os.getcwd(),'C:\lecroy_driver\\Library\\ArbStudioSDK.dll')
arbstudio = arbstudio_driver.ArbStudio1104('ArbStudio', dll_path, server_name='')

pulseblaster = pulseblaster_driver.PulseBlaster('PulseBlaster',
                                                api_path='spinapi.py',
                                                server_name='')

SIM900 = SIM900_driver.SIM900('SIM900', 'GPIB0::4::INSTR',
                              server_name='')

# Each DC voltage source has format (name, slot number, divider, max raw voltage)
DC_sources = [('TG',1,8,18), ('LB',2,4,8), ('RB',3,4,8), ('TGAC',4,5,4),
         ('SRC',5,1,1), ('DS',7,4,3.2), ('DF',6,4,3.2)]
for ch_name, ch, ratio,max_voltage in DC_sources:
    SIM900.define_slot(channel=ch, name=ch_name+'_raw', max_voltage=max_voltage)
    SIM900.update()
    param_raw = SIM900.parameters[ch_name+'_raw']
    param = ScaledParameter(param_raw, name=ch_name, label=ch_name, ratio=ratio)

    exec('{ch_name}_raw = param_raw'.format(ch_name=ch_name))
    exec('{ch_name} = param'.format(ch_name=ch_name))

ATS = ATS_driver.ATS9440('ATS', server_name='Alazar_server')
ATS_controller = ATS_controller_driver.Basic_AcquisitionController(name='ATS_controller', 
                                                           alazar_name='ATS',
                                                           server_name='Alazar_server')

pulsemaster=pulsemaster_driver.PulseMaster(pulseblaster=pulseblaster, 
                                           arbstudio=arbstudio, 
                                           ATS=ATS, 
                                           ATS_controller=ATS_controller, 
                                           server_name='PulseMaster_server')
