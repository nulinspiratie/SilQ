from .interface import InstrumentInterface, Channel
import qcodes as qc
station = qc.Station()

instrument_interfaces = {
    "ArbStudio1104": {
        "module": ".lecroy.ArbStudio1104_interface",
        "class": "ArbStudio1104Interface",
    },
    "MockArbStudio": {
        "module": ".lecroy.ArbStudio1104_interface",
        "class": "ArbStudio1104Interface",
    },
    "PulseBlasterDDS": {
        "module": ".spincore.PulseBlasterDDS_interface",
        "class": "PulseBlasterDDSInterface",
    },
    "PulseBlasterESRPRO": {
        "module": ".spincore.PulseBlasterESRPRO_interface",
        "class": "PulseBlasterESRPROInterface",
    },
    "MockPulseBlaster": {
        "module": ".spincore.PulseBlasterESRPRO_interface",
        "class": "PulseBlasterESRPROInterface",
    },
    "Chip": {"module": ".chip_interface", "class": "ChipInterface"},
    "ATS9440": {"module": ".AlazarTech.ATS_interface", "class": "ATSInterface"},
    "MockATS": {"module": ".AlazarTech.ATS_interface", "class": "ATSInterface"},
    "Keysight_E8267D": {"module": ".keysight", "class": "E8267DInterface"},
    "SD_AWG": {"module": ".keysight", "class": "Keysight_SD_AWG_Interface"},
    "SD_DIG": {"module": ".keysight", "class": "Keysight_SD_DIG_Interface"},
    "SD_FPGA": {"module": ".keysight", "class": "Keysight_SD_FPGA_Interface"},
    "Tektronix_AWG520": {
        "module": ".Tektronix.AWG520_interface",
        "class": "AWG520Interface",
    },
    "TriggerFPGAController": {
        "module": ".keysight.fpga_interfaces.trigger_FPGA_interface",
        "class": "TriggerFPGAInterface",
    },
    "Bayesian_Update_FPGA": {
        "module": ".keysight.fpga_interfaces.Bayesian_update_interface",
        "class": "Bayesian_Update_Interface",
    },
    "PCDDS": {
        "module": ".keysight.fpga_interfaces.PCDDS_interface",
        "class": "PCDDSInterface",
    },
    "Keysight_81180A": {"module": ".keysight", "class": "Keysight81180AInterface"},
    "RohdeSchwarz_SGS100A": {"module": ".rohde_schwarz", "class": "SGS100AInterface"},
}


def get_instrument_interface(instrument, *args, **kwargs):
    from . import instrument_interfaces

    instrument_class = instrument.__class__.__name__
    import_dict = instrument_interfaces[instrument_class]
    exec(f'from {import_dict["module"]} import {import_dict["class"]}')
    instrument_interface_class = eval(import_dict["class"])

    instrument_interface = instrument_interface_class(
        instrument_name=instrument.name, *args, **kwargs
    )
    station.add_component(instrument_interface)
    return instrument_interface
