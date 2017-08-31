# from qcodes.instrument_drivers.keysight.SD_common.SD_FPGA import SD_FPGA
from qcodes.instrument.base import Instrument
from qcodes.instrument_drivers.Keysight.M3300A import Keysight_M3300A_FPGA
from qcodes.utils.validators import Ints
from scipy.interpolate import interp1d
import numpy as np

class Bayesian_Update_FPGA(Instrument):
    """
            This class makes use of a Bayesian update module onboard the FPGA
            of a Keysight digitizer board.
        """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.fpga = Keysight_M3300A_FPGA('FPGA')
        self.port = 0

        self.write_port_signals = {
            "reset":            {"address": 0x0, 'width': 1, 'type':bool},
            "enable":           {"address": 0x1, 'width': 1, 'type':bool},
            "blip_threshold":   {"address": 0x2, 'width': 1, 'type':np.int16},
            "blip_timeout":     {"address": 0x3, 'width': 1, 'type':np.uint32},
            "trace_sel":        {"address": 0x4, 'width': 1, 'type':np.uint8},
            "pxi_sel":          {"address": 0x5, 'width': 1, 'type':np.uint8}
        }

        self.read_port_signals = {
            "counter": {"address": 0, 'width': 1, 'type':np.uint32},
            "trace_out": {"address": 1, 'width': 1, 'type': np.uint32},
        }

        self.add_parameter(
            'blip_threshold',
            set_cmd=self._set_blip_threshold,
            vals=Ints(-0x8000, 0x7FFF),
            docstring='The blip threshold in the range of [-0x8000, 0x7FFF].'
        )
        self.add_parameter(
            'blip_timeout',
            set_cmd=self._set_blip_timeout,
            vals=Ints(0, 0xFFFFFFFF),
            docstring='The blip timeout in samples.'
        )
        self.add_parameter(
            'trace_select',
            set_cmd=self._set_trace_select,
            vals=Ints(0,7),
            docstring='The channel you want to select from 0 to 7.'
        )

        self.add_parameter(
            'pxi_select',
            set_cmd=self._set_pxi_select,
            vals=Ints(0,7),
            docstring='The PXI channel the trigger will be output on, from 0 to 7.'
        )

    def _set_blip_threshold(self, threshold):
        """ Sets the blip threshold to reset the Bayesian update timer. """
        # v_min = -3;
        # v_max = 3
        # m = interp1d([v_min, v_max], [-0x8000, 0x7FFF])
        # threshold = m(threshold)
        self.fpga.set_fpga_pc_port(self.port, [threshold],
                              self.write_port_signals['blip_threshold']['address'], 0, 1)

    def _set_blip_timeout(self, timeout):
        """ Sets the blip timeout for the trigger of the Bayesian update module. """
        self.fpga.set_fpga_pc_port(self.port, [timeout],
                              self.write_port_signals['blip_timeout']['address'], 0, 1)

    def _set_trace_select(self, trace_sel):
        self.fpga.set_fpga_pc_port(self.port, [trace_sel],
                              self.write_port_signals['trace_sel']['address'], 0, 1)

    def _set_pxi_select(self, pxi_sel):
        self.fpga.set_fpga_pc_port(self.port, [pxi_sel],
                              self.write_port_signals['pxi_sel']['address'], 0, 1)

    def _get_counter_value(self):
        return self.fpga.get_fpga_pc_port(self.port, 1,
                                     self.read_port_signals['counter']['address'], 0, 1)

    def start(self):
        """ Starts the currently loaded FPGA firmware. """
        # Also pulls module out of reset
        self.fpga.set_fpga_pc_port(self.port, [0, 1],
                              self.write_port_signals['reset']['address'], 0, 1)

    def stop(self):
        """ Stops the currently loaded FPGA firmware. """
        self.fpga.set_fpga_pc_port(self.port, [0],
                              self.write_port_signals['enable']['address'], 0, 1)

    def reset(self):
        """ Resets the currently loaded FPGA firmware. """
        self.fpga.set_fpga_pc_port(self.port, [1],
                              self.write_port_signals['reset']['address'], 0, 1)