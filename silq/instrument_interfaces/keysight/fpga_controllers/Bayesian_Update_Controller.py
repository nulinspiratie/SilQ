from qcodes.instrument_drivers.keysight.SD_common.SD_FPGA import SD_FPGA
from qcodes.utils.validators import Ints
import numpy as np

class Bayesian_Update_FPGA(SD_FPGA):
    """
            This class makes use of a Bayesian update module onboard the FPGA
            of a Keysight digitizer board.
        """

    def __init__(self, name, chassis, slot, **kwargs):
        super().__init__(name, chassis, slot, **kwargs)

        self.write_port_signals = {
            "reset":            {"address": 0x0, 'width': 1, 'type':bool},
            "enable":           {"address": 0x1, 'width': 1, 'type':bool},
            "blip_threshold":   {"address": 0x2, 'width': 1, 'type':np.int16},
            "blip_timeout":     {"address": 0x3, 'width': 1, 'type':np.uint32},
            "trace_sel":        {"address": 0x4, 'width': 1, 'type':np.uint8}
        }

        self.read_port_signals = {
            "counter": {"address": 0, 'width': 1, 'type':np.uint32},
        }

        self.add_parameter(
            'blip_threshold',
            set_cmd=self._set_blip_threshold,
            validator=Ints(-0x8000, 0x7FFF),
            docstring='The blip threshold in the range of [-0x8000, 0x7FFF].'
        )
        self.add_parameter(
            'blip_timeout',
            set_cmd=self._set_blip_timeout,
            validator=Ints(0, 0xFFFFFFFF),
            docstring='The blip timeout in samples.'
        )
        self.add_parameter(
            'trace_select',
            set_cmd=self._set_trace_select,
            validator=Ints(0,7),
            docstring='The channel you want to select from 0 to 7.'
        )

    def _set_blip_threshold(self, threshold):
        """ Sets the blip threshold to reset the Bayesian update timer. """
        self.set_fpga_pc_port(0, [threshold],
                              self.write_port_signals['blip_threshold']['address'], 1, 1)

    def _set_blip_timeout(self, timeout):
        """ Sets the blip timeout for the trigger of the Bayesian update module. """
        self.set_fpga_pc_port(0, [timeout],
                              self.write_port_signals['blip_timeout']['address'], 1, 1)

    def _set_trace_select(self, trace_sel):
        self.set_fpga_pc_port(0, [trace_sel],
                              self.write_port_signals['trace_sel']['address'], 1, 1)

    def _get_counter_value(self):
        return self.get_fpga_pc_port(0, 1,
                                     self.read_port_signals['counter']['address'], 1, 1)

    def start(self):
        """ Starts the currently loaded FPGA firmware. """
        # Also pulls module out of reset
        self.set_fpga_pc_port(0, [0, 1],
                              self.write_port_signals['reset']['address'], 1, 1)

    def stop(self):
        """ Stops the currently loaded FPGA firmware. """
        self.set_fpga_pc_port(0, [0],
                              self.write_port_signals['enable']['address'], 1, 1)

    def reset(self):
        """ Resets the currently loaded FPGA firmware. """
        self.set_fpga_pc_port(0, [1],
                              self.write_port_signals['reset']['address'], 1, 1)