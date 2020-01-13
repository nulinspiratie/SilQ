from qcodes.instrument.base import Instrument
from qcodes.instrument_drivers.Keysight.SD_common.SD_FPGA import SD_FPGA
from qcodes.utils import validators as vals
from scipy.interpolate import interp1d
import numpy as np
from functools import partial

class Bayesian_Update_FPGA(Instrument):
    """
            This class makes use of a Bayesian update module onboard the FPGA
            of a Keysight digitizer board.
        """

    def __init__(self, name, FPGA, **kwargs):
        super().__init__(name, **kwargs)

        self._clk_freq = int(100e6)
        self.FPGA = FPGA
        self.port = 0

        self.write_port_signals = {
            "reset":            {"address": 0, 'width': 1, 'type': bool},
            "enable":           {"address": 1, 'width': 1, 'type': bool},
            "blip_threshold":   {"address": 2, 'width': 1, 'type': np.int16},
            "blip_t_wait":      {"address": 3, 'width': 1, 'type': np.uint32},
            "blip_timeout":     {"address": 4, 'width': 1, 'type': np.uint32},
            "trace_sel":        {"address": 5, 'width': 1, 'type': np.uint8},
            "pxi_sel":          {"address": 6, 'width': 1, 'type': np.uint8},
            "timer_threshold":  {"address": 7, 'width': 2, 'type': list},
            "trigger_delay":    {"address": 9, 'width': 1, 'type': np.uint32}
        }



        self.read_port_signals = {
            "update_counter":   {"address": 0, 'width': 1, 'type': np.uint32},
            "timeout_counter":  {"address": 1, 'width': 1, 'type': np.uint32},
            "trigger_counter":  {"address": 2, 'width': 1, 'type': np.uint32},
            "timer_counter":    {"address": 3, 'width': 2, 'type': np.uint64},
            "trace_out":        {"address": 5, 'width': 1, 'type': np.int16},
            "bayesian_state":   {"address": 6, 'width': 1, 'type': np.int16},
        }

        self.add_parameter(
            'blip_threshold',
            set_cmd=self._set_blip_threshold,
            vals=vals.Ints(-0x8000, 0x7FFF),
            docstring='The blip threshold in the range of [-0x8000, 0x7FFF].'
        )

        self.add_parameter(
            'blip_t_wait',
            set_cmd=self._set_blip_t_wait,
            vals=vals.Ints(0, 0xFFFFFFFF),
            docstring='The blip timeout in samples.'
        )
        self.add_parameter(
            'blip_timeout',
            set_cmd=self._set_blip_timeout,
            vals=vals.Ints(0, 0xFFFFFFFF),
            docstring='The blip timeout in samples.'
        )
        self.add_parameter(
            'trace_select',
            set_cmd=self._set_trace_select,
            vals=vals.Ints(0,7),
            docstring='The channel you want to select from 0 to 7.'
        )

        self.add_parameter(
            'pxi_select',
            set_cmd=self._set_pxi_select,
            vals=vals.Ints(0,7),
            docstring='The PXI channel the trigger will be output on, from 0 to 7.'
        )

        self.add_parameter(
            'timer_duration',
            set_cmd=self._set_timer_duration,
            vals=vals.Ints(0,int(2**48-1)),
            docstring="The post-update timer duration to match pulse sequence duration."
        )

    def _set_blip_threshold(self, threshold):
        """ Sets the blip threshold to reset the Bayesian update timer. """
        # v_min = -3;
        # v_max = 3
        # m = interp1d([v_min, v_max], [-0x8000, 0x7FFF])
        # threshold = m(threshold)
        self.FPGA.set_fpga_pc_port(self.port, [threshold],
                                   self.write_port_signals['blip_threshold']['address'], 1, 1)

    def _set_blip_t_wait(self, t_wait):
        """ Sets the post-blip wait time for the trigger of the Bayesian update module. """
        self.FPGA.set_fpga_pc_port(self.port, [t_wait],
                                   self.write_port_signals['blip_t_wait']['address'], 1, 1)

    def _set_blip_timeout(self, timeout):
        """ Sets the maximum timeout from the beginning of the Bayesian update procedure. """
        self.FPGA.set_fpga_pc_port(self.port, [timeout],
                                   self.write_port_signals['blip_timeout']['address'], 1, 1)

    def _set_trace_select(self, trace_sel):
        self.FPGA.set_fpga_pc_port(self.port, [trace_sel],
                                   self.write_port_signals['trace_sel']['address'], 1, 1)

    def _set_pxi_select(self, pxi_sel):
        self.FPGA.set_fpga_pc_port(self.port, [pxi_sel],
                                   self.write_port_signals['pxi_sel']['address'], 1, 1)

    def _set_timer_duration(self, timer_duration):
        # Split timer duration into 2 32 bit words.
        int_max = np.iinfo(np.uint32).max
        #                     Low word                , High word
        vals_list = np.array([timer_duration & int_max, timer_duration >> 32], dtype=np.uint32)
        self.FPGA.set_fpga_pc_port(self.port, vals_list,
                                   self.write_port_signals['timer_threshold']['address'], 0, 1)

    def _get_update_counter_value(self, samples=1):
        return self.FPGA.get_fpga_pc_port(self.port, samples,
                                          self.read_port_signals['update_counter']['address'], 1, 1)

    def _get_update_timeout_counter_value(self, samples=1):
        return self.FPGA.get_fpga_pc_port(self.port, samples,
                                          self.read_port_signals['timeout_counter']['address'], 1, 1)

    def _get_trigger_counter_value(self, samples=1):
        return self.FPGA.get_fpga_pc_port(self.port, samples,
                                          self.read_port_signals['trigger_counter']['address'], 1, 1)

    def _get_timer_counter_value_lo(self, samples=1):
        # Just read the lo word from the pulse sequence timer counter
        vals = self.FPGA.get_fpga_pc_port(self.port, samples,
                                          self.read_port_signals['timer_counter']['address'], 1, 1)
        return vals

    def _get_timer_counter_value(self, samples=1):
        vals = self.FPGA.get_fpga_pc_port(self.port, 2,
                                          self.read_port_signals['timer_counter']['address'], 0, 1)
        # Split timer duration into 2 32 bit words.
        # vals_list = [timer_duration >> 32, timer_duration & int_max]
        vals.dtype = np.uint64 # The zeroth word is the lowest word (little-endian)
        return vals

    def _get_trace_voltage(self, samples=1):
        val = self.FPGA.get_fpga_pc_port(self.port, samples,
                                   self.read_port_signals['trace_out'][
                                       'address'], 1, 1)
        val.dtype = self.read_port_signals['trace_out']['type']
        # Val is currently a 32 bit word, setting the data type to 16 bits will
        # convert this into two 16-bit words, so we just select the relevant ones
        return val[::2]

    def _get_bayesian_state(self, samples=1):
        """ Returns the internal state of the Bayesian update FSM (finite-state machine)
            Returns:
                state:
                    0 : "Performing Bayesian update"
                    1 : "Outputting trigger"
                    2 : "Post-trigger delay until end of sequence"
                    3 : "Invalid state"
        """
        return self.FPGA.get_fpga_pc_port(self.port, samples,
                                   self.read_port_signals['bayesian_state'][
                                       'address'], 1, 1)

    # def _get_counter_value(self):
    #     return self.FPGA.get_fpga_pc_port(self.port, 1,
    #                                       self.read_port_signals['counter']['address'], 1, 1)

    def start(self):
        """ Starts the currently loaded FPGA firmware. """
        # Also pulls module out of reset
        self.FPGA.set_fpga_pc_port(self.port, [0, 1],
                                   self.write_port_signals['reset']['address'], 0, 1)

    def stop(self):
        """ Stops the currently loaded FPGA firmware. """
        self.FPGA.set_fpga_pc_port(self.port, [0],
                                   self.write_port_signals['enable']['address'], 0, 1)

    def reset(self):
        """ Resets the currently loaded FPGA firmware. """
        self.FPGA.set_fpga_pc_port(self.port, [1],
                                   self.write_port_signals['reset']['address'], 0, 1)