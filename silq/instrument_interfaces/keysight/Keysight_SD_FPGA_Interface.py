from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses.pulse_modules import PulseImplementation
from silq.pulses.pulse_types import TriggerPulse
from silq.meta_instruments.layout import SingleConnection
from qcodes import ManualParameter
import qcodes.utils.validators as vals
import logging

from scipy.interpolate import interp1d
import numpy as np

logger = logging.getLogger(__name__)

class Bayesian_Update_Interface(InstrumentInterface):
    def __init__(self, instrument_name, fpga_controller_name=None, **kwargs):
        super().__init__(instrument_name, **kwargs)
        self.pulse_sequence.allow_targeted_pulses = True
        self.pulse_sequence.allow_untargeted_pulses = False
        self.pulse_sequence.allow_pulse_overlap = False

        self.pulse_implementations = [
            TriggerPulseImplementation(
                pulse_requirements=[]
            )
        ]

        if fpga_controller_name is not None:
            self.fpga_controller = self.find_instrument(fpga_controller_name)
        else:
            #TODO: Should this be required?
            pass

        self._data_channels = {
            f'trace_{k}':
                Channel(instrument_name=self.instrument_name(),
                        name=f'trace_{k}', id=k,
                            output=False, input=True)
            for k in range(8)
        }

        self._pxi_channels = {
            'pxi{}'.format(k):
                Channel(instrument_name=self.instrument_name(),
                        name='pxi{}'.format(k), id=4000 + k,
                        input_trigger=True, output=True, input=True) for k in
            range(8)
        }

        self._channels = {
            **self._data_channels,
            **self._pxi_channels,
        }

        self.add_parameter(
            'blip_threshold',
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            docstring='The blip threshold in volts.'
        )

        self.add_parameter(
            'full_scale_range',
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            docstring='The full scale range of the trace channel in volts.'
        )

        self.add_parameter(
            'channel_offset',
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            docstring='The voltage offset of the trace channel.'
        )

        self.add_parameter(
            'blip_timeout',
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            docstring='The blip timeout in seconds.'
        )

        self.add_parameter(
            'trace_select',
            parameter_class=ManualParameter,
            vals=vals.Ints(0, 7),
            docstring='The channel you want to select from 0 to 7.'
        )

        self.add_parameter(
            'pxi_select',
            parameter_class=ManualParameter,
            vals=vals.Ints(4000,4007),
            docstring='The PXI channel the trigger will be output on.'
        )

        self.add_parameter(
            'clk_freq',
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            docstring='The onboard clock frequency of the digitizer module.'
        )

    def setup(self, **kwargs):
        ctrl = self.fpga_controller
        ctrl.reset()

        # Implement each pulse, sets the PXI trigger
        for pulse in self.pulse_sequence:
            self.pxi_select(pulse.implementation.implement())

        # Set the PXI trigger, translate from pxi id to range [0,7]
        ctrl.pxi_select(self.pxi_select() - 4000)

        # Set the trace_select
        ctrl.trace_select(self.trace_select())

        # Set the blip_threshold
        threshold = (self.blip_threshold() - self.channel_offset())
        # Scale signal to volts
        v_min = - self.full_scale_range()
        v_max = self.full_scale_range()
        m = interp1d([v_min, v_max], [-0x8000, 0x7FFF])

        ctrl.blip_threshold(np.int16(m(threshold)))

        # Set the blip_timeout
        timeout = int(self.blip_timeout()*self.clk_freq())
        ctrl.blip_timeout(timeout)

    def start(self):
        self.fpga_controller.start()

    def stop(self):
        self.fpga_controller.stop()

    def reset(self):
        self.fpga_controller.reset()


class TriggerPulseImplementation(PulseImplementation):
    pulse_class = TriggerPulse
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, **kwargs)

    def implement(self):
        if isinstance(self.pulse.connection, SingleConnection):
            channel = self.pulse.connection.output['channel'].id
        else:
            raise Exception('No implementation for connection {}'.format(
                self.pulse.connection))

        return channel



