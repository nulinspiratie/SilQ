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
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)
        self.pulse_sequence.allow_targeted_pulses = True
        self.pulse_sequence.allow_untargeted_pulses = False
        self.pulse_sequence.allow_pulse_overlap = False

        self._clk_freq = int(100e6)

        self.pulse_implementations = [
            TriggerPulseImplementation(
                pulse_requirements=[]
            )
        ]

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
            vals=vals.Numbers(-3,3),
            unit='V',
            docstring='The blip threshold in volts.'
        )

        self.add_parameter(
            'full_scale_range',
            parameter_class=ManualParameter,
            vals=vals.Numbers(-3,3),
            unit='V',
            docstring='The full scale range of the trace channel in volts.'
        )

        self.add_parameter(
            'update_time',
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            unit='s',
            docstring='The length of time in seconds after a blip required to perform the '
                      'Bayesian update.'
        )

        self.add_parameter(
            'timeout',
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            unit='s',
            docstring='The duration since the Bayesian update starts from where you cancel the '
                      'Bayesian update and continue with the experiment.'
        )

        self.add_parameter(
            'trace_select',
            parameter_class=ManualParameter,
            vals=vals.Ints(0, 7),
            docstring='The channel you want to select from 0 to 7.'
        )

        self.add_parameter(
            'pxi_channel',
            parameter_class=ManualParameter,
            vals=vals.Ints(4000,4007),
            docstring='The PXI channel the trigger will be output on.'
        )

        # self.add_parameter(
        #     'timer_duration',
        #     parameter_class=ManualParameter,
        #     vals=Numbers(),
        #     unit='s',
        #     docstring='The duration of the post-trigger wait time before re-priming '
        #               'the Bayesian update.'
        # )

    def setup(self, **kwargs):
        ctrl = self.instrument
        ctrl.reset()

        # Implement each pulse, sets the PXI trigger
        for pulse in self.pulse_sequence:
            for key, value in pulse.implementation.implement().items():
                self.parameters[key](value)


        # Set the PXI trigger, translate from pxi id to range [0,7]
        ctrl.pxi_select(self.pxi_channel() - 4000)

        # Set the trace_select
        ctrl.trace_select(self.trace_select())
        # 10% overhead in resetting the pulse sequence
        timer_duration = self.pulse_sequence.duration
        ctrl.timer_duration(int(round(timer_duration*self._clk_freq)))

        # Scale signal to volts
        v_min = - self.full_scale_range()
        v_max = - v_min
        m = interp1d([v_min, v_max], [-0x8000, 0x7FFF])

        ctrl.blip_threshold(np.int16(m(self.blip_threshold())))

        # Set the blip_timeout
        update_ticks = int(self.update_time()*self._clk_freq)
        ctrl.blip_t_wait(update_ticks)
        timeout_ticks = int(self.timeout()*self._clk_freq)
        ctrl.blip_timeout(timeout_ticks)

        # targeted_pulse_sequence is the pulse sequence that is currently setup
        self.targeted_pulse_sequence = self.pulse_sequence
        self.targeted_input_pulse_sequence = self.input_pulse_sequence

    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()

    def reset(self):
        self.instrument.reset()


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

        return {"pxi_channel" : channel}
