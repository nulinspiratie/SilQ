from silq.instrument_interfaces import InstrumentInterface
from qcodes import ManualParameter
from qcodes.utils.validators as vals
import logging
logger = logging.getLogger(__name__)

class Bayesian_Update_Interface(InstrumentInterface):
    def __init__(self, instrument_name, fpga_controller_name=None, **kwargs):
        super().__init__(instrument_name, **kwargs)
        self.pulse_sequence.allow_targeted_pulses = True
        self.pulse_sequence.allow_untargeted_pulses = False
        self.pulse_sequence.allow_pulse_overlap = False

        if fpga_controller_name is not None:
            self.fpga_controller = self.find_instrument(fpga_controller_name)
        else:
            #TODO: Should this be required?
            pass

        self._pxi_channels = {
            'pxi{}'.format(k):
                Channel(instrument_name=self.instrument_name(),
                        name='pxi{}'.format(k), id=4000 + k,
                        input_trigger=True, output=True, input=True) for k in
            range(8)
        }

        self._channels = {
            **self._pxi_channels,
        }

        self.add_parameter(
            'blip_threshold',
            parameter_class=ManualParameter,
            set_cmd=self._set_blip_threshold,
            validator=Numbers(),
            docstring='The blip threshold in the range of [-0x8000, 0x7FFF].'
        )

        self.add_parameter(
            'full_scale_range',
            parameter_class=ManualParameter,
            validator=Numbers(),
            docstring='The full scale range of the trace channel.'
        )

        self.add_parameter(
            'channel_offset',
            parameter_class=ManualParameter,
            validator=Numbers(),
            docstring='The voltage offset of the trace channel.'
        )

        self.add_parameter(
            'blip_timeout',
            parameter_class=ManualParameter,
            set_cmd=self._set_blip_timeout,
            validator=Numbers(),
            docstring='The blip timeout in seconds.'
        )

        self.add_parameter(
            'trace_select',
            parameter_class=ManualParameter,
            set_cmd=self._set_trace_select,
            validator=Ints(0, 7),
            docstring='The channel you want to select from 0 to 7.'
        )

        self.add_parameter(
            'pxi_select',
            parameter_class=ManualParameter,
            validator=vals.Ints(0,7),
            docstring='The PXI channel the trigger will be output on.'
        )

        self.add_parameter(
            'sample_rate',
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            docstring='The sample rate of the digitizer.'
        )

    #
    #   Interface Functions
    #

    def get_final_additional_pulses(self, **kwargs):
        pass

    def setup(self, **kwargs):
        ctrl = self.fpga_controller
        ctrl.reset()

        # Implement each pulse, sets the PXI trigger
        for pulse in self.pulse_sequence:
            self.pxi_select(pulse.implement())

        # Set the PXI trigger
        ctrl.pxi_select(self.pxi_select)

        # Set the trace_select
        ctrl.trace_select(self.trace_select())

        # Set the blip_threshold
        threshold = (self.blip_threshold() - self.channel_offset())/self.full_scale_range()
        ctrl.blip_threshold(threshold)

        # Set the blip_timeout
        timeout = int(self.blip_timeout()*self.sample_rate())
        ctrl.blip_timeout(timeout)

    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()

    def reset(self):
        self.instrument.reset()


    class TriggerPulseImplementation(PulseImplementation, TriggerPulse):
        def __init__(self, **kwargs):
            PulseImplementation.__init__(self, pulse_class=TriggerPulse,
                                         **kwargs)

        def implement(self, instrument, sampling_rates, threshold):
            if isinstance(self.connection, SingleConnection):
                channel = self.connection.output['channel'].name
            else:
                raise Exception('No implementation for connection {}'.format(
                    self.connection))

            return channel



