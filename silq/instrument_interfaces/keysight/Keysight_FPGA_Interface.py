from silq.instrument_interfaces import InstrumentInterface
import logging
logger = logging.getLogger(__name__)

class Bayesian_Update_Interface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)
        self.pulse_sequence.allow_targeted_pulses = False
        self.pulse_sequence.allow_untargeted_pulses = False
        self.pulse_sequence.allow_pulse_overlap = False


    def get_final_additional_pulses(self, **kwargs):
        pass

    def setup(self, **kwargs):
        self.instrument

    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()

    def reset(self):
        self.instrument.reset()