from silq.instrument_interfaces import InstrumentInterface


class M3201AInterface(InstrumentInterface):
    def stop(self):
        pass

    def setup(self):
        pass

    def start(self):
        pass

    def write_raw(self, cmd):
        pass

    def ask_raw(self, cmd):
        pass

    def get_final_additional_pulses(self, **kwargs):
        pass
