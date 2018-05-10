from qcodes.instrument.base import Instrument


class TriggerFPGAController(Instrument):
    def __init__(self, name, FPGA, **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter('trigger_interval',
                           unit='s',
                           set_cmd=self._set_trigger_interval)
        self.add_parameter('trigger_duration',
                           unit='s',
                           set_cmd=self._set_trigger_duration)
        self.add_parameter('pxi_port',
                           set_cmd=self._set_pxi_port)

        self.FPGA = FPGA
        self.sample_rate = 100e6
        self.stop()

    def _set_trigger_interval(self, trigger_interval):
        interval_cycles = int(round(trigger_interval * self.sample_rate))
        self.FPGA.set_fpga_pc_port(port=0,
                                   data=[interval_cycles],
                                   address=2,
                                   address_mode=0,
                                   access_mode=1)

    def _set_trigger_duration(self, trigger_duration):
        trigger_cycles = int(round(trigger_duration * self.sample_rate))
        self.FPGA.set_fpga_pc_port(port=0,
                                   data=[trigger_cycles],
                                   address=3,
                                   address_mode=0,
                                   access_mode=1)

    def _set_pxi_port(self, pxi_port):
        self.FPGA.set_fpga_pc_port(port=0,
                                   data=[pxi_port],
                                   address=4,
                                   address_mode=0,
                                   access_mode=1)

    def stop(self):
        # Also set trigger duration to zero because signal otherwise goes to high
        self._set_trigger_duration(0)
        self.FPGA.set_fpga_pc_port(port=0,
                                   data=[1, 0],
                                   address=0,
                                   address_mode=0,
                                   access_mode=1)

    def start(self):
        assert None not in [self.trigger_interval(),
                            self.trigger_duration(),
                            self.pxi_port()]

        # Reset trigger duration because it's set to 0 during stop
        self._set_trigger_duration(self.trigger_duration())
        self.FPGA.set_fpga_pc_port(port=0,
                                   data=[0, 1],
                                   address=0,
                                   address_mode=0,
                                   access_mode=1)

