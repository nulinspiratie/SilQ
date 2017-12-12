from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses.pulse_types import TriggerPulse
from qcodes.utils import validators as vals
from qcodes import ManualParameter
import logging
logger = logging.getLogger(__name__)

import numpy as np

class Keysight_SD_DIG_interface(InstrumentInterface):
    def __init__(self, instrument_name, acquisition_controller_names=[], **kwargs):
        super().__init__(instrument_name, **kwargs)
        self.pulse_sequence.allow_untargeted_pulses = True
        self.pulse_sequence.allow_pulse_overlap = True

        # Initialize channels
        self._acquisition_channels  = {
            'ch{}'.format(k): Channel(instrument_name=self.instrument_name(),
                                      name='ch{}'.format(k), id=k, input=True)
            for k in range(self.instrument.n_channels)
            }

        self._pxi_channels = {
            'pxi{}'.format(k):
                Channel(instrument_name=self.instrument_name(),
                        name='pxi{}'.format(k), id=4000 + k,
                        input_trigger=True, output=True, input=True) for k in
        range(self.instrument.n_triggers)}

        self._channels = {
            **self._acquisition_channels ,
            **self._pxi_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input=True),
        }

        # Organize acquisition controllers
        self.acquisition_controllers = {}
        for acquisition_controller_name in acquisition_controller_names:
            self.add_acquisition_controller(acquisition_controller_name)

        self.add_parameter(name='default_acquisition_controller',
                           parameter_class=ManualParameter,
                           initial_value='None',
                           vals=vals.Enum(None,
                               'None', *self.acquisition_controllers.keys()))

        self.add_parameter(name='acquisition_controller',
                           parameter_class=ManualParameter,
                           vals=vals.Enum(
                               'None', *self.acquisition_controllers.keys()))

        # Names of acquisition channels [chA, chB, etc.]
        self.add_parameter(name='acquisition_channels',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Anything())

        # Add ManualParameters which will be distributed to the active acquisition
        # controller during the setup routine
        self.add_parameter(
            'sample_rate',
            vals=vals.Numbers(),
            parameter_class=ManualParameter
        )

        self.add_parameter(
            'samples',
            vals=vals.Numbers(),
            parameter_class=ManualParameter
        )

        self.add_parameter(
            'channel_selection',
            vals=vals.Anything(),
            parameter_class=ManualParameter
        )

        self.add_parameter(
            'trigger_channel',
            vals=vals.Enum(0, 1, 2, 3, 4, 5, 6, 7),
            parameter_class=ManualParameter
        )

        self.add_parameter(
            'trigger_edge',
            vals=vals.Enum('rising', 'falling', 'both'),
            parameter_class=ManualParameter
        )

        self.add_parameter(
            'trigger_threshold',
            parameter_class=ManualParameter
        )

        # Set up the driver to a known default state
        self.initialize_driver()

    @property
    def _acquisition_controller(self):
        return self.acquisition_controllers.get(
            self.acquisition_controller(), None)

    def acquisition(self):
        """
        Perform acquisition
        """
        self.start()
        data = {}
        self._acquisition_controller.pre_acquire()
        acq_data = self._acquisition_controller.acquire()
        acq_data = self._acquisition_controller.post_acquire(acq_data)

        acquisition_average_mode = self._acquisition_controller.average_mode()

        # The start of acquisition
        t_0 = min(pulse.t_start for pulse in
                  self.pulse_sequence.get_pulses(acquire=True))

        # Split data into pulse traces
        for pulse in self.pulse_sequence.get_pulses(acquire=True):
            name = pulse.full_name
            data[name] = {}
            for ch in self.channel_selection():
                ch_data = acq_data[ch]
                ch_name = 'ch{}'.format(ch)
                ts = (pulse.t_start - t_0, pulse.t_stop - t_0)
                sample_range = [int(round(t * self.sample_rate())) for t in ts]
                pts = len(sample_range)

                # Extract pulse data from the channel data
                if acquisition_average_mode == 'none':
                    data[name][ch_name] = ch_data[:, sample_range[0]:sample_range[1]]
                    # Further average the pulse data
                    if pulse.average == 'none':
                        pass
                    elif pulse.average == 'trace':
                        data[name][ch_name] = np.mean(data[name][ch_name], axis=0)
                    elif pulse.average == 'point':
                        data[name][ch_name] = np.mean(data[name][ch_name])
                    elif 'point_segment' in pulse.average:
                        segments = int(pulse.average.split(':')[1])
                        pts = data[name][ch_name].shape[1]

                        segments_idx = [int(round(pts * idx / segments))
                                        for idx in np.arange(segments + 1)]
                        pulse_traces = np.zeros(segments)
                        for k in range(segments):
                            pulse_traces[k] = np.mean(data[name][ch_name][:, segments_idx[k]:segments_idx[k + 1]])

                        data[name][ch_name] = pulse_traces
                    else:
                        raise SyntaxError(f'average mode {pulse.average} not configured')

                elif acquisition_average_mode == 'trace':
                    data[name][ch_name] = ch_data[sample_range[0]:sample_range[1]]
                    # Further average the pulse data
                    if pulse.average == 'point':
                        data[name][ch_name] = np.mean(data[name][ch_name])

        return data

    def add_acquisition_controller(self, acquisition_controller_name,
                                   cls_name=None):
        """
        Adds an acquisition controller to the available controllers.
        If another acquisition controller exists of the same class, it will
        be overwritten.
        Args:
            acquisition_controller_name: instrument name of controller.
                Must be on same server as interface and Keysight digitizer
            cls_name: Optional name of class, which is used as controller key.
                If no cls_name is provided, it is found from the instrument
                class name

        Returns:
            None
        """
        acquisition_controller = self.find_instrument(
            acquisition_controller_name)
        if cls_name is None:
            cls_name = acquisition_controller.__class__.__name__
        # Remove _Controller from cls_name

        cls_name = cls_name.replace('_Controller', '')

        self.acquisition_controllers[cls_name] = acquisition_controller

    def initialize_driver(self):
        """
            Puts driver into a known initial state. Further configuration will
            be done in the configure_driver and get_additional_pulses
            functions.
        """
        for k in range(self.instrument.n_channels):
            self.instrument.parameters['impedance_{}'.format(k)].set(1) # 50 Ohm impedance
            self.instrument.parameters['coupling_{}'.format(k)].set(0)  # DC Coupled
            self.instrument.parameters['full_scale_{}'.format(k)].set(3.0)  # 3.0 Volts

    def get_additional_pulses(self, **kwargs):
        if not self.pulse_sequence.get_pulses(acquire=True):
            # No pulses need to be acquired
            return []
        else:
            # Add a single trigger pulse when starting acquisition
            t_start = min(pulse.t_start for pulse in
                          self.pulse_sequence.get_pulses(acquire=True))
            if (self.input_pulse_sequence.get_pulses(trigger=True, t_start=t_start)):
                logger.warning('Trigger manually defined for M3300A digitizer. '
                               'This is inadvisable as this could limit the responsiveness'
                               ' of the machine.')
                return []
            acquisition_pulse = \
                TriggerPulse(t_start=t_start, duration=15e-6,
                             connection_requirements={
                                 'input_instrument': self.instrument_name(),
                                 'trigger': True
                             })
            return [acquisition_pulse]

    def setup(self, **kwargs):
        controller = self._acquisition_controller
        self.samples(kwargs.pop('samples', 1))

        # Find all unique pulse_connections to choose which channels to acquire on
        channel_selection = [int(ch_name[-1]) for ch_name in self.acquisition_channels()]
        self.channel_selection(sorted(channel_selection))
        controller.channel_selection(self.channel_selection())

        # Check what averaging mode is needed by each pulse
        if any(self.input_pulse_sequence.get_pulses(average='none')):
            controller.average_mode('none')
        else:
            controller.average_mode('trace')

        if controller() == 'Triggered':
            if self.input_pulse_sequence.get_pulses(name='Bayes'):
                bayesian_pulse = self.input_pulse_sequence.get_pulses(name='Bayes')[0]
                for ch in range(8):
                    self.instrument.parameters[f'DAQ_trigger_delay_{ch}'].set(int(bayesian_pulse.t_start*2*self.sample_rate()))
            # Get trigger connection to determine how to trigger the controller
            trigger_pulse = self.input_pulse_sequence.get_pulses(trigger=True)[0]
            trigger_connection = trigger_pulse.connection
            self.trigger_threshold(trigger_pulse.get_voltage(trigger_pulse.t_start) / 2)
            self.trigger_edge('rising')

            t_0 = min(pulse.t_start for pulse in
                      self.pulse_sequence.get_pulses(acquire=True))
            t_f = max(pulse.t_stop for pulse in
                  self.pulse_sequence.get_pulses(acquire=True))
            acquisition_window = t_f - t_0

            duration = self.pulse_sequence.duration

            controller.sample_rate(int(round((self.sample_rate()))))
            controller.traces_per_acquisition(int(round(self.samples())))

            controller.trigger_channel(trigger_connection.input['channel'].id)
            controller.trigger_threshold(self.trigger_threshold())
            # Map the string value of trigger edge to a device integer
            controller.trigger_edge(self.trigger_edge())

            # Capture maximum number of samples on all channels
            controller.samples_per_record(int(round(acquisition_window * self.sample_rate())))


            #TODO : This is all low-level, should figure out a way to shift this
            #TODO : to the acquisition controller.
            max_timeout = np.iinfo(np.uint16).max
            # Separate reads to ensure the total read can be contained within a
            # single timeout. Note a 20% overhead is assumed. At the driver level
            # timeout is measured in ms.
            if int(max_timeout) < int(round(duration * self.samples()*1.2)*1e3):
                samples_per_read = max((max_timeout// int(1e3 * duration) * 100) // 120, 1)
            else:
                samples_per_read = self.samples()
            # read_timeout = duration * samples_per_read * 10
            read_timeout = 64.0
            logger.debug(f'Read timeout is set to {read_timeout:.3f}s.')
            controller.samples_per_read(samples_per_read)
            controller.read_timeout(read_timeout)

    def start(self):
        self._acquisition_controller.pre_start_capture()
        self._acquisition_controller.start()

    def stop(self):
        # Stop all DAQs
        self.instrument.daq_stop_multiple((1 << 8) - 1)


