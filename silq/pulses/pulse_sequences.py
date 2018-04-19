from collections import Iterable
from .pulse_modules import PulseSequence, PulseMatch
from .pulse_types import DCPulse, SinePulse, FrequencyRampPulse, Pulse
from copy import deepcopy

class PulseSequenceGenerator(PulseSequence):
    def __init__(self, pulses=[], **kwargs):
        super().__init__(pulses=pulses, **kwargs)
        self.pulse_settings = {}
        self._latest_pulse_settings = None

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

        # Update value in pulse settings if it exists
        try:
            if key in self.pulse_settings:
                self.pulse_settings[key] = value
        except AttributeError:
            pass

    def generate(self):
        raise NotImplementedError('Needs to be implemented in subclass')

    def up_to_date(self):
        # Compare to attributes when pulse sequence was created
        return self.pulse_settings == self._latest_pulse_settings


class ESRPulseSequence(PulseSequenceGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pulse_settings['pre_pulses'] = self.pre_pulses = []

        self.pulse_settings['ESR'] = self.ESR = {
            'pulse': FrequencyRampPulse('ESR'),
            'plunge_pulse': DCPulse('plunge'),
            'read_pulse': DCPulse('read_initialize', acquire=True),
            'pre_delay': 5e-3,
            'post_delay': 5e-3,
            'pulses': ['pulse']}

        self.pulse_settings['EPR'] = self.EPR = {
            'enabled': True,
            'pulses':[
            DCPulse('empty', acquire=True),
            DCPulse('plunge', acquire=True),
            DCPulse('read_long', acquire=True)]}

        self.pulse_settings['post_pulses'] = self.post_pulses = [DCPulse('final')]

    def add_ESR_pulses(self, ESR_frequencies=None):
        if ESR_frequencies is None:
            ESR_frequencies = [pulse.frequency if isinstance(pulse, Pulse)
                               else self.ESR[pulse].frequency
                               for pulse in self.ESR['pulses']]

        if self._latest_pulse_settings is None or \
                (self.ESR['pulse'] != self._latest_pulse_settings['ESR']['pulse']) \
                or (len(ESR_frequencies) != len(self.ESR['pulses'])):
            # Resetting ESR pulses
            self.ESR['pulses'] = [deepcopy(self.ESR['pulse'])
                                  for _ in range(len(ESR_frequencies))]
        else:
            # Convert any pulse strings to pulses if necessary
            self.ESR['pulses'] = [
                deepcopy(self.ESR[p]) if isinstance(p, str) else p
                for p in self.ESR['pulses']]

        # Make sure all pulses have proper ESR frequency
        for pulse, ESR_frequency in zip(self.ESR['pulses'], ESR_frequencies):
            pulse.frequency = ESR_frequency

        for ESR_pulse in self.ESR['pulses']:
            # Add a plunge and read pulse for each frequency
            plunge_pulse, = self.add(self.ESR['plunge_pulse'])
            ESR_pulse, = self.add(ESR_pulse)
            ESR_pulse.t_start = PulseMatch(plunge_pulse, 't_start',
                                           delay=self.ESR['pre_delay'])
            plunge_pulse.t_stop = PulseMatch(ESR_pulse, 't_stop',
                                             delay=self.ESR['post_delay'])
            self.add(self.ESR['read_pulse'])

    def generate(self, ESR_frequencies=None):
        self.clear()
        self.add(*self.pulse_settings['pre_pulses'])

        # Update self.ESR['pulses']. Converts any pulses that are strings to
        # actual pulses, and sets correct frequencies
        self.add_ESR_pulses(ESR_frequencies=ESR_frequencies)

        if self.EPR['enabled']:
            self.add(*self.EPR['pulses'])

        self.add(*self.post_pulses)

        self._latest_pulse_settings = deepcopy(self.pulse_settings)


class NMRPulseSequence(PulseSequenceGenerator):
    def __init__(self, pulses=[], **kwargs):
        super().__init__(pulses=pulses, **kwargs)
        self.pulse_settings['NMR'] = self.NMR = {
            'stage_pulse': DCPulse('empty'),
            'NMR_pulse': SinePulse('NMR'),
            'pulses': ['NMR_pulse'],
            'pre_delay': 5e-3,
            'inter_delay': 1e-3,
            'post_delay': 2e-3}
        self.pulse_settings['ESR'] = self.ESR = {
            'ESR_pulse': FrequencyRampPulse('adiabatic_ESR'),
            'pulses': ['ESR_pulse'],
            'plunge_pulse': DCPulse('plunge'),
            'read_pulse': DCPulse('read_initialize', acquire=True),
            'pre_delay': 5e-3,
            'post_delay': 5e-3,
            'inter_delay': 1e-3,
            'shots_per_frequency': 25}
        self.pulse_settings['pre_pulses'] = self.pre_pulses = []
        self.pulse_settings['post_pulses'] = self.post_pulses = []

        self.generate()

    def add_NMR_pulses(self, pulse_sequence=None):
        if pulse_sequence is None:
            pulse_sequence = self

        NMR_stage_pulse, = pulse_sequence.add(self.NMR['stage_pulse'])

        NMR_pulses = []
        for pulse in self.NMR['pulses']:
            if isinstance(pulse, str):
                # Pulse is a reference to some pulse in self.NMR
                pulse = self.NMR[pulse]
            NMR_pulse, = pulse_sequence.add(pulse)

            if not NMR_pulses:
                NMR_pulse.t_start = PulseMatch(NMR_stage_pulse, 't_start',
                                               delay=self.NMR['pre_delay'])
            else:
                NMR_pulse.t_start = PulseMatch(NMR_pulses[-1], 't_stop',
                                               delay=self.NMR['inter_delay'])
            NMR_pulses.append(NMR_pulse)

        NMR_stage_pulse.duration = self.NMR['pre_delay']
        NMR_stage_pulse.duration += (len(NMR_pulses) - 1) * self.NMR['inter_delay']
        NMR_stage_pulse.duration += sum(pulse.duration for pulse in NMR_pulses)
        NMR_stage_pulse.duration += self.NMR['post_delay']
        return pulse_sequence

    def add_ESR_pulses(self, pulse_sequence=None):
        if pulse_sequence is None:
            pulse_sequence = self

        for _ in range(self.ESR['shots_per_frequency']):
            for ESR_pulses in self.ESR['pulses']:
                # Add a plunge and read pulse for each frequency

                if not isinstance(ESR_pulses, list):
                    # Treat frequency as list, as one could add multiple ESR
                    # pulses
                    ESR_pulses = [ESR_pulses]

                plunge_pulse, = pulse_sequence.add(self.ESR['plunge_pulse'])
                for k, ESR_pulse in enumerate(ESR_pulses):

                    if isinstance(ESR_pulse, str):
                        # Pulse is a reference to some pulse in self.ESR
                        ESR_pulse = self.ESR[ESR_pulse]

                    ESR_pulse, = pulse_sequence.add(ESR_pulse)

                    # Delay also depends on any previous ESR pulses
                    delay = self.ESR['pre_delay'] + k * self.ESR['inter_delay']
                    ESR_pulse.t_start = PulseMatch(plunge_pulse, 't_start',
                                                   delay=delay)
                plunge_pulse.t_stop = PulseMatch(ESR_pulse, 't_stop',
                                                 delay=self.ESR['post_delay'])
                pulse_sequence.add(self.ESR['read_pulse'])

    def generate(self):
        """
        Updates the pulse sequence
        """

        # Initialize pulse sequence
        self.clear()

        self.add(*self.pulse_settings['pre_pulses'])

        self.add_NMR_pulses()

        self.add_ESR_pulses()

        self.add(*self.pulse_settings['post_pulses'])

        # Create copy of current pulse settings for comparison later
        self._latest_pulse_settings = deepcopy(self.pulse_settings)

class T2ElectronPulseSequence(PulseSequenceGenerator):
    def __init__(self, **kwargs):
        super().__init__(pulses=[], **kwargs)

        self.pulse_settings['ESR'] = self.ESR = {
            'stage_pulse': DCPulse('plunge'),
            'initial_pulse': SinePulse('ESR_PiHalf'),
            'refocusing_pulse': SinePulse('ESR_Pi'),
            'final_pulse': SinePulse('ESR_PiHalf'),
            'read_pulse': DCPulse('read'),

            'num_refocusing_pulses': 0,

            'pre_delay': None,
            'inter_delay': None,
            'post_delay': None,
        }

        self.pulse_settings['EPR'] = self.EPR = {
            'enabled': True,
            'pulses':[
            DCPulse('empty', acquire=True),
            DCPulse('plunge', acquire=True),
            DCPulse('read_long', acquire=True)]}

        self.pulse_settings['pre_pulses'] = self.pre_pulses = []
        self.pulse_settings['post_pulses'] = self.post_pulses = [
            DCPulse('empty'),
            DCPulse('plunge'),
            DCPulse('read_long', acquire=True),
            DCPulse('final')]

    def add_ESR_pulses(self):
        # Add stage pulse, duration will be specified later
        stage_pulse, = self.add(self.ESR['stage_pulse'])

        t = stage_pulse.t_start + self.ESR['pre_delay']

        # Add initial pulse (evolve to state where T2 effects can be observed)
        if self.ESR['initial_pulse'] is not None:
            ESR_initial_pulse, = self.add(self.ESR['initial_pulse'])
            ESR_initial_pulse.t_start = t
            t += ESR_initial_pulse.duration

        for k in range(self.ESR['num_refocusing_pulses']):
            t += self.ESR['inter_delay']
            ESR_refocusing_pulse = self.add(self.ESR['refocusing_pulse'])
            ESR_refocusing_pulse.t_start = t
            t += ESR_refocusing_pulse.duration

        t += self.ESR['inter_delay']
        if self.ESR['final_pulse'] is not None:
            ESR_final_pulse, = self.add(self.ESR['final_pulse'])
            ESR_final_pulse.t_start = t
            t += ESR_final_pulse.duration

        t += self.ESR['post_delay']

        stage_pulse.duration = t - stage_pulse.t_start

        # Add final read pulse
        self.add(self.ESR['read_pulse'])

    def generate(self):
        """
        Updates the pulse sequence
        """

        # Initialize pulse sequence
        self.clear()

        self.add(*self.pulse_settings['pre_pulses'])

        self.add_ESR_pulses()

        if self.EPR['enabled']:
            self.add(*self.EPR['pulses'])

        self.add(*self.pulse_settings['post_pulses'])

        # Create copy of current pulse settings for comparison later
        self._latest_pulse_settings = deepcopy(self.pulse_settings)