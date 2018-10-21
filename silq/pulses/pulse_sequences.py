from functools import partial
import numpy as np
import logging
from collections import Iterable
from .pulse_modules import PulseSequence
from .pulse_types import DCPulse, SinePulse, FrequencyRampPulse, Pulse
from copy import deepcopy


logger = logging.getLogger(__name__)


class PulseSequenceGenerator(PulseSequence):
    """Base class for a `PulseSequence` that is generated from settings.
    """
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
    """`PulseSequenceGenerator` for electron spin resonance (ESR).

    This pulse sequence can handle many of the basic pulse sequences involving
    ESR. The pulse sequence is generated from its pulse settings attributes.

    In general the pulse sequence is as follows:

    1. Perform any pre_pulses defined in ``ESRPulseSequence.pre_pulses``.
    2. Perform stage pulse ``ESRPulseSequence.ESR['stage_pulse']``.
       By default, this is the ``plunge`` pulse.
    3. Perform ESR pulse within plunge pulse, the delay from start of plunge
       pulse is defined in ``ESRPulseSequence.ESR['pulse_delay']``.
    4. Perform read pulse ``ESRPulseSequence.ESR['read_pulse']``.
    5. Repeat steps 2 and 3 for each ESR pulse in
       ``ESRPulseSequence.ESR['ESR_pulses']``, which by default contains single
       pulse ``ESRPulseSequence.ESR['ESR_pulse']``.
    6. Perform empty-plunge-read sequence (EPR), but only if
       ``ESRPulseSequence.EPR['enabled']`` is True.
       EPR pulses are defined in ``ESRPulseSequence.EPR['pulses']``.
    7. Perform any post_pulses defined in ``ESRPulseSequence.post_pulses``.

    Parameters:
        ESR (dict): Pulse settings for the ESR part of the pulse sequence.
            Contains the following items:

            * ``stage_pulse`` (Pulse): Stage pulse in which to perform ESR
              (e.g. plunge). Default is 'plunge `DCPulse`.
            * ``ESR_pulse`` (Pulse): Default ESR pulse to use.
              Default is 'ESR' ``SinePulse``.
            * ``ESR_pulses`` (List[Union[str, Pulse]]): List of ESR pulses to
              use. Can be strings, in which case the string should be an item in
              ``ESR`` whose value is a `Pulse`.
            * ``pulse_delay`` (float): ESR pulse delay after beginning of stage
              pulse. Default is 5 ms.
            * ``read_pulse`` (Pulse): Pulse after stage pulse for readout and
              initialization of electron. Default is 'read_initialize`
              `DCPulse`.

        EPR (dict): Pulse settings for the empty-plunge-read (EPR) part of the
            pulse sequence. This part is optional, and is used for non-ESR
            contast, and to measure dark counts and hence ESR contrast.
            Contains the following items:

            * ``enabled`` (bool): Enable EPR sequence.
            * ``pulses`` (List[Pulse]): List of pulses for EPR sequence.
              Default is ``empty``, ``plunge``, ``read_long`` `DCPulse`.

        pre_pulses (List[Pulse]): Pulses before main pulse sequence.
            Empty by default.
        post_pulses (List[Pulse]): Pulses after main pulse sequence.
            Empty by default.
        pulse_settings (dict): Dict containing all pulse settings.
        **kwargs: Additional kwargs to `PulseSequence`.

    Examples:
        The following code measures two ESR frequencies and performs an EPR
        from which the contrast can be determined for each ESR frequency:

        >>> ESR_pulse_sequence = ESRPulseSequence()
        >>> ESR_pulse_sequence.ESR['pulse_delay'] = 5e-3
        >>> ESR_pulse_sequence.ESR['stage_pulse'] = DCPulse['plunge']
        >>> ESR_pulse_sequence.ESR['ESR_pulse'] = FrequencyRampPulse('ESR_adiabatic')
        >>> ESR_pulse_sequence.ESR_frequencies = [39e9, 39.1e9]
        >>> ESR_pulse_sequence.EPR['enabled'] = True
        >>> ESR_pulse_sequence.pulse_sequence.generate()

        The total pulse sequence is plunge-read-plunge-read-empty-plunge-read
        with an ESR pulse in the first two plunge pulses, 5 ms after the start
        of the plunge pulse. The ESR pulses have different frequencies.

    Notes:
        For given pulse settings, `ESRPulseSequence.generate` will recreate the
        pulse sequence from settings.
"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pulse_settings['pre_pulses'] = self.pre_pulses = []

        self.pulse_settings['ESR'] = self.ESR = {
            'ESR_pulse': SinePulse('ESR'),
            'stage_pulse': DCPulse('plunge'),
            'read_pulse': DCPulse('read_initialize', acquire=True),
            'pre_delay': 5e-3,
            'inter_delay': 5e-3,
            'post_delay': 5e-3,
            'ESR_pulses': ['ESR_pulse']}

        self.pulse_settings['EPR'] = self.EPR = {
            'enabled': True,
            'pulses':[
            DCPulse('empty', acquire=True),
            DCPulse('plunge', acquire=True),
            DCPulse('read_long', acquire=True)]}

        self.pulse_settings['post_pulses'] = self.post_pulses = [DCPulse('final')]

        # Primary ESR pulses, the first ESR pulse in each plunge.
        # Used for assigning names during analysis
        self.primary_ESR_pulses = []

    @property
    def ESR_frequencies(self):
        ESR_frequencies = []
        for pulse in self.ESR['ESR_pulses']:
            if isinstance(pulse, Pulse):
                ESR_frequencies.append(pulse.frequency)
            elif isinstance(pulse, str):
                ESR_frequencies.append(self.ESR[pulse].frequency)
            elif isinstance(pulse, list):
                # Pulse is a list containing other pulses
                # These pulses will be joined in a single plunge
                ESR_subfrequencies = []
                for subpulse in pulse:
                    if isinstance(subpulse, Pulse):
                        ESR_subfrequencies.append(subpulse.frequency)
                    elif isinstance(subpulse, str):
                        ESR_subfrequencies.append(self.ESR[subpulse].frequency)
                    else:
                        raise RuntimeError('ESR subpulse must be a pulse or'
                                           f'a string: {repr(subpulse)}')
                ESR_frequencies.append(ESR_subfrequencies)
            else:
                raise RuntimeError('ESR pulse must be Pulse, str, or list '
                                   f'of pulses: {pulse}')
        return ESR_frequencies

    def add_ESR_pulses(self, ESR_frequencies=None):
        """Add ESR pulses to the pulse sequence

        Args:
            ESR_frequencies: List of ESR frequencies. If provided, the pulses in
                ESR['ESR_pulses'] will be reset to  copies of ESR['ESR_pulse']
                with the provided frequencies

        Note:
              Each element in ESR_frequencies can also be a list of multiple
              frequencies, in which case multiple pulses with the provided
              subfrequencies will be used.
        """
        # Manually set ESR frequencies if not explicitly provided, and a pulse
        # sequence has not yet been generated or the ``ESR['ESR_pulse']`` has
        # been modified
        if ESR_frequencies is None and \
                (self._latest_pulse_settings is None or
                 self.ESR['ESR_pulse'] != self._latest_pulse_settings['ESR']['ESR_pulse']):
            # Generate ESR frequencies via property
            ESR_frequencies = self.ESR_frequencies

        if ESR_frequencies is None and \
                (self._latest_pulse_settings is None or
                 self.ESR['ESR_pulse'] != self._latest_pulse_settings['ESR']['ESR_pulse']):
            # Generate ESR frequencies via property
            ESR_frequencies = self.ESR_frequencies

        if ESR_frequencies is not None:
            logger.warning("Resetting all ESR pulses to default ESR['ESR_pulse']")
            self.ESR['ESR_pulses'] = []
            for ESR_frequency in ESR_frequencies:
                if isinstance(ESR_frequency, (float, int)):
                    ESR_pulse = deepcopy(self.ESR['ESR_pulse'])
                    ESR_pulse.frequency = ESR_frequency
                elif isinstance(ESR_frequency, list):
                    ESR_pulse = []
                    for ESR_subfrequency in ESR_frequency:
                        ESR_subpulse = deepcopy(self.ESR['ESR_pulse'])
                        ESR_subpulse.frequency = ESR_subfrequency
                        ESR_pulse.append(ESR_subpulse)
                else:
                    raise RuntimeError('Each ESR frequency must be a number or a'
                                       f' list of numbers. {ESR_frequencies}')
                self.ESR['ESR_pulses'].append(ESR_pulse)

        # Convert any pulse strings to pulses if necessary
        for k, pulse in enumerate(self.ESR['ESR_pulses']):
            if isinstance(pulse, str):
                pulse_copy = deepcopy(self.ESR[pulse])
                self.ESR['ESR_pulses'][k] = pulse_copy
            elif isinstance(pulse, list):
                # Pulse is a list containing other pulses
                # These pulses will be joined in a single plunge
                for kk, subpulse in enumerate(pulse):
                    if isinstance(subpulse, str):
                        subpulse_copy = deepcopy(self.ESR[subpulse])
                        self.ESR['ESR_pulses'][k][kk] = subpulse_copy

        self.primary_ESR_pulses = []  # Clear primary ESR pulses (first in each plunge)
        # Add pulses to pulse sequence
        for single_plunge_ESR_pulses in self.ESR['ESR_pulses']:
            # Each element should be the ESR pulses to apply within a single
            # plunge, between elements there is a read

            if not isinstance(single_plunge_ESR_pulses, list):
                # Single ESR pulse provided, turn into list
                single_plunge_ESR_pulses = [single_plunge_ESR_pulses]

            plunge_pulse, = self.add(self.ESR['stage_pulse'])
            t_connect = partial(plunge_pulse['t_start'].connect,
                                offset=self.ESR['pre_delay'])

            for k, ESR_subpulse in enumerate(single_plunge_ESR_pulses):
                # Add a plunge and read pulse for each frequency
                ESR_pulse, = self.add(ESR_subpulse)
                t_connect(ESR_pulse['t_start'])
                t_connect = partial(ESR_pulse['t_stop'].connect,
                                    offset=self.ESR['inter_delay'])
                if not k:
                    self.primary_ESR_pulses.append(ESR_pulse)

            ESR_pulse['t_stop'].connect(plunge_pulse['t_stop'],
                                        offset=self.ESR['post_delay'])
            self.add(self.ESR['read_pulse'])

    def generate(self, ESR_frequencies=None):
        self.clear()
        self.add(*self.pulse_settings['pre_pulses'])

        # Update self.ESR['ESR_pulses']. Converts any pulses that are strings to
        # actual pulses, and sets correct frequencies
        self.add_ESR_pulses(ESR_frequencies=ESR_frequencies)

        if self.EPR['enabled']:
            self._EPR_pulses = self.add(*self.EPR['pulses'])

        self._ESR_pulses = self.add(*self.post_pulses)

        self._latest_pulse_settings = deepcopy(self.pulse_settings)


class T2ElectronPulseSequence(PulseSequenceGenerator):
    """`PulseSequenceGenerator` for electron coherence (T2) measurements.

    This pulse sequence can handle measurements on the electron coherence time,
    including adding refocusing pulses

    In general the pulse sequence is as follows:

    1. Perform any pre_pulses defined in ``T2ElectronPulseSequence.pre_pulses``.
    2. Perform stage pulse ``T2ElectronPulseSequence.ESR['stage_pulse']``.
       By default, this is the ``plunge`` pulse.
    3. Perform initial ESR pulse
       ``T2ElectronPulseSequence.ESR['ESR_initial_pulse']`` within plunge pulse,
       the delay until start of the ESR pulse is defined in
       ``T2ElectronPulseSequence.ESR['pre_delay']``.
    4. For ``T2ElectronPulseSequence.ESR['num_refocusing_pulses']`` times, wait
       for ``T2ElectronPulseSequence.ESR['inter_delay']` and apply
       ``T2ElectronPulseSequence.ESR['ESR_refocusing_pulse]``.
    5. Wait ``T2ElectronPulseSequence.ESR['ESR_refocusing_pulse']`` and apply
       ``T2ElectronPulseSequence.ESR['ESR_final_pulse']``.
    6. Wait ``T2ElectronPulseSequence.ESR['post_delay']``, then stop stage pulse
       and Perform read pulse ``T2ElectronPulseSequence.ESR['read_pulse']``.
    7. Perform empty-plunge-read sequence (EPR), but only if
       ``T2ElectronPulseSequence.EPR['enabled']`` is True.
       EPR pulses are defined in ``T2ElectronPulseSequence.EPR['pulses']``.
    8. Perform any post_pulses defined in ``ESRPulseSequence.post_pulses``.

    Parameters:
        ESR (dict): Pulse settings for the ESR part of the pulse sequence.
            Contains the following items:

            :stage_pulse (Pulse): Stage pulse in which to perform ESR
              (e.g. plunge). Default is 'plunge `DCPulse`.
            :ESR_initial_pulse (Pulse): Initial ESR pulse to apply within
              stage pulse. Default is ``ESR_piHalf`` `SinePulse`.
              Ignored if set to ``None``
            :ESR_refocusing_pulse (Pulse): Refocusing ESR pulses between
              initial and final ESR pulse. Zero refocusing pulses measures
              T2star, one refocusing pulse measures T2Echo.
              Default is ``ESR_pi`` `SinePulse`.
            :ESR_final_pulse (Pulse): Final ESR pulse within stage pulse.
              Default is ``ESR_piHalf`` `SinePulse`.
              Ignored if set to ``None``.
            :num_refocusing_pulses (int): Number of refocusing pulses
              ``T2ElectronPulseSequence.ESR['ESR_refocusing_pulse']`` to apply.
            :pre_delay (float): Delay after stage pulse before first ESR
              pulse.
            :inter_delay (float): Delay between successive ESR pulses.
            :post_delay (float): Delay after final ESR pulse.
            :read_pulse (Pulse): Pulse after stage pulse for readout and
              initialization of electron. Default is 'read_initialize`
              `DCPulse`.

        EPR (dict): Pulse settings for the empty-plunge-read (EPR) part of the
            pulse sequence. This part is optional, and is used for non-ESR
            contast, and to measure dark counts and hence ESR contrast.
            Contains the following items:

            :enabled: (bool): Enable EPR sequence.
            :pulses: (List[Pulse]): List of pulses for EPR sequence.
              Default is ``empty``, ``plunge``, ``read_long`` `DCPulse`.

        pre_pulses (List[Pulse]): Pulses before main pulse sequence.
            Empty by default.
        post_pulses (List[Pulse]): Pulses after main pulse sequence.
            Empty by default.
        pulse_settings (dict): Dict containing all pulse settings.
        **kwargs: Additional kwargs to `PulseSequence`.

    Notes:
        For given pulse settings, `T2ElectronPulseSequence.generate` will
        recreate the pulse sequence from settings.
"""
    def __init__(self, **kwargs):
        super().__init__(pulses=[], **kwargs)

        self.pulse_settings['ESR'] = self.ESR = {
            'stage_pulse': DCPulse('plunge'),
            'ESR_initial_pulse': SinePulse('ESR_PiHalf'),
            'ESR_refocusing_pulse': SinePulse('ESR_Pi'),
            'ESR_final_pulse': SinePulse('ESR_PiHalf'),
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
        self.pulse_settings['post_pulses'] = self.post_pulses = []

    def add_ESR_pulses(self):
        # Add stage pulse, duration will be specified later
        stage_pulse, = self.add(self.ESR['stage_pulse'])

        t = stage_pulse.t_start + self.ESR['pre_delay']

        # Add initial pulse (evolve to state where T2 effects can be observed)
        if self.ESR['ESR_initial_pulse'] is not None:
            ESR_initial_pulse, = self.add(self.ESR['ESR_initial_pulse'])
            ESR_initial_pulse.t_start = t
            t += ESR_initial_pulse.duration

        for k in range(self.ESR['num_refocusing_pulses']):
            t += self.ESR['inter_delay']
            ESR_refocusing_pulse = self.add(self.ESR['ESR_refocusing_pulse'])
            ESR_refocusing_pulse.t_start = t
            t += ESR_refocusing_pulse.duration

        t += self.ESR['inter_delay']
        if self.ESR['ESR_final_pulse'] is not None:
            ESR_final_pulse, = self.add(self.ESR['ESR_final_pulse'])
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


class NMRPulseSequence(PulseSequenceGenerator):
    """`PulseSequenceGenerator` for nuclear magnetic resonance (NMR).

    This pulse sequence can handle many of the basic pulse sequences involving
    NMR. The pulse sequence is generated from its pulse settings attributes.

    In general, the pulse sequence is as follows:

    1. Perform any pre_pulses defined in ``NMRPulseSequence.pre_pulses``.
    2. Perform NMR sequence

       1. Perform stage pulse ``NMRPulseSequence.NMR['stage_pulse']``.
          Default is 'empty' `DCPulse`.
       2. Perform NMR pulses within the stage pulse. The NMR pulses defined
          in ``NMRPulseSequence.NMR['NMR_pulses']`` are applied successively.
          The delay after start of the stage pulse is
          ``NMRPulseSequence.NMR['pre_delay']``, delays between NMR pulses is
          ``NMRPulseSequence.NMR['inter_delay']``, and the delay after the final
          NMR pulse is ``NMRPulseSequence.NMR['post_delay']``.

    3. Perform ESR sequence

       1. Perform stage pulse ``NMRPulseSequence.ESR['stage_pulse']``.
          Default is 'plunge' `DCPulse`.
       2. Perform ESR pulse within stage pulse for first pulse in
          ``NMRPulseSequence.ESR['ESR_pulses']``.
       3. Perform ``NMRPulseSequence.ESR['read_pulse']``, and acquire trace.
       4. Repeat steps 1 - 3 for each ESR pulse. The different ESR pulses
          usually correspond to different ESR frequencies (see
          `NMRPulseSequence`.ESR_frequencies).
       5. Repeat steps 1 - 4 for ``NMRPulseSequence.ESR['shots_per_frequency']``
          This effectively interleaves the ESR pulses, which counters effects of
          the nucleus flipping within an acquisition.

    By measuring the average up proportion for each ESR frequency, a switching
    between high and low up proportion indicates a flipping of the nucleus

    Parameters:
        NMR (dict): Pulse settings for the NMR part of the pulse sequence.
            Contains the following items:

            * ``stage_pulse`` (Pulse): Stage pulse in which to perform NMR
              (e.g. plunge). Default is 'empty' `DCPulse`. Duration of stage
              pulse is adapted to NMR pulses and delays.
            * ``NMR_pulse`` (Pulse): Default NMR pulse to use.
              By default 'NMR' `SinePulse`.
            * ``NMR_pulses`` (List[Union[str, Pulse]]): List of NMR pulses to
              successively apply. Can be strings, in which case the string
              should be an item in ``NMR`` whose value is a `Pulse`. Default is
              single element ``NMRPulseSequence.NMR['NMR_pulse']``.
            * ``pre_delay`` (float): Delay after start of ``stage`` pulse,
              until first NMR pulse.
            * ``inter_delay`` (float): Delay between successive NMR pulses.
            * ``post_delay`` (float): Delay after final NMR pulse until stage
              pulse end.

        ESR (dict): Pulse settings for the ESR part of the pulse sequence.
            Contains the following items:

            * ``stage_pulse`` (Pulse): Stage pulse in which to perform ESR
              (e.g. plunge). Default is 'plunge `DCPulse`.
            * ``ESR_pulse`` (Pulse): Default ESR pulse to use.
              Default is 'ESR' ``SinePulse``.
            * ``ESR_pulses`` (List[Union[str, Pulse]]): List of ESR pulses to
              use. Can be strings, in which case the string should be an item in
              ``ESR`` whose value is a `Pulse`.
            * ``pulse_delay`` (float): ESR pulse delay after beginning of stage
              pulse. Default is 5 ms.
            * ``read_pulse`` (Pulse): Pulse after stage pulse for readout and
              initialization of electron. Default is 'read_initialize`
              `DCPulse`.

        EPR (dict): Pulse settings for the empty-plunge-read (EPR) part of the
            pulse sequence. This part is optional, and is used for non-ESR
            contast, and to measure dark counts and hence ESR contrast.
            Contains the following items:

            * ``enabled`` (bool): Enable EPR sequence.
            * ``pulses`` (List[Pulse]): List of pulses for EPR sequence.
              Default is ``empty``, ``plunge``, ``read_long`` `DCPulse`.

        pre_pulses (List[Pulse]): Pulses before main pulse sequence.
            Empty by default.
        post_pulses (List[Pulse]): Pulses after main pulse sequence.
            Empty by default.
        pulse_settings (dict): Dict containing all pulse settings.
        **kwargs: Additional kwargs to `PulseSequence`.

    See Also:
        NMRParameter

    Notes:
        For given pulse settings, `NMRPulseSequence.generate` will recreate the
        pulse sequence from settings.
    """
    def __init__(self, pulses=[], **kwargs):
        super().__init__(pulses=pulses, **kwargs)
        self.pulse_settings['NMR'] = self.NMR = {
            'stage_pulse': DCPulse('empty'),
            'NMR_pulse': SinePulse('NMR'),
            'NMR_pulses': ['NMR_pulse'],
            'pre_delay': 5e-3,
            'inter_delay': 1e-3,
            'post_delay': 2e-3}
        self.pulse_settings['ESR'] = self.ESR = {
            'ESR_pulse': FrequencyRampPulse('adiabatic_ESR'),
            'ESR_pulses': ['ESR_pulse'],
            'stage_pulse': DCPulse('plunge'),
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
        for pulse in self.NMR['NMR_pulses']:
            if isinstance(pulse, str):
                # Pulse is a reference to some pulse in self.NMR
                pulse = self.NMR[pulse]
            NMR_pulse, = pulse_sequence.add(pulse)

            if not NMR_pulses:
                NMR_stage_pulse['t_start'].connect(NMR_pulse['t_start'],
                                                   offset=self.NMR['pre_delay'])
            else:
                NMR_pulses[-1]['t_stop'].connect(NMR_pulse['t_start'],
                                                 offset=self.NMR['inter_delay'])
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
            for ESR_pulses in self.ESR['ESR_pulses']:
                # Add a plunge and read pulse for each frequency

                if not isinstance(ESR_pulses, list):
                    # Treat frequency as list, as one could add multiple ESR
                    # pulses
                    ESR_pulses = [ESR_pulses]

                plunge_pulse, = pulse_sequence.add(self.ESR['stage_pulse'])
                for k, ESR_pulse in enumerate(ESR_pulses):

                    if isinstance(ESR_pulse, str):
                        # Pulse is a reference to some pulse in self.ESR
                        ESR_pulse = self.ESR[ESR_pulse]

                    ESR_pulse, = pulse_sequence.add(ESR_pulse)

                    # Delay also depends on any previous ESR pulses
                    delay = self.ESR['pre_delay'] + k * self.ESR['inter_delay']
                    plunge_pulse['t_start'].connect(ESR_pulse['t_start'],
                                                    offset=delay)
                ESR_pulse['t_stop'].connect(plunge_pulse['t_stop'],
                                            offset=self.ESR['post_delay'])
                pulse_sequence.add(self.ESR['read_pulse'])

    def generate(self):
        """Updates the pulse sequence"""

        # Initialize pulse sequence
        self.clear()

        self.add(*self.pulse_settings['pre_pulses'])

        self.add_NMR_pulses()

        self.add_ESR_pulses()

        self.add(*self.pulse_settings['post_pulses'])

        # Create copy of current pulse settings for comparison later
        self._latest_pulse_settings = deepcopy(self.pulse_settings)


class FlipFlopPulseSequence(PulseSequenceGenerator):
    """`PulseSequenceGenerator` for hitting the flip-flop transition

    The flip-flop transition is the one where both the electron and nucleus flip
    in opposite direction, thus keeping the total spin constant.

    This pulse sequence is mainly used to flip the nucleus to a certain state
    without having to perform NMR or even having to measure the electron.

    The flip-flop transitions between a nuclear spin state S1 and (S1+1) is:

    f_ESR(S1) + A/2 + gamma_n * B_0,

    where f_ESR(S1) is the ESR frequency for nuclear state S1, A is the
    hyperfine, gamma_n is the nuclear Zeeman, and B_0 is the static magnetic
    field. The transition will flip (electron down, nucleus S+1) to
    (electron up, nucleus S) and vice versa.

    Parameters:
        ESR (dict): Pulse settings for the ESR part of the pulse sequence.
            Contains the following items:

            * ``frequency`` (float): ESR frequency below the flip-flop
              transition (Hz).
            * ``hyperfine`` (float): Hyperfine interaction (Hz).
            * ``nuclear_zeeman`` (float): Nuclear zeeman strength (gamma_n*B_0)
            * ``stage_pulse`` (Pulse): Stage pulse in which to perform ESR
              (e.g. plunge). Default is `DCPulse`('plunge').
            * ``pre_flip_ESR_pulse`` (Pulse): ESR pulse to use before the
              flip-flop pulse to pre-flip the electron to spin-up,
              which allows the nucleus to be flipped to a higher state.
              Default is `SinePulse`('ESR').
            * ``flip_flop_pulse`` (Pulse): Flip-flop ESR pulse, whose
              frequency will be set to A/2 + gamma_n*B_0 higher than the
              ``frequency`` setting. Default pulse is `SinePulse`('ESR')
            * ``pre_flip`` (bool): Whether to pre-flip the electron, to
              transition to a higher nuclear state. Default is False.
            * ``pre_delay`` (float): Delay between start of stage pulse and
              first pulse (``pre_flip_ESR_pulse`` or ``flip_flop_pulse``).
            * ``inter_delay`` (float): Delay between ``pre_flip_ESR_pulse`` and
              ``flip_flop_pulse``. Ignored if pre_flip is False
            * ``post_delay`` (float): Delay after last frequency pulse and end
              of stage pulse.

        pre_pulses (List[Pulse]): Pulses before main pulse sequence.
            Empty by default.
        post_pulses (List[Pulse]): Pulses after main pulse sequence.
            Empty by default.
        pulse_settings (dict): Dict containing all pulse settings.
        **kwargs: Additional kwargs to `PulseSequence`.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pulse_settings['ESR'] = self.ESR = {
            'frequencies': [28e9, 28e9],
            'hyperfine': None,
            'nuclear_zeeman': -5.5e6,
            'stage_pulse': DCPulse('plunge', acquire=True),
            'pre_flip_ESR_pulse': SinePulse('ESR'),
            'flip_flop_pulse': SinePulse('ESR'),
            'pre_flip': False,
            'pre_delay': 5e-3,
            'inter_delay': 1e-3,
            'post_delay': 5e-3}

        self.pulse_settings['pre_pulses'] = self.pre_pulses = []
        self.pulse_settings['post_pulses'] = self.post_pulses = [DCPulse('read')]

        self.generate()

    def add_ESR_pulses(self):
        stage_pulse, = self.add(self.ESR['stage_pulse'])
        ESR_t_start = partial(stage_pulse['t_start'].connect,
                              offset=self.ESR['pre_delay'])

        if self.ESR['pre_flip']:
            # First add the pre-flip the ESR pulses (start with excited electron)
            for ESR_frequency in self.ESR['frequencies']:
                pre_flip_ESR_pulse, = self.add(self.ESR['pre_flip_ESR_pulse'])
                pre_flip_ESR_pulse.frequency = ESR_frequency
                ESR_t_start(pre_flip_ESR_pulse['t_start'])

                # Update t_start of next ESR pulse
                ESR_t_start = partial(pre_flip_ESR_pulse['t_stop'].connect,
                                      offset=self.ESR['inter_delay'])

        flip_flop_ESR_pulse, = self.add(self.ESR['flip_flop_pulse'])
        ESR_t_start(flip_flop_ESR_pulse['t_start'])

        # Calculate flip-flop frequency
        ESR_max_frequency = np.max(self.ESR['frequencies'])
        hyperfine = self.ESR['hyperfine']
        if hyperfine is None:
            # Choose difference between two ESR frequencies
            hyperfine = float(np.abs(np.diff(self.ESR['frequencies'])))

        flip_flop_ESR_pulse.frequency = (ESR_max_frequency
                                         - hyperfine / 2
                                         - self.ESR['nuclear_zeeman'])
        flip_flop_ESR_pulse['t_stop'].connect(stage_pulse['t_stop'],
                                              offset=self.ESR['post_delay'])

    def generate(self):
        """Updates the pulse sequence"""
        self.clear()

        self.add(*self.pulse_settings['pre_pulses'])

        self.add_ESR_pulses()

        self.add(*self.pulse_settings['post_pulses'])