from functools import partial
import numpy as np
import logging
from collections import Iterable, Sequence
from pathlib import Path
from copy import deepcopy, copy

from .pulse_modules import PulseSequence
from .pulse_types import DCPulse, SinePulse, FrequencyRampPulse, Pulse
from silq.tools.circuit_tools import convert_circuit, load_circuits
from qcodes import Parameter, Measurement, DataSet
from qcodes.instrument.parameter_node import parameter
from qcodes.utils import validators as vals

from qcodes.config.config import DotDict

logger = logging.getLogger(__name__)


class PulseSequenceGenerator(PulseSequence):
    delegate_attr_dicts = ['parameters', 'parameter_nodes', 'functions',
                           'submodules', 'pulse_settings']
    """Base class for a `PulseSequence` that is generated from settings.
    """
    def __init__(self, pulses=[], **kwargs):
        super().__init__(pulses=pulses, **kwargs)
        self.pulse_settings = {}
        self._latest_pulse_settings = None

        self._meta_attrs.append('pulse_settings')

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

    def _update_latest_pulse_settings(
            self,
            deepcopy_pulse=False,
            deepcopy_pulse_sequence=False,
            **kwargs
    ):
        element = kwargs.get('element', self.pulse_settings)

        if isinstance(element, dict):
            # print(f'copying dict {element}')
            copied_element = {
                key: self._update_latest_pulse_settings(
                    element=val,
                    deepcopy_pulse_sequence=deepcopy_pulse_sequence,
                    deepcopy_pulse=deepcopy_pulse
                ) for key, val in element.items()
            }
            if isinstance(element, DotDict):
                copied_element = DotDict(copied_element)
        elif isinstance(element, (list, tuple, set)):
            # print('copying iterable')
            copied_element = [
                self._update_latest_pulse_settings(
                    element=val,
                    deepcopy_pulse_sequence=deepcopy_pulse_sequence,
                    deepcopy_pulse=deepcopy_pulse
                ) for val in element
            ]
            if isinstance(element, tuple):
                copied_element = tuple(copied_element)
            elif isinstance(element, set):
                copied_element = set(copied_element)
        elif isinstance(element, Pulse):
            # print('copying pulse')
            if deepcopy_pulse:
                copied_element = deepcopy(element)
            else:
                copied_element = element.copy()
        elif isinstance(element, PulseSequence):
            # print('copying pulse sequence')
            if deepcopy_pulse_sequence:
                copied_element = deepcopy(element)
            else:
                copied_element = element.copy()
        else:
            # print('copying other')
            copied_element = copy(element)

        # Update latest pulse settings
        if not kwargs:
            self._latest_pulse_settings = copied_element

        return copied_element


    def up_to_date(self):
        # Compare to attributes when pulse sequence was created
        return self.pulse_settings == self._latest_pulse_settings


class ElectronReadoutPulseSequence(PulseSequenceGenerator):
    """`PulseSequenceGenerator` for a general electron readout

    This pulse sequence can handle nearly all of the basic pulse sequencing
    involving ESR / NMR.
    More specific pulse sequences, such as The `ESRPulseSequenceComposite` and
    `NMRPulseSequenceComposite` are composed by nesting multiple
    `ElectronReadoutPulseSequence`. The `ElectronReadoutPulseSequence` can
    therefore be seen as a building block for more complicated sequences.

    The basic pulse sequence scheme is as follows:

    1. Perform stage pulse ``ElectronReadoutPulseSequence.settings.stage_pulse``.
       By default the stage pulse is a ``plunge`` pulse.
    2. Perform RF pulse(s) within the stage pulse, as defined in the list
       ``ElectronReadoutPulseSequence.settings.RF_pulses``. More on this later.
    3. Perform read pulse, defined in ``ElectronReadoutPulseSequence.settings.read_pulse``.

    This basic scheme can be extended in several ways, the relevant settings are
    grouped together in ``ElectronReadoutPulseSequence.settings``.
    The pulse sequence is generated from these settings via
    ``ElectronReadoutPulseSequence.generate()``. The different settings will now
    be described.

    - RF_pulses: This is the most important setting, as it determines the number
      of readouts and the pulses in each stage.
      Each element in ``RF_pulses`` is applied in a single stage, and followed
      by a readout. An element can be:

      - `Pulse`
      - List of pulses, in which case multiple pulses are applied in a single stage.
        The delays between pulses are specified in the setting ``inter_delay``
      - `PulseSequence`, in which case the entire pulse sequence is considered
        as a single stage. Useful if more complex pulsing is needed in a single
        stage. An actual stage pulse is added if it is not present in the pulse
        sequence.
      - String referring to pulse defined in ``ElectronReadoutPulseSequence.settings``.

      The elements in ``RF_pulses`` can be a combination of the above three types.
    - pre_pulses: Pulses to be applied before first stage pulse
    - post_pulses:Pulses to be applied after last read pulse
    - stage_pulse: Pulse applied during RF pulses.
      Usually a DC pulse to plunge/empty the donor
    - transition_pulse: Optional pulse between stage pulse and read pulse.
      Can be used to transition to an intermediate bias voltage.
    - read_pulse: Pulse applied after stage pulse, usually to read out electron.
    - pre_delay: Delay between beginning of stage pulse and first RF pulse
    - inter_delay: Delay between successive RF pulses. Can be a list in which
        case element k is the delay between RF_pulse k and k+1
    - post_delay: Delay between last RF pulse and end of stage pulse
    - shots_per_frequency: Number of times to repeat the stage pulses.
        Usually done for repeated nuclear readout.
    - min_duration: Minimum stage duration. If stage duration (as calculated
        from pre/inter/post_delay and RF_pulses) is shorter than min_duration,
        the post_delay is extended such that the total stage duration equals
         min_duration.

    Examples:
        The following code performs a stage with ESR_adiabatic pulse,
        followed by a readout

        >>> ESR_pulse_sequence = ElectronReadoutPulseSequence()
        >>> ESR_pulse_sequence.settings.stage_pulse = DCPulse('plunge')
        >>> ESR_pulse_sequence.settings.RF_pulses = [FrequencyRampPulse('ESR_adiabatic')]
        >>> ESR_pulse_sequence.generate()

"""
    pulse_settings = DotDict({
        'RF_pulse': SinePulse('ESR'),
        'stage_pulse': DCPulse('plunge'),
        'transition_pulse': None,
        'read_pulse': DCPulse('read_initialize', acquire=True),  # Can be set to None
        'pre_delay': 5e-3,
        'inter_delay': 5e-3,
        'post_delay': 5e-3,
        'min_duration': None,
        'RF_pulses': ['RF_pulse'],
        'pre_pulses': (),
        'post_pulses': (),
        'shots_per_frequency': 1
    })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create a copy of pulse settings
        self.pulse_settings = self._update_latest_pulse_settings(
            element=ElectronReadoutPulseSequence.pulse_settings
        )

        self.frequencies = Parameter()

    @property
    def settings(self):
        return self.pulse_settings

    @parameter
    def frequencies_get(self, parameter):
        frequencies = []
        for pulse in self.pulse_settings['RF_pulses']:
            if isinstance(pulse, Pulse):
                frequencies.append(pulse.frequency)
            elif isinstance(pulse, str):
                frequencies.append(self.pulse_settings[pulse].frequency)
            elif isinstance(pulse, list):
                # Pulse is a list containing other pulses
                # These pulses will be joined in a single plunge
                subfrequencies = []
                for subpulse in pulse:
                    if isinstance(subpulse, Pulse):
                        subfrequencies.append(subpulse.frequency)
                    elif isinstance(subpulse, str):
                        subfrequencies.append(self.pulse_settings[subpulse].frequency)
                    elif isinstance(subpulse, PulseSequence):
                        subfrequencies.append(None)
                    else:
                        raise RuntimeError(
                            f'RF subpulse must be a pulse or a string: {repr(subpulse)}'
                        )
                frequencies.append(subfrequencies)
            elif isinstance(pulse, PulseSequence):
                frequencies.append(None)
            else:
                raise RuntimeError(
                    f'RF pulse must be Pulse, str, or list of pulses: {pulse}'
                )
        return frequencies

    @parameter
    def frequencies_set(self, parameter, frequencies):
        logger.warning("Resetting all RF pulses to default 'RF_pulse'")
        self.pulse_settings['RF_pulses'] = []
        for frequency in frequencies:
            if isinstance(frequency, (float, int)):
                # Apply a single RF pulse within the stage
                RF_pulses = deepcopy(self.pulse_settings['RF_pulse'])
                RF_pulses.frequency = frequency
            elif isinstance(frequency, list):
                # Apply multiple RF pulses within the stage
                RF_pulses = []
                for subfrequency in frequency:
                    RF_subpulse = deepcopy(self.pulse_settings['RF_pulse'])
                    RF_subpulse.frequency = subfrequency
                    RF_pulses.append(RF_subpulse)
            else:
                raise RuntimeError(
                    f'Each RF frequency must be a number or a list of numbers. {frequencies}'
                )
            self.pulse_settings['RF_pulses'].append(RF_pulses)

        self.generate()

    def convert_RF_pulse_labels_to_pulses(self, RF_pulses=None):
        """Convert any RF pulse strings to the corresponding pulse

        Every RF pulse string should be defined as a pulse in self.pulse_settings
        """
        if RF_pulses is None:
            RF_pulses = self.pulse_settings['RF_pulses']

        converted_RF_pulses = []
        # Convert any pulse strings to pulses if necessary
        for k, RF_pulse in enumerate(RF_pulses):
            if isinstance(RF_pulse, str):  # Convert string to pulse
                pulse_copy = deepcopy(self.pulse_settings[RF_pulse])
                converted_RF_pulses.append(pulse_copy)
            elif isinstance(RF_pulse, Pulse):
                converted_RF_pulses.append(RF_pulse)
            elif isinstance(RF_pulse, PulseSequence):
                converted_RF_pulses.append(RF_pulse)
            elif isinstance(RF_pulse, list):
                # Pulse is a list containing other pulses, to be joined in a single stage
                converted_RF_subpulses = self.convert_RF_pulse_labels_to_pulses(RF_pulse)
                converted_RF_pulses.append(converted_RF_subpulses)
            elif RF_pulse is None:
                converted_RF_pulses.append([])
            else:
                raise RuntimeError(f'Cannot understand RF pulse {repr(RF_pulse)}')

        return converted_RF_pulses

    def _add_RF_pulses_single_stage(self, RF_pulses_single_stage):
        # Each element should be the RF pulses to apply within a single
        # plunge, between elements there is a read
        if not isinstance(RF_pulses_single_stage, list):
            # Single RF pulse provided, turn into list
            RF_pulses_single_stage = [RF_pulses_single_stage]

        RF_pulse = None

        stage_pulse, = self.add(self.pulse_settings['stage_pulse'])
        t_connect = partial(stage_pulse['t_start'].connect,
                            offset=self.pulse_settings['pre_delay'])

        for k, RF_subpulse in enumerate(RF_pulses_single_stage):
            if RF_subpulse is None:
                continue
            elif isinstance(RF_subpulse, Pulse):
                # Add a plunge and read pulse for each frequency
                RF_pulse, = self.add(RF_subpulse, connect=False)
                t_connect(RF_pulse['t_start'])

                if k < len(RF_pulses_single_stage) - 1:
                    # Determine delay between NMR pulses
                    inter_delay = self.pulse_settings['inter_delay']
                    if isinstance(inter_delay, Sequence):
                        # inter_delay contains an element for each pulse
                        inter_delay = inter_delay[k]

                    t_connect = partial(RF_pulse['t_stop'].connect, offset=inter_delay)
            elif isinstance(RF_subpulse, PulseSequence):
                for pulse in RF_subpulse:
                    RF_pulse, = self.add(pulse, connect=False)
                    t_connect(RF_pulse['t_start'], offset=pulse.t_start)

                final_delay = RF_subpulse.duration - pulse.t_stop
                inter_delay = self.pulse_settings['inter_delay']
                assert not isinstance(inter_delay, Sequence)
                t_connect = partial(
                    RF_pulse['t_stop'].connect,
                    offset=final_delay + inter_delay
                )
            else:
                raise ValueError(
                    f'Pulse {RF_subpulse} not understood. It must either be '
                    f'a pulse or pulse sequence.'
                )

        if RF_pulse is not None:
            # Either connect stage_pulse.t_stop to the last RF_pulse, or
            # not if the RF_pulse starts too soon (depending on min_duration)
            t_stop = RF_pulse.t_stop + self.pulse_settings['post_delay']
            duration = t_stop - stage_pulse.t_start
            min_duration = self.pulse_settings['min_duration']
            if min_duration is not None and duration < min_duration:
                # Do not connect t_stop to last RF pulse since the post_delay
                # is too little
                # TODO There should ideally still be a connection such that
                # if the RF_pulse t_stop becomes larger, stage_pulse will
                # still extend
                stage_pulse.t_stop = stage_pulse.t_start + min_duration
            else:
                RF_pulse['t_stop'].connect(
                    stage_pulse['t_stop'], offset=self.pulse_settings['post_delay']
                )
        else:
            duration = self.pulse_settings['pre_delay'] + self.pulse_settings['post_delay']
            if self.pulse_settings['min_duration'] is not None:
                duration = max(duration, self.pulse_settings['min_duration'])
            stage_pulse.duration = duration

    def _add_RF_pulse_sequence_single_stage(self, RF_pulse_sequence):
        # Determine if a stage pulse is needed or not
        stage_pulse_needed = not any(
            pulse.connection_label == 'stage' for pulse in RF_pulse_sequence
        )
        if stage_pulse_needed:
            stage_pulse, = self.add(self.pulse_settings['stage_pulse'])
            # Note that we ignore pre_delay and post_delay
            stage_pulse.duration = RF_pulse_sequence.duration
            # Connect all pulses to stage_pulse.t_start
            connect_parameter = stage_pulse['t_start']
        else:
            # Connect all pulses to t_stop of last stage pulse
            last_pulse = self.get_pulse(connection_label='stage', t_stop=self.t_stop)
            connect_parameter = last_pulse['t_stop']

        pulses_add = []
        for pulse in RF_pulse_sequence:
            pulse_copy = deepcopy(pulse)
            pulse_copy.parent = self
            connect_parameter.connect(pulse_copy['t_start'], offset=pulse.t_start)
            pulses_add.append(pulse_copy)

        self.add(*pulses_add, copy=False)  # Already copied pulses

    def add_RF_pulses(self):
        """Add RF pulses to the pulse sequence

        Note:
              Each element in ESR_frequencies can also be a list of multiple
              frequencies, in which case multiple pulses with the provided
              subfrequencies will be used.
        """
        # Add pulses to pulse sequence
        for RF_pulses_single_stage in self.pulse_settings['RF_pulses']:

            if isinstance(RF_pulses_single_stage, PulseSequence):
                self._add_RF_pulse_sequence_single_stage(
                    RF_pulses_single_stage
                )
            else:
                self._add_RF_pulses_single_stage(RF_pulses_single_stage)

            if self.pulse_settings['transition_pulse'] is not None:
                self.add(self.pulse_settings['transition_pulse'])

            if self.pulse_settings['read_pulse'] is not None:
                self.add(self.pulse_settings['read_pulse'])

    def generate(self):
        # Clear all pulses from pulse sequence
        self.clear()

        self.pulse_settings['RF_pulses'] = self.convert_RF_pulse_labels_to_pulses()

        # Add pre_pulses
        self.add(*self.pulse_settings['pre_pulses'])
        # Update self.pulse_settings['RF_pulses']. Converts any pulses that are
        # strings to actual pulses, and sets correct frequencies
        for _ in range(self.pulse_settings['shots_per_frequency']):
            self.add_RF_pulses()

        # Add post_pulses
        self.add(*self.pulse_settings['post_pulses'])

        for modifier in self.modifiers:
            modifier(self)

        self._update_latest_pulse_settings()


class ESRPulseSequenceComposite(PulseSequence):
    """`PulseSequenceGenerator` for electron spin resonance (ESR).

    This pulse sequence combines two nested sequences:

    1. ESR: An `ElectronReadoutPulseSequence` (see docstring for details)
    2. EPR: A `PulseSequence` consisting of an empty, plunge, read pulse.
       The read pulse is a ``read_long`` pulse whose duration is supposed to
       be longer than a regular read pulse. The last portion of this pulse is
       used to estimate the dark counts, and consequently all ``up_proportion``
       results in the ``ESR`` pulse sequence will also have a corresponding
       ``contrast := up_proportion - dark_counts``.
       This pulse sequence can be disabled via
       ``ESRPulseSequenceComposite.EPR.enabled = False
"""
    def __init__(self, pulse_sequences=None, **kwargs):
        if pulse_sequences is None:
            pulse_sequences = [
                ElectronReadoutPulseSequence(name='ESR'),
                PulseSequence(
                    name='EPR',
                    pulses=[
                        DCPulse('empty', acquire=True),
                        DCPulse('plunge', acquire=True),
                        DCPulse('read_long', acquire=True)
                    ]
                )
            ]
        super().__init__(pulse_sequences=pulse_sequences, **kwargs)

        self.ESR = next(pseq for pseq in self.pulse_sequences if pseq.name == 'ESR')
        self.EPR = next(pseq for pseq in self.pulse_sequences if pseq.name == 'EPR')


class NMRPulseSequenceComposite(PulseSequence):
    """Pulse sequence for nuclear magnetic resonance (NMR).

    The pulse sequence consists of two nested pulse sequences:

    1. NMR: `ElectronReadoutPulseSequence` during which NMR pulses are to be
       applied.
    2. ESR: `ElectronReadoutPulseSequence` during which the NMR state is
       determined. The following settings are recommended:

       - ``shots_per_frequency should be larger than 1 for repeated non-
         demolition readout
       - ``RF_pulses`` Should contain two (or more) elements, one for each
         nuclear state.

    Additionally, the attribute ``initialization`` can be set to another
    `NMRPulseSequence`. This pulse sequence should correspond to a sequence
    that initializes the nucleus and reads it out. In this case, the samples
    are filtered to those where the initialization was successful.
    """
    def __init__(self, pulse_sequences=None, **kwargs):
        if pulse_sequences is None:
            pulse_sequences = [
                ElectronReadoutPulseSequence(name='NMR'),
                ElectronReadoutPulseSequence(name='ESR')
            ]

        super().__init__(pulse_sequences=pulse_sequences, **kwargs)

        self.ESR = next(pseq for pseq in self.pulse_sequences if pseq.name == 'ESR')
        self.NMR = next(pseq for pseq in self.pulse_sequences if pseq.name == 'NMR')
        try:
            self.initialization = next(
                pseq for pseq in self.pulse_sequences if pseq.name == 'initialization'
            )
        except:
            self.initialization = None

        # Disable read pulse acquire by default
        self.NMR.settings['read_pulse'].acquire = False


class T2PulseSequence(ElectronReadoutPulseSequence):
    """Pulse sequence for T2 measurements

    This is a subclass of the ElectronReadoutPulseSequence, and can be used for
    both ESR or NMR by passing it along as part of the ``pulse_sequences`` kwarg.

    This PulseSequence overrides the ``RF_pulses`` setting
    (see `ElectronReadoutPulseSequence` for details) with the following settings:

    - RF_initial_pulse: First RF pulse during stage (usually a pi/2 pulse)
    - RF_refocusing_pulse: Pulses between first and final RF pulse for refocusing.
      Ignored if num_refocusing == 0
    - RF_final_pulse: Final RF pulse during stage (usually a pi/2 pulse)
    - final_phase: Optional phase of the final RF pulse
    - artificial_frequency: Optional frequency with which to modify the phase of
      the final RF pulse, dependent on tau.
      This essentially creates oscillations without actually having to detune
      the RF pulses.
    - num_refocusing: Number of refocusing pulses to apply between initial and
      final RF pulse

    Additionally, this pulse sequence has the attribute ``tau``, which defines
    the time difference between the center of the initial pulse and the center of
    the final pulse.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pulse_settings.update({
            'RF_initial_pulse': 0,
            'RF_refocusing_pulse': 0,
            'RF_final_pulse': 0,
            'RF_inter_pulse': None,
            'final_phase': 0,
            'artificial_frequency': 0,
            'num_refocusing': 0,
            'additional_RF_pulses': []
        })
        self.tau: float = Parameter('tau', unit='s', initial_value=1e-3, parent=False)

    @parameter
    def tau_set(self, parameter, val):
        parameter._latest['value'] = val
        parameter._latest['raw_value'] = val
        # TODO this fails if there is more than one layer of nesting
        if isinstance(self.parent, PulseSequence):
            self.parent.generate()
        else:
            self.generate()

    def add_RF_pulses(self):
        self.settings['RF_pulses'] = [
            [
                self.settings['RF_initial_pulse'],
                *[self.settings['RF_refocusing_pulse'] for _ in range(self.settings['num_refocusing'])],
                self.settings['RF_final_pulse']
            ],
            *self.settings['additional_RF_pulses']
        ]

        self.settings['inter_delay'] = []
        for k, (RF_pulse, next_RF_pulse) in enumerate(zip(
                self.settings['RF_pulses'][0][:-1],
                self.settings['RF_pulses'][0][1:]
        )):
            inter_delay = self.tau / max(self.settings['num_refocusing'], 1)
            if self.settings['num_refocusing'] > 0 and (k == 0 or k == self.settings['num_refocusing']):
                inter_delay /= 2

            inter_delay -= RF_pulse.duration / 2
            inter_delay -= next_RF_pulse.duration / 2
            if inter_delay < 0:
                raise RuntimeError(
                    f'RF pulse inter_delay {inter_delay} is shorter than RF pulse duration'
                )
            self.settings['inter_delay'].append(inter_delay)

        # Replace all inter_delays by offresonant pulses
        if self.settings['RF_inter_pulse'] is not None and self.settings['RF_inter_pulse'].enabled:
            RF_pulses = []
            for RF_pulse, inter_delay in zip(self.settings['RF_pulses'][0], self.settings['inter_delay']):
                RF_pulses.append(RF_pulse)
                RF_pulse_inter = self.settings['RF_inter_pulse'].copy()
                RF_pulse_inter.duration = inter_delay
                RF_pulses.append(RF_pulse_inter)

            # Add final RF pulse (inter_delay has one less element than RF_pulses)
            RF_pulses.append(self.settings['RF_pulses'][0][-1])

            self.settings['RF_pulses'][0] = RF_pulses
            self.settings['inter_delay'] = 0

        # Calculate phase of final pulse
        final_phase = self.settings['final_phase']
        final_phase += 360 * self.tau * self.settings['artificial_frequency']
        final_phase = round(final_phase % 360)
        self.settings['RF_pulses'][0][-1].phase = final_phase

        super().add_RF_pulses()


# Deprecated pulse sequences

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

            ESR_pulse = None

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

            if ESR_pulse is not None:
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

        self._update_latest_pulse_settings()


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

            'num_refocusing': 0,

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
        self._update_latest_pulse_settings()


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
        pre_ESR_pulses (List[Pulse]): Pulses before ESR readout pulse sequence.
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
            'post_pulse': DCPulse('read', acquire=True),
            'intermediate_pulses': [],
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
        self.pulse_settings['pre_ESR_pulses'] = self.pre_ESR_pulses = []
        self.pulse_settings['post_pulses'] = self.post_pulses = []

    def add_NMR_pulses(self, pulse_sequence=None):
        if pulse_sequence is None:
            pulse_sequence = self

        # Convert any pulse strings to pulses if necessary
        for k, pulse in enumerate(self.NMR['NMR_pulses']):
            if isinstance(pulse, str):
                pulse_copy = deepcopy(self.NMR[pulse])
                self.NMR['NMR_pulses'][k] = pulse_copy
            elif isinstance(pulse, Iterable):
                # Pulse is a list containing sub-pulses
                # These pulses will be sequenced during a single stage pulse
                for kk, subpulse in enumerate(pulse):
                    if isinstance(subpulse, str):
                        subpulse_copy = deepcopy(self.NMR[subpulse])
                        self.NMR['NMR_pulses'][k][kk] = subpulse_copy

        self.primary_NMR_pulses = []  # Clear primary NMR pulses (first in each stage pulse)

        # Add pulses to pulse sequence
        for k, single_stage_NMR_pulses in enumerate(self.NMR['NMR_pulses']):
            # Each element should be the NMR pulses to apply within a single
            # stage, between each subsequence there will be a pre-delay and
            # post-delay

            if not isinstance(single_stage_NMR_pulses, Iterable):
                # Single NMR pulse provided, turn into list
                single_stage_NMR_pulses = [single_stage_NMR_pulses]

            NMR_stage_pulse, = self.add(self.NMR['stage_pulse'])
            t_connect = partial(NMR_stage_pulse['t_start'].connect,
                                offset=self.NMR['pre_delay'])

            self.primary_NMR_pulses.append(single_stage_NMR_pulses[0])

            NMR_pulse = None
            for kk, NMR_subpulse in enumerate(single_stage_NMR_pulses):
                NMR_pulse, = self.add(NMR_subpulse)
                t_connect(NMR_pulse['t_start'])

                if kk < len(single_stage_NMR_pulses) - 1:
                    # Determine delay between NMR pulses
                    if isinstance(self.NMR['inter_delay'], Iterable):
                        inter_delay = self.NMR['inter_delay'][kk]
                    else:
                        inter_delay = self.NMR['inter_delay']

                    t_connect = partial(NMR_pulse['t_stop'].connect, offset=inter_delay)

            if NMR_pulse is not None:
                NMR_pulse['t_stop'].connect(NMR_stage_pulse['t_stop'],
                                            offset=self.NMR['post_delay'])

            if k < len(self.NMR['NMR_pulses'])-1:
                # Add any intermediate pulses, except for the final NMR sequence
                t_connect = partial(NMR_stage_pulse['t_stop'].connect, offset=0)
                for intermediate_pulse in self.NMR['intermediate_pulses']:
                    int_pulse, = self.add(intermediate_pulse)
                    t_connect(int_pulse['t_start'])
                    t_connect = partial(int_pulse['t_stop'].connect, offset=0)
            elif self.NMR['post_pulse'] is not None:
                # Add final pulse
                post_pulse, = self.add(self.NMR['post_pulse'])
                NMR_stage_pulse['t_stop'].connect(post_pulse['t_start'])

        return pulse_sequence

    def add_ESR_pulses(self, pulse_sequence=None, previous_pulse=None):
        if pulse_sequence is None:
            pulse_sequence = self

        for _ in range(self.ESR['shots_per_frequency']):
            for ESR_pulses in self.ESR['ESR_pulses']:
                # Add a plunge and read pulse for each frequency

                if not isinstance(ESR_pulses, list):
                    # Treat frequency as list, as one could add multiple ESR
                    # pulses
                    ESR_pulses = [ESR_pulses]

                stage_pulse, = pulse_sequence.add(self.ESR['stage_pulse'])
                for k, ESR_pulse in enumerate(ESR_pulses):

                    if isinstance(ESR_pulse, str):
                        # Pulse is a reference to some pulse in self.ESR
                        ESR_pulse = self.ESR[ESR_pulse]

                    ESR_pulse, = pulse_sequence.add(ESR_pulse)

                    # Delay also depends on any previous ESR pulses
                    delay = self.ESR['pre_delay'] + k * self.ESR['inter_delay']
                    stage_pulse['t_start'].connect(ESR_pulse['t_start'],
                                                    offset=delay)
                ESR_pulse['duration'].connect(stage_pulse['t_stop'],
                                            offset=lambda p: p.parent.t_start + self.ESR['post_delay'])
                pulse_sequence.add(self.ESR['read_pulse'])

    def generate(self):
        """Updates the pulse sequence"""

        # Initialize pulse sequence
        self.clear()

        self.add(*self.pulse_settings['pre_pulses'])

        self.add_NMR_pulses()

        # Note: This was added when performing NMR on the electron-up manifold,
        # where it is important to reload a spin-down electron for readout.
        self.add(*self.pulse_settings['pre_ESR_pulses'])

        self.add_ESR_pulses()

        self.add(*self.pulse_settings['post_pulses'])

        # Create copy of current pulse settings for comparison later
        self._update_latest_pulse_settings()


class NMRCPMGPulseSequence(NMRPulseSequence):
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
        pre_ESR_pulses (List[Pulse]): Pulses before ESR readout pulse sequence.
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
            'post_delay': 2e-3,
            'final_phase': 0,
            'artificial_frequency': 0
        }
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
        self.pulse_settings['pre_ESR_pulses'] = self.pre_ESR_pulses = []
        self.pulse_settings['post_pulses'] = self.post_pulses = []

        self.primary_NMR_pulses = []  # Primary NMR pulses (first in each stage pulse)

        self.generate()

    def add_NMR_pulses(self, pulse_sequence=None):
        if pulse_sequence is None:
            pulse_sequence = self

        # Convert any pulse strings to pulses if necessary
        for k, pulse in enumerate(self.NMR['NMR_pulses']):
            if isinstance(pulse, str):
                pulse_copy = deepcopy(self.NMR[pulse])
                self.NMR['NMR_pulses'][k] = pulse_copy
            elif isinstance(pulse, Iterable):
                # Pulse is a list containing sub-pulses
                # These pulses will be sequenced during a single stage pulse
                for kk, subpulse in enumerate(pulse):
                    if isinstance(subpulse, str):
                        subpulse_copy = deepcopy(self.NMR[subpulse])
                        self.NMR['NMR_pulses'][k][kk] = subpulse_copy

        self.primary_NMR_pulses = []  # Clear primary NMR pulses (first in each stage pulse)

        # Add pulses to pulse sequence
        for single_stage_NMR_pulses in self.NMR['NMR_pulses']:
            # Each element should be the NMR pulses to apply within a single
            # stage, between each subsequence there will be a pre-delay and
            # post-delay

            if not isinstance(single_stage_NMR_pulses, Iterable):
                # Single NMR pulse provided, turn into list
                single_stage_NMR_pulses = [single_stage_NMR_pulses]

            NMR_stage_pulse, = self.add(self.NMR['stage_pulse'])
            t_connect = partial(NMR_stage_pulse['t_start'].connect,
                                offset=self.NMR['pre_delay'])

            self.primary_NMR_pulses.append(single_stage_NMR_pulses[0])

            NMR_pulse = None
            for k, NMR_subpulse in enumerate(single_stage_NMR_pulses):
                NMR_pulse, = self.add(NMR_subpulse)
                t_connect(NMR_pulse['t_start'])
                # The first and last pulses have only a single inter-delay time between pulses
                if k == 0 or k == len(single_stage_NMR_pulses) - 2:
                    t_connect = partial(NMR_pulse['t_stop'].connect,
                                        offset=self.NMR['inter_delay'])
                else:
                    t_connect = partial(NMR_pulse['t_stop'].connect,
                                        offset=self.NMR['inter_delay']*2)

            if NMR_pulse is not None:
                NMR_pulse['t_stop'].connect(NMR_stage_pulse['t_stop'],
                                            offset=self.NMR['post_delay'])

        return pulse_sequence

    def add_ESR_pulses(self, pulse_sequence=None, previous_pulse=None):
        if pulse_sequence is None:
            pulse_sequence = self

        for _ in range(self.ESR['shots_per_frequency']):
            for ESR_pulses in self.ESR['ESR_pulses']:
                # Add a plunge and read pulse for each frequency

                if not isinstance(ESR_pulses, list):
                    # Treat frequency as list, as one could add multiple ESR
                    # pulses
                    ESR_pulses = [ESR_pulses]

                stage_pulse, = pulse_sequence.add(self.ESR['stage_pulse'])
                delay = self.ESR['pre_delay']
                for k, ESR_pulse in enumerate(ESR_pulses):

                    if isinstance(ESR_pulse, str):
                        # Pulse is a reference to some pulse in self.ESR
                        ESR_pulse = self.ESR[ESR_pulse]

                    ESR_pulse, = pulse_sequence.add(ESR_pulse)

                    # Delay also depends on any previous ESR pulses
                    stage_pulse['t_start'].connect(ESR_pulse['t_start'],
                                                   offset=delay)
                    delay = delay + ESR_pulse['duration'].get() + self.ESR['inter_delay']

                ESR_pulse['duration'].connect(stage_pulse['t_stop'],
                                              offset=lambda p: p.parent.t_start + self.ESR['post_delay'])
                pulse_sequence.add(self.ESR['read_pulse'])

    def generate(self):
        """Updates the pulse sequence"""

        # Initialize pulse sequence
        self.clear()

        self.add(*self.pulse_settings['pre_pulses'])

        self.add_NMR_pulses()

        # Note: This was added when performing NMR on the electron-up manifold,
        # where it is important to reload a spin-down electron for readout.
        self.add(*self.pulse_settings['pre_ESR_pulses'])

        self.add_ESR_pulses()

        self.add(*self.pulse_settings['post_pulses'])

        # Create copy of current pulse settings for comparison later
        self._update_latest_pulse_settings()


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


class CircuitPulseSequence(ElectronReadoutPulseSequence):
    def __init__(self, name='circuit', circuits_file=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.settings["pulses"] = {}

        self.circuit_index = Parameter(vals=vals.Ints(), parent=False)
        self.circuit = Parameter(vals=vals.Strings())
        self.circuits = Parameter(
            vals=vals.Lists(vals.Strings()), set_cmd=None, initial_value=[]
        )

        self.circuits_file = Parameter(initial_value=circuits_file, set_cmd=None)
        self.circuits_folder = Parameter(
            config_link='properties.circuits_folder', set_cmd=None
        )

    @parameter
    def circuit_index_set(self, parameter, idx):
        self.circuit = self.circuits[idx]

    @parameter
    def circuit_set(self, parameter, circuit):
        # Update current value of circuit
        parameter._latest["raw_value"] = parameter._latest["value"] = circuit

        gates = convert_circuit(circuit, target_type=list)

        unknown_gates = [gate for gate in gates if gate not in self.settings.pulses]
        if unknown_gates:
            raise RuntimeError(
                f'The following pulses are not registered in '
                f'CircuitPulseSequence.settings.pulses: {unknown_gates}'
            )

        self.settings.RF_pulses = self.gates_to_RF_pulses(gates)

        self.generate()

    def gates_to_RF_pulses(self, gates):
        RF_pulses = [[]]
        for gate in gates:
            pulse_element = self.settings.pulses[gate]
            if isinstance(pulse_element, (Pulse, PulseSequence)):
                RF_pulses[0].append(pulse_element)
            elif isinstance(pulse_element, (list, tuple)):
                RF_pulses[0].extend(pulse_element)
        return RF_pulses

    def load_circuits(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        if not filepath.exists():
            if self.circuits_folder is None:
                raise RuntimeError(f'Could not find filepath at {filepath}')

            filepath = Path(self.circuits_folder) / filepath

        if not filepath.exists():
            raise RuntimeError(f'Could not find filepath at {filepath}')

        self.circuits = load_circuits(filepath, target_type=str)

        return self.circuits

    def save_circuits(self, filepath, name='circuits.txt'):

        if isinstance(filepath, Measurement):
            filepath = filepath.dataset.filepath
        elif isinstance(filepath, DataSet):
            filepath = filepath.filepath

        filepath = Path(filepath)

        if filepath.is_dir():
            filepath = filepath / name

        with filepath.open('w') as f:
            f.write('\n'.join(self.circuits))

        return filepath

    def save(self, filepath):
        return self.save_circuits(filepath)


class ESRRamseyDetuningPulseSequence(ESRPulseSequence):
    """" Created to implement an arbitrary number of DC pulses in a Ramsey sequence during the wait time. Please Refer to ESRPulseSequence for the ESR pulses.

    Highlights:

   - DC pulses can be stored in  ['ESR']['detuning_pulses'] and will become the new 'stage_pulse'
   - t_start_detuning is the time at which the DC detuning pulses start. In the case the detuning starts right after the ESR pi/2 , then this time should
    be equal to ['pre_delay'] +ESR['piHalf'].duration
   - If  the time for the detuning pulses is shorter that the total stage duration, the final part of the pulse (called post_stage) will the standard stage pulse  """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pulse_settings['ESR']['t_start_detuning'] = 0
        self.pulse_settings['ESR']['detuning_pulses'] = []

    def add_ESR_pulses(self, ESR_frequencies=None):
        super().add_ESR_pulses(ESR_frequencies=ESR_frequencies)
        # At this point there is a single `stage` pulse for each group of ESR pulses.
        # We want to inject our detuning  pulses in between this stage pulse


        if self.pulse_settings['EPR']['enabled'] :
            raise NotImplementedError('Currently not programmed to include EPR pulse')



        stage_pulse = self.get_pulse(name=self.ESR['stage_pulse'].name)

        assert stage_pulse is not None, "Could not find existing stage pulse in pulse sequence"
        self.remove(stage_pulse)

        if any(pulse.connection_label != stage_pulse.connection_label
               or pulse.connection != stage_pulse.connection
               for pulse in self.ESR['detuning_pulses']):
            raise RuntimeError('All detuning pulses must have same connection as stage pulse')

        t = stage_pulse._delay + self.ESR['t_start_detuning']
        # Add an initial stage pulse if t_start_detuning > 0
        if self.pulse_settings['ESR']['t_start_detuning'] > 0:
            pre_stage_pulse, = self.add(stage_pulse)
            pre_stage_pulse.name = 'pre_stage'
            pre_stage_pulse.t_stop = t

        # Add all detuning pulses
        for pulse in self.ESR['detuning_pulses']:
            detuning_pulse, = self.add(pulse)
            detuning_pulse.t_start = t
            t += detuning_pulse.duration

        if t > stage_pulse.t_stop:
            raise RuntimeError('Total duration of detuning pulses exceeds total stage pulse duration')
        elif t < stage_pulse.t_stop:
            # Add a final stage pulse
            post_stage_pulse, = self.add(stage_pulse)
            post_stage_pulse.name = 'post_stage'
            post_stage_pulse.t_start = t
            post_stage_pulse.t_stop = stage_pulse.t_stop
