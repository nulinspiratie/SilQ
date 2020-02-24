from typing import List
import numpy as np
from copy import copy
from typing import Dict, Iterable

from silq.parameters.acquisition_parameters import AcquisitionParameter
from silq.pulses.pulse_sequences import ESRPulseSequenceComposite
from silq.pulses.pulse_types import Pulse
from silq.tools import property_ignore_setter
from silq.analysis import analysis

from qcodes.instrument.parameter_node import ParameterNode
from qcodes.config.config import DotDict


class ESRParameterComposite(AcquisitionParameter):
    """Parameter for most pulse sequences involving electron spin resonance.

    This parameter can handle many of the simple pulse sequences involving ESR.
    It uses the `ESRPulseSequence`, which will generate a pulse sequence from
    settings (see parameters below).

    In general the pulse sequence is as follows:

    1. Perform any pre_pulses defined in ``ESRParameter.pre_pulses``.
    2. Perform stage pulse ``ESRParameter.ESR['stage_pulse']``.
       By default, this is the ``plunge`` pulse.
    3. Perform ESR pulse within plunge pulse, the delay from start of plunge
       pulse is defined in ``ESRParameter.ESR['pulse_delay']``.
    4. Perform read pulse ``ESRParameter.ESR['read_pulse']``.
    5. Repeat steps 2 and 3 for each ESR pulse in
       ``ESRParameter.ESR['ESR_pulses']``, which by default contains single
       pulse ``ESRParameter.ESR['ESR_pulse']``.
    6. Perform empty-plunge-read sequence (EPR), but only if
       ``ESRParameter.EPR['enabled']`` is True.
       EPR pulses are defined in ``ESRParameter.EPR['pulses']``.
    7. Perform any post_pulses defined in ``ESRParameter.post_pulses``.

    A shorthand for using the default ESR pulse for multiple frequencies is by
    setting `ESRParameter.ESR_frequencies`. Settings this will create a copy
    of ESRParameter.ESR['ESR_pulse'] with the respective frequency.

    Examples:
        The following code measures two ESR frequencies and performs an EPR
        from which the contrast can be determined for each ESR frequency:

        >>> ESR_parameter = ESRParameter()
        >>> ESR_parameter.ESR['pulse_delay'] = 5e-3
        >>> ESR_parameter.ESR['stage_pulse'] = DCPulse['plunge']
        >>> ESR_parameter.ESR['ESR_pulse'] = FrequencyRampPulse('ESR_adiabatic')
        >>> ESR_parameter.ESR_frequencies = [39e9, 39.1e9]
        >>> ESR_parameter.EPR['enabled'] = True
        >>> ESR_parameter.pulse_sequence.generate()

        The total pulse sequence is plunge-read-plunge-read-empty-plunge-read
        with an ESR pulse in the first two plunge pulses, 5 ms after the start
        of the plunge pulse. The ESR pulses have different frequencies.

    Args:
        name: Name of acquisition parameter
        **kwargs: Additional kwargs passed to `AcquisitionParameter`.

    Parameters:
        ESR (dict): `ESRPulseSequence` generator settings for ESR. Settings are:
            ``stage_pulse``, ``ESR_pulse``, ``ESR_pulses``, ``pulse_delay``,
            ``read_pulse``.
        EPR (dict): `ESRPulseSequence` generator settings for EPR.
            This is optional and can be toggled in ``EPR['enabled']``.
            If disabled, contrast is not calculated.
            Settings are: ``enabled``, ``pulses``.
        pre_pulses (List[Pulse]): Pulses to place at the start of the sequence.
        post_pulses (List[Pulse]): Pulses to place at the end of the sequence.
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        t_skip (float): initial part of read trace to ignore for measuring
            blips. Useful if there is a voltage spike at the start, which could
            otherwise be measured as a ``blip``. Retrieved from
            ``silq.config.properties.t_skip``.
        t_read (float): duration of read trace to include for measuring blips.
            Useful if latter half of read pulse is used for initialization.
            Retrieved from ``silq.config.properties.t_read``.
        min_filter_proportion (float): Minimum number of read traces needed in
            which the voltage starts low (loaded donor). Otherwise, most results
            are set to zero. Retrieved from
            ``silq.config.properties.min_filter_proportion``.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties``.
            See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.

    Notes:
        - All pulse settings are copies of
          ``ESRParameter.pulse_sequence.pulse_settings``.
        - For given pulse settings, ``ESRParameter.pulse_sequence.generate``
          will recreate the pulse sequence from settings.
    """
    def __init__(self, name='ESR', **kwargs):
        self.pulse_sequence = ESRPulseSequenceComposite()
        self.ESR = self.pulse_sequence.ESR
        self.EPR = self.pulse_sequence.EPR
        self.frequencies = self.ESR.frequencies

        self.analyses = ParameterNode()
        self.analyses.EPR = analysis.AnalyseEPR('EPR')
        self.analyses.ESR = analysis.AnalyseElectronReadout('ESR')

        self.layout.sample_rate.connect(self.analyses.EPR.settings['sample_rate'])
        self.layout.sample_rate.connect(self.analyses.ESR.settings['sample_rate'])
        self.EPR['enabled'].connect(self.analyses.ESR.results['contrast'])
        self.EPR['enabled'].connect(self.analyses.EPR['enabled'])
        self.ESR['enabled'].connect(self.analyses.ESR['enabled'])

        num_frequencies = self.analyses.ESR.settings['num_frequencies']
        num_frequencies.get_raw = lambda: len(self.frequencies)
        num_frequencies.get = num_frequencies._wrap_get(num_frequencies.get_raw)

        super().__init__(name=name,
                         names=(),
                         snapshot_value=False,
                         properties_attrs=['analyses'],
                         **kwargs)

    @property_ignore_setter
    def names(self):
        names = []
        for analysis_name, analysis in self.analyses.parameter_nodes.items():
            if not analysis.enabled:
                continue

            for name in analysis.names:
                names.append(f'{analysis_name}.{name}')
        return names

    @property_ignore_setter
    def shapes(self):
        return ((), ) * len(self.names)

    @property_ignore_setter
    def units(self):
        units = []
        for analysis_name, analysis in self.analyses.parameter_nodes.items():
            if not analysis.enabled:
                continue

            units += analysis.units
        return tuple(units)

    def analyse(self, traces = None, plot=False):
        """Analyse ESR traces.

        If there is only one ESR pulse, returns ``up_proportion_{pulse.name}``.
        If there are several ESR pulses, adds a zero-based suffix at the end for
        each ESR pulse. If ``ESRParameter.EPR['enabled'] == True``, the results
        from `analyse_EPR` are also added, as well as ``contrast_{pulse.name}``
        (plus a suffix if there are several ESR pulses).
        """
        if traces is None:
            traces = self.traces
        # Convert to DotDict so we can access nested pulse sequences
        traces = DotDict(traces)

        results = DotDict()
        if 'EPR' in traces:
            results['EPR'] = self.analyses.EPR.analyse(
                empty_traces=traces[self.EPR[0].full_name]['output'],
                plunge_traces=traces[self.EPR[1].full_name]['output'],
                read_traces=traces[self.EPR[2].full_name]['output'],
                plot=plot
            )
            dark_counts = results['EPR']['dark_counts']
        else:
            dark_counts = None

        if 'ESR' in traces:
            results['ESR'] = self.analyses.ESR.analyse(
                traces=traces.ESR,
                dark_counts=dark_counts,
                plot=plot
            )

        self.results = results
        return results


class NMRParameterComposite(AcquisitionParameter):
    """ Parameter for most measurements involving an NMR pulse.

    This parameter can apply several NMR pulses, and also measure several ESR
    frequencies. It uses the `NMRPulseSequence`, which will generate a pulse
    sequence from settings (see parameters below).

    In general, the pulse sequence is as follows:

    1. Perform any pre_pulses defined in ``NMRParameter.pre_pulses``.
    2. Perform NMR sequence

       1. Perform stage pulse ``NMRParameter.NMR['stage_pulse']``.
          Default is 'empty' `DCPulse`.
       2. Perform NMR pulses within the stage pulse. The NMR pulses defined
          in ``NMRParameter.NMR['NMR_pulses']`` are applied successively.
          The delay after start of the stage pulse is
          ``NMRParameter.NMR['pre_delay']``, delays between NMR pulses is
          ``NMRParameter.NMR['inter_delay']``, and the delay after the final
          NMR pulse is ``NMRParameter.NMR['post_delay']``.

    3. Perform ESR sequence

       1. Perform stage pulse ``NMRParameter.ESR['stage_pulse']``.
          Default is 'plunge' `DCPulse`.
       2. Perform ESR pulse within stage pulse for first pulse in
          ``NMRParameter.ESR['ESR_pulses']``.
       3. Perform ``NMRParameter.ESR['read_pulse']``, and acquire trace.
       4. Repeat steps 1 - 3 for each ESR pulse. The different ESR pulses
          usually correspond to different ESR frequencies (see
          `NMRParameter`.ESR_frequencies).
       5. Repeat steps 1 - 4 for ``NMRParameter.ESR['shots_per_frequency']``
          This effectively interleaves the ESR pulses, which counters effects of
          the nucleus flipping within an acquisition.

    This acquisition is repeated ``NMRParameter.samples`` times. If the nucleus
    is in one of the states for which an ESR frequency is on resonance, a high
    ``up_proportion`` is measured, while for the other frequencies a low
    ``up_proportion`` is measured. By looking over successive samples and
    measuring how often the ``up_proportions`` switch between above/below
    ``NMRParameter.threshold_up_proportion``, nuclear flips can be measured
    (see `NMRParameter.analyse` and `analyse_flips`).

    Args:
        name: Parameter name
        **kwargs: Additional kwargs passed to `AcquisitionParameter`

    Parameters:
        NMR (dict): `NMRPulseSequence` pulse settings for NMR. Settings are:
            ``stage_pulse``, ``NMR_pulse``, ``NMR_pulses``, ``pre_delay``,
            ``inter_delay``, ``post_delay``.
        ESR (dict): `NMRPulseSequence` pulse settings for ESR. Settings are:
            ``ESR_pulse``, ``stage_pulse``, ``ESR_pulses``, ``read_pulse``,
            ``pulse_delay``.
        EPR (dict): `PulseSequenceGenerator` settings for EPR. This is optional
            and can be toggled in ``EPR['enabled']``. If disabled, contrast is
            not calculated.
        pre_pulses (List[Pulse]): Pulses to place at the start of the sequence.
        post_pulses (List[Pulse]): Pulses to place at the end of the sequence.
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        ESR_frequencies (List[float]): List of ESR frequencies to use. When set,
            a copy of ``NMRParameter.ESR['ESR_pulse']`` is created for each
            frequency, and added to ``NMRParameter.ESR['ESR_pulses']``.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        t_skip (float): initial part of read trace to ignore for measuring
            blips. Useful if there is a voltage spike at the start, which could
            otherwise be measured as a ``blip``. Retrieved from
            ``silq.config.properties.t_skip``.
        t_read (float): duration of read trace to include for measuring blips.
            Useful if latter half of read pulse is used for initialization.
            Retrieved from ``silq.config.properties.t_read``.
        threshold_up_proportion (Union[float, Tuple[float, float]): threshold
            for up proportions needed to determine ESR pulse to be on-resonance.
            If tuple, first element is threshold below which ESR pulse is
            off-resonant, and second element is threshold above which ESR pulse
            is on-resonant. Useful for filtering of up proportions at boundary.
            Retrieved from
            ``silq.config.properties.threshold_up_proportion``.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties`` See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.

    Note:
        - The `NMRPulseSequence` does not have an empty-plunge-read (EPR)
          sequence, and therefore does not add a contrast or dark counts.
          Verifying that the system is in tune is therefore a little bit tricky.

    """
    def __init__(self, name: str = 'NMR', **kwargs):
        # self.pulse_sequence = NMRPulseSequence()
        self.NMR = self.pulse_sequence.NMR
        self.ESR = self.pulse_sequence.ESR
        self.pre_pulses = self.pulse_sequence.pulse_settings['pre_pulses']
        self.pre_ESR_pulses = self.pulse_sequence.pulse_settings['pre_ESR_pulses']
        self.post_pulses = self.pulse_sequence.pulse_settings['post_pulses']

        self.combined_flips_mode = combined_flips_mode

        super().__init__(name=name,
                         names=(),
                         snapshot_value=False,
                         properties_attrs=['analyses'],
                         **kwargs)

    @property
    def names(self):
        names = []

        for name in self._names:
            if name in ['flips', 'flip_probability', 'up_proportions']:
                if len(self.ESR_frequencies) == 1:
                    names.append(name)
                else:
                    names += [f'{name}_{k}'
                              for k in range(len(self.ESR_frequencies))]
            elif name in ['combined_flips', 'combined_flip_probability',
                          'filtered_combined_flips',
                          'filtered_combined_flip_probability'] and \
                            len(self.ESR_frequencies) > 1:
                if self.combined_flips_mode == 'neighbouring':
                    names += [f'{name}_{k}{k+1}'
                              for k in range(len(self.ESR_frequencies) - 1)]
                elif self.combined_flips_mode == 'all':
                    for k in range(len(self.ESR_frequencies) - 1):
                        for kk in range(k+1, len(self.ESR_frequencies)):
                            names.append(f'{name}_{k}{k+1}')
                else:
                    raise SyntaxError(f'Unknown combined_flips_mode: {self.combined_flips_mode}')
            elif name in ['filtered_flips', 'filtered_flip_probability'] and \
                            len(self.ESR_frequencies) > 1:
                for k in range(0, len(self.ESR_frequencies)):
                    if k > 0:
                        names.append(f'{name}_{k}_{k-1}{k}')
                    if k < len(self.ESR_frequencies) - 1:
                        names.append(f'{name}_{k}_{k}{k+1}')
        return names

    @names.setter
    def names(self, names):
        self._names = names

    @property_ignore_setter
    def shapes(self):
        return tuple((self.samples,) if 'up_proportions' in name else ()
                     for name in self.names)

    @property_ignore_setter
    def units(self):
        return ('', ) * len(self.names)

    @property
    def ESR_frequencies(self):
        """ESR frequencies to measure.

        For each ESR frequency, ``NMRParameter.ESR['shots_per_read']`` reads
        are performed.
        """
        ESR_frequencies = []
        for pulse in self.ESR['ESR_pulses']:
            if isinstance(pulse, Pulse):
                ESR_frequencies.append(pulse.frequency)
            elif isinstance(pulse, str):
                ESR_frequencies.append(self.ESR[pulse].frequency)
            elif isinstance(pulse, Iterable):
                ESR_subfrequencies = []
                for subpulse in pulse:
                    if isinstance(subpulse, Pulse):
                        ESR_subfrequencies.append(subpulse.frequency)
                    elif isinstance(subpulse, str):
                        ESR_subfrequencies.append(self.ESR[subpulse].frequency)
                    else:
                        raise SyntaxError(f'Subpulse type not allowed: {subpulse}')
                ESR_frequencies.append(ESR_subfrequencies)
            else:
                raise SyntaxError(f'pulse type not allowed: {pulse}')
        return ESR_frequencies

    @ESR_frequencies.setter
    def ESR_frequencies(self, ESR_frequencies: List):
        assert len(ESR_frequencies) == len(self.ESR['ESR_pulses']), \
        'Different number of frequencies to ESR pulses.'

        updated_ESR_pulses = []
        for ESR_subpulses, ESR_subfrequencies in zip(self.ESR['ESR_pulses'], ESR_frequencies):
            if isinstance(ESR_subpulses, str):
                ESR_subpulses = copy(self.ESR[ESR_subpulses])
            elif isinstance(ESR_subpulses, Iterable):
                ESR_subpulses = [
                    copy(self.ESR[p]) if isinstance(p, str) else p
                    for p in ESR_subpulses]

            # Either both the subpulses and subfrequencies must be iterable, or neither are (XNOR)
            assert \
                (
                    isinstance(ESR_subpulses, Iterable) and
                    isinstance(ESR_subfrequencies, Iterable)
                ) or (
                    not (isinstance(ESR_subpulses, Iterable) or isinstance(
                        ESR_subfrequencies, Iterable))
                ), \
            'Data structures for frequencies and pulses do not have the same shape.'

            if not isinstance(ESR_subpulses, Iterable):
                ESR_subpulses = [ESR_subpulses]
            if not isinstance(ESR_subfrequencies, Iterable):
                ESR_subfrequencies = [ESR_subfrequencies]

            for pulse, frequency in zip(ESR_subpulses,
                                        ESR_subfrequencies):
                    pulse.frequency = frequency

            updated_ESR_pulses.append(ESR_subpulses)
        self.ESR['ESR_pulses'] = updated_ESR_pulses

    def analyse(self, traces: Dict[str, Dict[str, np.ndarray]] = None):
        """Analyse flipping events between nuclear states

        Returns:
            (Dict[str, Any]): Dict containing:

            :results_read (dict): `analyse_traces` results for each read
              trace
            :up_proportions_{idx} (np.ndarray): Up proportions, the
              dimensionality being equal to ``NMRParameter.samples``.
              ``{idx}`` is replaced with the zero-based ESR frequency index.
            :Results from `analyse_flips`. These are:

              - flips_{idx},
              - flip_probability_{idx}
              - combined_flips_{idx1}{idx2}
              - combined_flip_probability_{idx1}{idx2}

              Additionally, each of the above results will have another result
              with the same name, but prepended with ``filtered_``, and appended
              with ``_{idx1}{idx2}`` if not already present. Here, all the
              values are filtered out where the corresponding pair of
              up_proportion samples do not have exactly one high and one low for
              each sample. The values that do not satisfy the filter are set to
              ``np.nan``.

              :filtered_scans_{idx1}{idx2}:
        """
        if traces is None:
            traces = self.traces

        results = {'results_read': []}

        if hasattr(self, 'threshold_voltage'):
            threshold_voltage = getattr(self, 'threshold_voltage')
        else:
            # Calculate threshold voltages from combined read traces
            high_low = analysis.find_high_low(
                np.ravel([trace['output'] for pulse_name, trace in traces.items()
                          if pulse_name.startswith('read_initialize')]))
            threshold_voltage = high_low['threshold_voltage']

        # Extract points per shot from a single read trace
        single_read_traces_name = f"{self.ESR['read_pulse'].name}[0]"
        single_read_traces = traces[single_read_traces_name]['output']
        points_per_shot = single_read_traces.shape[1]

        self.read_traces = np.zeros((len(self.ESR_frequencies), self.samples,
                                     self.ESR['shots_per_frequency'],
                                     points_per_shot))
        up_proportions = np.zeros((len(self.ESR_frequencies), self.samples))
        for f_idx, ESR_frequency in enumerate(self.ESR_frequencies):
            for sample in range(self.samples):
                # Create array containing all read traces
                read_traces = np.zeros(
                    (self.ESR['shots_per_frequency'], points_per_shot))
                for shot_idx in range(self.ESR['shots_per_frequency']):
                    # Read traces of different frequencies are interleaved
                    traces_idx = f_idx + shot_idx * len(self.ESR_frequencies)
                    traces_name = f"{self.ESR['read_pulse'].name}[{traces_idx}]"
                    read_traces[shot_idx] = traces[traces_name]['output'][sample]
                self.read_traces[f_idx, sample] = read_traces
                read_result = analysis.analyse_traces(
                    traces=read_traces,
                    sample_rate=self.sample_rate,
                    t_read=self.t_read,
                    t_skip=self.t_skip,
                    threshold_voltage=threshold_voltage)
                up_proportions[f_idx, sample] = read_result['up_proportion']
                results['results_read'].append(read_result)

            if len(self.ESR_frequencies) > 1:
                results[f'up_proportions_{f_idx}'] = up_proportions[f_idx]
            else:
                results['up_proportions'] = up_proportions[f_idx]

        # Add singleton dimension because analyse_flips handles 3D up_proportions
        up_proportions = np.expand_dims(up_proportions, 1)
        results_flips = analysis.analyse_flips(
            up_proportions_arrs=up_proportions,
            threshold_up_proportion=self.threshold_up_proportion)
        # Add results, only choosing first element so its no longer an array
        results.update({k: v[0] for k, v in results_flips.items()})
        return results
