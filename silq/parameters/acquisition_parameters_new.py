from typing import List
import numpy as np
from copy import copy

from silq.parameters.acquisition_parameters import AcquisitionParameter
from silq.pulses.pulse_sequences import ESRPulseSequenceNew
from silq.tools import property_ignore_setter
from silq import analysis

from qcodes.instrument.parameter_node import ParameterNode
from qcodes.config.config import DotDict


class ESRParameterNew(AcquisitionParameter):
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
        self._names = []

        self.pulse_sequence = ESRPulseSequenceNew()
        self.ESR = self.pulse_sequence.ESR
        self.EPR = self.pulse_sequence.EPR
        self.frequencies = self.ESR.frequencies

        super().__init__(name=name,
                         names=['contrast', 'dark_counts',
                                'voltage_difference_read'],
                         snapshot_value=False,
                         properties_attrs=['analyses'],
                         **kwargs)
        self.analyses = ParameterNode()
        self.analyses.EPR = analysis.AnalyseEPR('EPR')
        self.analyses.ESR = analysis.AnalyseElectronReadout('ESR')

    @property
    def names(self):
        names = []
        for analysis_name, analysis in self.analyses.parameter_nodes.items():
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
            results['EPR'] = self.analyses['EPR'].analyse(
                empty_traces=traces.EPR[0]['output'],
                plunge_traces=traces.EPR[1]['output'],
                read_traces=traces.EPR[2]['output'],
                plot=plot
            )
            dark_counts = results['EPR']['dark_counts']
        else:
            dark_counts = None

        if 'ESR' in traces:
            results['ESR'] = self.analyses['ESR'].analyse(
                traces=traces.ESR,
                dark_counts=dark_counts,
                plot=plot
            )

        self.results = results
        return results