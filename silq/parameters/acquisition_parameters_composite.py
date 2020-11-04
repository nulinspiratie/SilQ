from typing import List
import numpy as np
from typing import Dict, Iterable
from functools import partial
import logging

from silq.parameters.acquisition_parameters import AcquisitionParameter
from silq.pulses.pulse_sequences import (
    ESRPulseSequenceComposite,
    NMRPulseSequenceComposite,
)
from silq.pulses.pulse_types import Pulse
from silq.tools import property_ignore_setter
from silq.analysis.analysis import AnalyseElectronReadout, AnalyseEPR, AnalyseMultiStateReadout

from qcodes.instrument.parameter_node import ParameterNode
from qcodes.config.config import DotDict
from qcodes.plots.qcmatplotlib import MatPlot

__all__ = [
    'AcquisitionParameterComposite',
    'ESRParameterComposite',
    'NMRParameterComposite'
]

logger = logging.getLogger(__name__)


class AcquisitionParameterComposite(AcquisitionParameter):
    def __init__(self, name, **kwargs):
        super().__init__(
            name=name,
            names=(),
            snapshot_value=False,
            properties_attrs=["analyses"],
            **kwargs,
        )
        self.results = DotDict()

    @property_ignore_setter
    def names(self):
        names = []
        for analysis_name, analysis in self.analyses.parameter_nodes.items():
            if not analysis.enabled:
                continue

            for name in analysis.names:
                names.append(f'{analysis.name}.{name}')

        return names

    @property_ignore_setter
    def shapes(self):
        shapes = []
        for analysis_name, analysis in self.analyses.parameter_nodes.items():
            if not analysis.enabled:
                continue

            shapes += analysis.shapes
        return shapes

    @property_ignore_setter
    def units(self):
        units = []
        for analysis_name, analysis in self.analyses.parameter_nodes.items():
            if not analysis.enabled:
                continue

            units += analysis.units
        return units


class ESRParameterComposite(AcquisitionParameterComposite):
    """Parameter for most pulse sequences involving electron spin resonance.

    The main attributes of this parameter are

    - ``pulse_sequence``, which contains two pulse sequences ``ESR`` and ``EPR``.
    - ``analyses``, which has an analysis for ESR and EPR.

    Args:
        name: Name of acquisition parameter
        **kwargs: Additional kwargs passed to `AcquisitionParameter`.

    Parameters:
        ESR (ElectronReadoutPulseSequence): Performs ESR pulse sequence
        EPR (PulseSequence): Empty - plunge - read pulse sequence
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.

    Notes:
        - All pulse settings are copies of
          ``ESRParameter.pulse_sequence.pulse_settings``.
        - For given pulse settings, ``ESRParameter.pulse_sequence.generate``
          will recreate the pulse sequence from settings.
    """

    def __init__(
            self,
            name="ESR",
            pulse_sequences=None,
            **kwargs
    ):
        self.pulse_sequence = ESRPulseSequenceComposite(pulse_sequences=pulse_sequences)
        self.ESR = self.pulse_sequence.ESR
        self.EPR = self.pulse_sequence.EPR

        self.analyses = ParameterNode()
        self.analyses.EPR = AnalyseEPR("EPR")
        self.analyses.ESR = AnalyseElectronReadout("ESR")

        self.layout.sample_rate.connect(self.analyses.EPR.settings["sample_rate"])
        self.layout.sample_rate.connect(self.analyses.ESR.settings["sample_rate"])
        self.EPR["enabled"].connect(self.analyses.ESR.outputs["contrast"])
        self.EPR["enabled"].connect(self.analyses.EPR["enabled"])
        self.ESR["enabled"].connect(self.analyses.ESR["enabled"])

        self.analyses.ESR.settings["num_frequencies"].define_get(
            lambda: len(self.ESR.frequencies)
        )
        self.analyses.ESR.settings["samples"].define_get(lambda: self.samples)

        super().__init__(name=name, **kwargs)

    def analyse(self, traces=None, plot=False, plot_high_low=False):
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

        # First analyse EPR because we want to use dark_counts
        if "EPR" in traces:
            results["EPR"] = self.analyses.EPR.analyse(
                empty_traces=traces[self.EPR[0].full_name]["output"],
                plunge_traces=traces[self.EPR[1].full_name]["output"],
                read_traces=traces[self.EPR[2].full_name]["output"],
                plot=plot,
            )
            dark_counts = results["EPR"]["dark_counts"]
        else:
            dark_counts = None

        for name, analysis in self.analyses.parameter_nodes.items():
            if name == 'EPR':
                continue
            if isinstance(analysis, AnalyseElectronReadout):
                results[name] = analysis.analyse(
                    traces=traces[name],
                    dark_counts=dark_counts,
                    plot=plot,
                    plot_high_low=plot_high_low
                )
            else:
                raise SyntaxError(f'Cannot process analysis {name} {type(analysis)}')

        self.results = results
        return results


class NMRParameterComposite(AcquisitionParameterComposite):
    """ Parameter for most measurements involving an NMR pulse.

    Args:
        name: Parameter name
        **kwargs: Additional kwargs passed to `AcquisitionParameter`

    Parameters:
        NMR (ElectronReadoutPulseSequence): Pulse sequence for NMR portion
            of pulse sequence
        ESR (ElectronReadoutPulseSequence): pulse sequence to read out nuclear
            state via electron. Generally the setting ``shots_per_frequency``
            should be larger than 1
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.

    Note:
        - The `NMRPulseSequence` does not have an empty-plunge-read (EPR)
          sequence, and therefore does not add a contrast or dark counts.
          Verifying that the system is in tune is therefore a little bit tricky.

    """

    def __init__(self, name: str = "NMR", pulse_sequence=None, **kwargs):
        if pulse_sequence is None:
            pulse_sequence = NMRPulseSequenceComposite()

        self.pulse_sequence = pulse_sequence
        self.NMR = self.pulse_sequence.NMR
        self.ESR = self.pulse_sequence.ESR

        self.analyses = ParameterNode()
        self.analyses.ESR = AnalyseElectronReadout('ESR')
        self.analyses.NMR_electron_readout = AnalyseElectronReadout('NMR_electron_readout')
        self.analyses.NMR = AnalyseMultiStateReadout('NMR')

        # Turn off NMR_electron_readout analysis
        self.analyses.NMR_electron_readout.enabled = False

        self._connect_analyses()

        super().__init__(name=name, **kwargs)

    def _connect_analyses(self):
        self.layout.sample_rate.connect(self.analyses.ESR.settings["sample_rate"])
        self.layout.sample_rate.connect(self.analyses.NMR_electron_readout.settings["sample_rate"])
        self.analyses.ESR.settings['labels'].connect(
            self.analyses.NMR.settings['labels']
        )
        self.analyses.NMR.settings['labels'].connect(
            self.analyses.ESR.settings['labels']
        )

        self.analyses.ESR.settings["shots_per_frequency"].define_get(
            partial(self.ESR.settings.__getitem__, 'shots_per_frequency')
        )
        self.analyses.NMR_electron_readout.settings["shots_per_frequency"].define_get(
            partial(self.NMR.settings.__getitem__, 'shots_per_frequency')
        )
        self.analyses.ESR.settings["num_frequencies"].define_get(
            lambda: len(self.ESR.frequencies)
        )
        self.analyses.NMR_electron_readout.settings["num_frequencies"].define_get(
            lambda: len(self.NMR.frequencies)
        )
        self.analyses.NMR.settings["num_frequencies"].define_get(
            lambda: len(self.ESR.frequencies)
        )
        self.analyses.ESR.settings["samples"].define_get(
            lambda: self.samples
        )
        self.analyses.NMR_electron_readout.settings["samples"].define_get(
            lambda: self.samples
        )

    def analyse(self, traces: Dict[str, Dict[str, np.ndarray]] = None, plot: bool = False):
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
        # Convert to DotDict so we can access nested pulse sequences
        traces = DotDict(traces)

        self.results = DotDict()

        initialization_sequence = getattr(self.pulse_sequence, 'initialization', None)
        if initialization_sequence is not None and initialization_sequence.enabled:
            # An initialization sequence is added, we need to filter the results
            # based on whether initialization was successful
            self.results["initialization"] = self.analyses.initialization.analyse(
                traces=traces.ESR_initialization, plot=plot
            )
            try:
                filtered_shots = next(
                    val for key, val in self.results["initialization"].items()
                    if key.startswith("filtered_shots")
                )
            except StopIteration:
                logger.warning(
                    "No filtered_shots found, be sure to set "
                    "analyses.initialization.settings.threshold_up_proportion"
                )
                filtered_shots = self.results.initialization.up_proportions > 0.5
        else:
            # Do not use filtered shots
            filtered_shots = None

        self.results["ESR"] = self.analyses.ESR.analyse(
            traces=traces.ESR, plot=plot
        )

        up_proportions_arrs = np.array([
            val for key, val in self.results['ESR'].items()
            if key.startswith('up_proportion') and 'idxs' not in key
        ])
        if self.analyses.NMR.enabled:
            self.results["NMR"] = self.analyses.NMR.analyse(
                up_proportions_arrs=up_proportions_arrs,
                filtered_shots=filtered_shots
            )
        if self.analyses.NMR_electron_readout.enabled:
            self.results["NMR_electron_readout"] = self.analyses.NMR_electron_readout.analyse(
                traces=traces.NMR, filtered_shots=filtered_shots, plot=plot,
                threshold_voltage=self.results['ESR']['threshold_voltage']
            )

        return self.results

    def plot_flips(self , figsize=(8, 3)):
        up_proportion_arrays = [
            val for key, val in self.results.ESR.items()
            if key.startswith('up_proportion') and 'idxs' not in key
        ]
        assert len(up_proportion_arrays) >= 2

        plot = MatPlot(up_proportion_arrays, marker='o', ms=5, linestyle='', figsize=figsize)

        ax = plot[0]
        ax.set_xlim(-0.5, len(up_proportion_arrays[0])-0.5)
        ax.set_ylim(-0.015, 1.015)
        ax.set_xlabel('Shot index')
        ax.set_ylabel('Up proportion')

        # Add threshold lines
        ax.hlines(self.results.NMR.threshold_up_proportion, *ax.get_xlim(), lw=3)
        ax.hlines(self.results.NMR.threshold_low, *ax.get_xlim(), color='grey', linestyle='--', lw=2)
        ax.hlines(self.results.NMR.threshold_high, *ax.get_xlim(), color='grey', linestyle='--', lw=2)

        if 'initialization' in self.results:
            plot.add(self.results.initialization.up_proportions, marker='o', ms=8,  linestyle='', color='grey', zorder=0, alpha=0.5)
            initialization_filtered_shots = next(
                val for key, val in self.results["initialization"].items()
                if key.startswith("filtered_shots")
            )
        else:
            initialization_filtered_shots = [True] * len(self.results.NMR.filtered_shots)

        NMR_filtered_shots = self.results.NMR.filtered_shots
        for k, up_proportion_tuple in enumerate(zip(*up_proportion_arrays)):
            if not initialization_filtered_shots[k]:
                color = 'orange'
            elif not NMR_filtered_shots[k]:
                color = 'red'
            else:
                color = 'green'

            ax.plot([k, k], sorted(up_proportion_tuple)[-2:], color=color, zorder=-1)

        plot.tight_layout()

        # print contrast
        sorted_up_proportions = np.sort(np.array(up_proportion_arrays), axis=0)
        up_proportion = np.mean(sorted_up_proportions[-1])
        dark_counts = np.mean(sorted_up_proportions[:-1])
        contrast = up_proportion - dark_counts
        print(f'Contrast = {up_proportion:.2f} - {dark_counts:.2f} = {contrast:.2f}')
        return plot
