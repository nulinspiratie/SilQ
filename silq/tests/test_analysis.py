import logging
import unittest
from copy import copy, deepcopy
import pickle

from silq.analysis.analysis import AnalyseElectronReadout, AnalyseFlips


class TestAnalysis(unittest.TestCase):
    def test_analyse_electron_readout(self):
        ESR = AnalyseElectronReadout('ESR')

        ESR.settings.num_frequencies = 1
        ESR.settings.shots_per_frequency = 1

        self.assertTupleEqual(
            ESR.names, ('up_proportion', 'num_traces', 'voltage_difference', 'threshold_voltage')
        )
        self.assertTupleEqual(ESR.units, ('', '', 'V', 'V'))
        self.assertTupleEqual(ESR.shapes, ((), (), (), ()))

        ESR.settings.num_frequencies = 2

        self.assertTupleEqual(
            ESR.names, (
                'up_proportion0',
                'up_proportion1',
                'num_traces0',
                'num_traces1',
                'voltage_difference',
                'threshold_voltage')
        )
        self.assertTupleEqual(ESR.units, ('', '', '', '', 'V', 'V'))
        self.assertTupleEqual(ESR.shapes, ((), (), (), (), (), ()))

        ESR.settings.shots_per_frequency = 3
        ESR.settings.samples = 5

        self.assertTupleEqual(
            ESR.names, (
                'up_proportion0',
                'up_proportion1',
                'num_traces0',
                'num_traces1',
                'voltage_difference',
                'threshold_voltage')
        )
        self.assertTupleEqual(ESR.units, ('', '', '', '', 'V', 'V'))
        self.assertTupleEqual(ESR.shapes, ((5,), (5,), (), (), (), ()))

    def test_analyse_flips(self):
        analysis = AnalyseFlips('ESR')

        print(analysis.result_parameters)
        print(analysis.names)
        print(analysis.units)
        print(analysis.shapes)
