import numpy as np
from unittest import TestCase

from qcodes import Loop, Parameter, load_data, ParameterNode

from silq.tools.loop_tools import Measurement, Sweep



class TestOldLoop(TestCase):
    def setUp(self) -> None:
        self.p_sweep = Parameter('p_sweep', set_cmd=None)
        self.p_measure = Parameter('p_measure', set_cmd=None)
        self.p_sweep.connect(self.p_measure)

    def test_old_loop_1D(self):
        loop = Loop(self.p_sweep.sweep(0, 10, 1)).each(
            self.p_measure,
            self.p_measure
        )
        loop.run(name='old_loop_1D', thread=False)

    def test_old_loop_2D(self):
        self.p_sweep2 = Parameter('p_sweep2', set_cmd=None)

        loop = Loop(self.p_sweep.sweep(0, 5, 1)).loop(
            self.p_sweep2.sweep(0, 5, 1)
        ).each(
            self.p_measure,
            self.p_measure
        )
        loop.run(name='old_loop_2D', thread=False)

    def test_old_loop_1D_2D(self):
        self.p_sweep2 = Parameter('p_sweep2', set_cmd=None)

        loop = Loop(self.p_sweep.sweep(0, 5, 1)).each(
            self.p_measure,
            Loop(self.p_sweep2.sweep(0, 5, 1)).each(
            self.p_measure
            )
        )
        loop.run(name='old_loop_1D_2D', thread=False)


def verify_msmt(msmt, verification_arrays=None):
    dataset = load_data(msmt.dataset.location)

    for action_indices, data_array in msmt.data_arrays.items():
        if verification_arrays is not None:
            verification_array = verification_arrays[action_indices]
            np.testing.assert_array_almost_equal(data_array, verification_array)

        dataset_array = dataset.arrays[data_array.array_id]
        np.testing.assert_array_almost_equal(data_array, dataset_array)


class TestNewLoop(TestCase):
    def setUp(self) -> None:
        self.p_sweep = Parameter('p_sweep', set_cmd=None, initial_value=10)
        self.p_measure = Parameter('p_measure', set_cmd=None)
        self.p_sweep.connect(self.p_measure, scale=10)

    def test_new_loop_1D(self):
        arrs = {}

        with Measurement('new_loop_1D') as msmt:
            for k, val in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.1))):
                arr = arrs.setdefault(msmt.action_indices, np.zeros(msmt.loop_dimensions))
                arr[k] = msmt.measure(self.p_measure)

        verify_msmt(msmt, arrs)

    def test_new_loop_1D_double(self):
        arrs = {}

        with Measurement('new_loop_1D_double') as msmt:
            for k, val in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.1))):
                arr = arrs.setdefault(msmt.action_indices,
                                      np.zeros(msmt.loop_dimensions))
                arr[k] = msmt.measure(self.p_measure)

                arr = arrs.setdefault(msmt.action_indices,
                                      np.zeros(msmt.loop_dimensions))
                arr[k] = msmt.measure(self.p_measure)

        verify_msmt(msmt, arrs)

    def test_new_loop_2D(self):
        arrs = {}
        self.p_sweep2 = Parameter('p_sweep2', set_cmd=None)

        with Measurement('new_loop_1D_double') as msmt:
            for k, val in enumerate(Sweep(self.p_sweep2.sweep(0, 1, 0.1))):
                for kk, val2 in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.1))):
                    self.assertEqual(msmt.loop_dimensions, (11,11))
                    arr = arrs.setdefault(msmt.action_indices,
                                          np.zeros(msmt.loop_dimensions))
                    arr[k,kk] = msmt.measure(self.p_measure)

        verify_msmt(msmt, arrs)

    def test_new_loop_0D(self):
        # Does not work yet
        with Measurement('new_loop_0D') as msmt:
            self.assertEqual(msmt.loop_dimensions, ())
            msmt.measure(self.p_measure)

        # self.verify_msmt(msmt, arrs)


class TestNewLoopDictResults(TestCase):
    class MeasurableNode(ParameterNode):
        def get(self):
            return {'result1': np.random.rand(),
                    'result2': np.random.rand()}

    def test_measure_node(self):
        arrs = {}
        node = self.MeasurableNode('measurable_node')
        p_sweep = Parameter('sweep', set_cmd=None)

        with Measurement('measure_node') as msmt:
            for k, val in enumerate(Sweep(p_sweep.sweep(0, 1, 0.1))):
                results = msmt.measure(node)

                # Save results to verification arrays
                for kk, result in enumerate(results.values()):
                    arrs.setdefault((0, 0, kk), np.zeros(msmt.loop_dimensions))
                    arrs[(0, 0, kk)][k] = result

        verify_msmt(msmt, arrs)

    def measurable_function(self):
        return {'result1': np.random.rand(),
                'result2': np.random.rand()}

    def test_measure_function(self):
        arrs = {}
        p_sweep = Parameter('sweep', set_cmd=None)

        with Measurement('measure_node') as msmt:
            for k, val in enumerate(Sweep(p_sweep.sweep(0, 1, 0.1))):
                results = msmt.measure(self.measurable_function)

                # Save results to verification arrays
                for kk, result in enumerate(results.values()):
                    arrs.setdefault((0, 0, kk), np.zeros(msmt.loop_dimensions))
                    arrs[(0, 0, kk)][k] = result

        verify_msmt(msmt, arrs)


class TestNewLoopArray(TestCase):
    def setUp(self) -> None:
        self.p_sweep = Parameter('p_sweep', set_cmd=None, initial_value=10)
        self.p_measure = Parameter('p_measure', get_cmd=lambda: np.random.rand(5))

    def test_measure_parameter_array(self):
        arrs = {}

        with Measurement('new_loop_1D_array') as msmt:
            for k, val in enumerate(Sweep(self.p_sweep.sweep(0, 1, 0.1))):
                arr = arrs.setdefault(msmt.action_indices,
                                      np.zeros(msmt.loop_dimensions) + (5,))
                arr[k] = msmt.measure(self.p_measure)

        verify_msmt(msmt, arrs)