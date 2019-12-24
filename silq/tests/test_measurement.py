import numpy as np
from unittest import TestCase

from qcodes import Loop, Parameter

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


class TestNewLoop(TestCase):
    def setUp(self) -> None:
        self.p_sweep = Parameter('p_sweep', set_cmd=None, initial_value=10)
        self.p_measure = Parameter('p_measure', set_cmd=None)
        self.p_sweep.connect(self.p_measure, scale=10)

    def test_new_loop_1D(self):
        with Measurement('new_loop_1D') as msmt:
            for k in Sweep(self.p_sweep.sweep(0, 1, 0.1)):
                msmt.measure(self.p_measure)

    def test_new_loop_1D_double(self):
        with Measurement('new_loop_1D_double') as msmt:
            for k in Sweep(self.p_sweep.sweep(0, 1, 0.1)):
                msmt.measure(self.p_measure)
                msmt.measure(self.p_measure)
        msmt.dataset

    def test_new_loop_2D(self):
        self.p_sweep2 = Parameter('p_sweep2', set_cmd=None)

        with Measurement('new_loop_1D_double') as msmt:
            for k in Sweep(self.p_sweep2.sweep(0, 1, 0.1)):
                for kk in Sweep(self.p_sweep.sweep(0, 1, 0.1)):
                    val = msmt.measure(self.p_measure)
                    val
        msmt.dataset
