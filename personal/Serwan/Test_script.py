
if __name__ == "__main__":
    # import qcodes as qc
    # from qcodes import ManualParameter
    #
    # dummy_parameter1 = ManualParameter(name='dummy1')
    # dummy_parameter2 = ManualParameter(name='dummy2')
    #
    # loop = qc.Loop(dummy_parameter1[0:10:1]).loop(
    #     dummy_parameter2[0:10:1]).each(dummy_parameter2)
    # loop.run()
    import numpy as np
    import qcodes as qc

    class TestParameter(qc.ArrayParameter):
        def __init__(self, shape=(5,), **kwargs):
            super().__init__(name='test',
                             shape=shape,
                             setpoint_names=('setpoints',),
                             setpoints=(list(np.linspace(0, 1, shape[0])),),
                             **kwargs)

        def get(self):
            return np.random.randint(0, 10, size=self.shape)

    test_parameter = TestParameter()

    qc.Measure(test_parameter).run('test')
    # class TestParameter(MultiParameter):
    #     def __init__(self, name, shape=(5,), **kwargs):
    #         super().__init__(name=name, names=['DC_voltage'], shapes=(shape,),
    #                          setpoint_names=(('setpoints',),),
    #                          setpoints=((np.linspace(0, 1, shape[0]),),),
    #                          **kwargs)
    #
    #         self.vals = np.random.randint(0, 10, size=shape)
    #
    #     def setup(self):
    #         self.setpoints
    #
    #     def get():
    #         return self.vals,

