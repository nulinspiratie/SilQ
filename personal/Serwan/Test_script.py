x = 2
# # Imports
# import os
# import clr
# import sys
# from imp import reload
# from System import Array
# from time import sleep, time
# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.misc import factorial
# import peakutils
# # sys.path.append(os.getcwd())
#
# import qcodes as qc
#
# import qcodes.instrument.parameter as parameter
#
# loc_provider = qc.data.location.FormatLocation(fmt='data/{date}/#{counter}_{name}_{time}')
# qc.data.data_set.DataSet.location_provider=loc_provider
#
# import qcodes.instrument_drivers.lecroy.ArbStudio1104 as ArbStudio_driver
# import qcodes.instrument_drivers.spincore.PulseBlasterESRPRO as PulseBlaster_driver
# import qcodes.instrument_drivers.stanford_research.SIM900 as SIM900_driver
# import qcodes.instrument_drivers.AlazarTech.ATS9440 as ATS_driver
# import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ATS_controller_driver
#
#
#
from functools import partial
if __name__ == "__main__":
    def print_function(*args, **kwargs):
        print('args={args}, kwargs={kwargs}'.format(args=args, kwargs=kwargs))


    class AddChannelFunctions:
        def __init__(self, channels, functions):
            self.channels = channels
            self.functions = functions

        def __call__(self, cls):
            def print_function(*args, **kwargs):
                print('args={args}, kwargs={kwargs}'.format(args=args,
                                                            kwargs=kwargs))

            for channel in self.channels:
                for function in self.functions:
                    print_function_targeted = partial(print_function,
                                                      ch=channel,
                                                      function=function)
                    exec("cls.{ch}_{fn} = print_function_targeted".format(
                        ch=str(channel), fn=function))
                    #         cls.ch1_trig_in = print_function
            return cls


    channels = ['ch1', 'ch2', 'ch3', 'ch4']
    functions = ['trigger_source', 'trigger_mode', 'add_waveform', 'sequence']


    @AddChannelFunctions(channels, functions)
    class MockArbStudio:
        def __init__(self):
            pass


    arbstudio = MockArbStudio()

    from silq.instrument_interfaces import \
        get_instrument_interface

    get_instrument_interface(arbstudio)