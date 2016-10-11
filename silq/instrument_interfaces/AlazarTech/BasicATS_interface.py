import numpy as np
import inspect

from silq.instrument_interfaces import InstrumentInterface

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from qcodes.instrument_drivers.AlazarTech.ATS import AcquisitionController


class BasicATSInterface(InstrumentInterface, AcquisitionController):
    """Basic AcquisitionController tested on ATS9360
    returns unprocessed data averaged by record with 2 channels
    """
    def __init__(self, name, instrument_name, **kwargs):
        InstrumentInterface.__init__(name, instrument_name, **kwargs)

        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.buffer = None

        self.acquisitionkwargs = {}
        # Obtain a list of all valid ATS acquisition kwargs
        self._acquisitionkwargs_names = inspect.signature(
            self.instrument.acquire).parameters.keys()

        self.add_parameter(name='average_mode',
                           parameter_class=ManualParameter,
                           initial_value='trace',
                           vals=vals.Enum('none', 'trace', 'point'))
        # Names and shapes must have initial value, even through they will be
        # overwritten in set_acquisitionkwargs. If we don't do this, the
        # remoteInstrument will not recognize that it returns multiple values.
        self.add_parameter(name="acquisition",
                           names=['channel_signal'],
                           get_cmd=self.do_acquisition,
                           shapes=((),),
                           snapshot_value=False)

    def get_acquisition_kwarg(self, kwarg):
        """
        Obtain an acquisition kwarg for the ATS.
        It first checks if the kwarg is an actual ATS acquisition kwarg, and
        raises an error otherwise.
        It then checks if the kwarg is in ATS_controller._acquisitionkwargs.
        If not, it will retrieve the ATS latest parameter value

        Args:
            kwarg: acquisition kwarg to look for

        Returns:
            Value of the acquisition kwarg
        """
        assert kwarg in self._acquisitionkwargs_names, \
            "Kwarg {} is not a valid ATS acquisition kwarg".format(kwarg)
        if kwarg in self.acquisitionkwargs.keys():
            return self.acquisitionkwargs[kwarg]
        else:
            # Must get latest value, since it may not be updated in ATS
            return self.instrument.parameters[kwarg].get_latest()

    def update_acquisitionkwargs(self, **kwargs):
        self.acquisitionkwargs.update(**kwargs)

        # Update acquisition parameter values. These depend on the average mode
        channel_selection = self.get_acquisition_kwarg('channel_selection')
        samples_per_record = self.get_acquisition_kwarg('samples_per_record')
        records_per_buffer = self.get_acquisition_kwarg('records_per_buffer')
        buffers_per_acquisition = self.get_acquisition_kwarg('buffers_per_acquisition')
        self.acquisition.names = tuple(['Channel_{}_signal'.format(ch) for ch in
                                        self.get_acquisition_kwarg('channel_selection')])

        self.acquisition.labels = self.acquisition.names
        self.acquisition.units = ['V'*len(channel_selection)]

        if self.average_mode() == 'point':
            self.acquisition.shapes = tuple([()]*len(channel_selection))
        elif self.average_mode() == 'trace':
            shape = (samples_per_record,)
            self.acquisition.shapes = tuple([shape] * len(channel_selection))
        else:
            shape = (records_per_buffer * buffers_per_acquisition, samples_per_record)
            self.acquisition.shapes = tuple([shape] * len(channel_selection))

    def pre_start_capture(self):
        self.samples_per_record = self.instrument.samples_per_record()
        self.records_per_buffer = self.instrument.records_per_buffer()
        self.buffers_per_acquisition = self.instrument.buffers_per_acquisition()
        self.number_of_channels = len(self.instrument.channel_selection())
        self.buffer_idx = 0
        if self.average_mode() in ['point', 'trace']:
            self.buffer = np.zeros(self.samples_per_record *
                                  self.records_per_buffer *
                                  self.number_of_channels)
        else:
            self.buffer = np.zeros((self.buffers_per_acquisition,
                                    self.samples_per_record *
                                    self.records_per_buffer *
                                    self.number_of_channels))

    def pre_acquire(self):
        # gets called after 'AlazarStartCapture'
        pass

    def do_acquisition(self):
        records = self.instrument.acquire(acquisition_controller=self,
                                      **self.acquisitionkwargs)
        return records

    def handle_buffer(self, data):
        print('ADDING BUFFER')
        if self.buffer_idx < self.buffers_per_acquisition:
            if self.average_mode() in ['point', 'trace']:
                self.buffer += data
            else:
                    self.buffer[self.buffer_idx] = data
        else:
            pass
            # print('*'*20+'\nIgnoring extra ATS buffer')
        self.buffer_idx += 1

    def post_acquire(self):
        # average over records in buffer:
        # for ATS9360 samples are arranged in the buffer as follows:
        # S0A, S0B, ..., S1A, S1B, ...
        # with SXY the sample number X of channel Y.
        records_per_acquisition = self.buffers_per_acquisition * self.records_per_buffer
        channel_offset = lambda channel: channel * self.samples_per_record * self.records_per_buffer

        if self.average_mode() == 'none':
            records = [self.buffer[:, channel_offset(ch):channel_offset(ch+1)
                                  ].reshape((records_per_acquisition,
                                             self.samples_per_record))
                       for ch in range(self.number_of_channels)]
        elif self.average_mode() == 'trace':
            records = [np.zeros(self.samples_per_record) for k in range(self.number_of_channels)]

            for channel in range(self.number_of_channels):
                for i in range(self.records_per_buffer):
                    i0 = channel_offset(channel) + i * self.samples_per_record
                    i1 = i0 + self.samples_per_record
                    records[channel] += self.buffer[i0:i1] / records_per_acquisition
        elif self.average_mode() == 'point':
            trace_length = self.samples_per_record * self.records_per_buffer
            records = [np.mean(self.buffer[i*trace_length:(i+1)*trace_length])/ records_per_acquisition
                       for i in range(self.number_of_channels)]

        # Scale datapoints
        for i, record in enumerate(records):
            channel_range = eval('self.instrument.channel_range{}()'.format(i + 1))
            records[i] = 2 * (record / 2 ** 16 - 0.5) * channel_range
        return records
