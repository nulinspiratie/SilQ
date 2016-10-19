import numpy as np
import inspect

from silq.instrument_interfaces.AlazarTech import ATSInterface

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from qcodes.instrument_drivers.AlazarTech.ATS import AcquisitionController



class BasicATSInterface(ATSInterface):
    """Basic AcquisitionController tested on ATS9440
    """
    def __init__(self, name, instrument_name, **kwargs):
        super().__init__(name, instrument_name, **kwargs)

        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.buffer = None

        self.add_parameter(name='average_mode',
                           parameter_class=ManualParameter,
                           initial_value='trace',
                           vals=vals.Enum('none', 'trace', 'point'))

    def setup(self):
        # Update acquisition parameter values. These depend on the average mode
        self.channel_selection = self.acquisition_setting('channel_selection')
        self.samples_per_record = self.acquisition_setting('samples_per_record')
        self.records_per_buffer = self.acquisition_setting('records_per_buffer')
        self.buffers_per_acquisition = self.acquisition_setting(
            'buffers_per_acquisition')
        self.records_per_acquisition = self.buffers_per_acquisition * \
                                       self.records_per_buffer
        self.number_of_channels = len(self.channel_selection)

        self.acquisition.names = tuple(['Channel_{}_signal'.format(ch) for ch in
                                        self.channel_selection])

        self.acquisition.labels = self.acquisition.names
        self.acquisition.units = ['V' * self.number_of_channels]

        # Set the shape of the output data depending on averaging mode
        if self.average_mode() == 'point':
            self.acquisition.shapes = tuple([()] * self.number_of_channels)
        elif self.average_mode() == 'trace':
            shape = (self.samples_per_record,)
            self.acquisition.shapes = tuple([shape] * self.number_of_channels)
        else: #average_mode is 'none'
            shape = (self.records_per_buffer * self.buffers_per_acquisition,
                     self.samples_per_record)
            self.acquisition.shapes = tuple([shape] * self.number_of_channels)

    def pre_start_capture(self):
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
        if self.buffer_idx < self.buffers_per_acquisition:
            if self.average_mode() in ['point', 'trace']:
                self.buffer += data
            else:
                    self.buffer[self.buffer_idx] = data
        else:
            # print('*'*20+'\nIgnoring extra ATS buffer')
            pass
        self.buffer_idx += 1

    def post_acquire(self):
        # average over records in buffer:
        # for ATS9360 samples are arranged in the buffer as follows:
        # S0A, S0B, ..., S1A, S1B, ...
        # with SXY the sample number X of channel Y.
        # Not sure if this also holds for ATS9440
        # TODO Restructure such that reshaping always happens before averaging

        channel_offset = lambda channel: channel * self.samples_per_record * \
                                         self.records_per_buffer

        if self.average_mode() == 'none':
            # Records is a list, where each element is a 2D array containing
            # all traces obtained in a channel
            records = [
                self.buffer[:, channel_offset(ch):channel_offset(ch+1)].reshape(
                    (self.records_per_acquisition, self.samples_per_record))
                for ch in range(self.number_of_channels)]
        elif self.average_mode() == 'trace':

            records = [np.zeros(self.samples_per_record)
                       for k in range(self.number_of_channels)]

            for channel in range(self.number_of_channels):
                for i in range(self.records_per_buffer):
                    i0 = channel_offset(channel) + i * self.samples_per_record
                    i1 = i0 + self.samples_per_record
                    records[channel] += self.buffer[i0:i1] / self.records_per_acquisition
        elif self.average_mode() == 'point':
            trace_length = self.samples_per_record * self.records_per_buffer
            records = [np.mean(self.buffer[i*trace_length:(i+1)*trace_length])/ self.records_per_acquisition
                       for i in range(self.number_of_channels)]

        # Scale datapoints
        # TODO fix small offset error
        for i, record in enumerate(records):
            channel_range = eval('self.instrument.channel_range{}()'.format(i + 1))
            records[i] = 2 * (record / 2 ** 16 - 0.5) * channel_range
        return records
