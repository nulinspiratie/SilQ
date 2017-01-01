from qcodes import config

from silq.measurements.measurement_types import *
from silq.tools.general_tools import JSONEncoder
measurement_config = config['user'].get('measurement', {})


class MeasurementSequence:
    def __init__(self, name=None, measurements=None, condition_sets=None):
        self.measurements = [] if measurements is None else measurements
        self.name = name
        self.condition_sets = [] if condition_sets is None else condition_sets

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.measurements[index]
        elif isinstance(index, str):
            measurements = [m for m in self.measurements if m.name == index]
            assert len(measurements) == 1, \
                "Found more than one measurement with name {}".format(index)
            return measurements[0]

    def __iter__(self):
        self.measurement = self.measurements[0]
        self.num_measurements = 0

    def __next__(self):
        if self.measurement is None:
            raise StopIteration

        self.num_measurements += 1
        self.measurement()
        result = self.measurement.check_condition_sets(*self.condition_sets)

        if result['action'] is None or result['action'][:4] == 'next':
            self.measurement = self.next_measurement
        else:
            self.measurement = None

        return result

    def __call__(self):
        self.results = [result for result in self]
        result = self.results[-1]
        if result['action'] is None:
            # TODO better way to discern success from fail
            result['action'] = 'success' if result['is_satisfied'] else 'fail'
        elif result['action'][:4] == 'next':
            result['action'] = result['action'][5:]

        #TODO correct return
        return result

    def _JSONEncoder(self):
        """
        Converts to JSON encoder for saving metadata

        Returns:
            JSON dict
        """
        return JSONEncoder(self, ignore_vals=[None, {}, []])

    @classmethod
    def load_from_dict(cls, load_dict):
        obj = cls()
        for attr, val in load_dict.items():
            if attr == '__class__':
                continue
            elif attr in ['condition_sets', 'measurements']:
                setattr(obj, attr, [])
                obj_attr = getattr(obj, attr)
                for elem_dict in val:
                    # Load condition class from globals
                    cls = globals()[elem_dict['__class__']]
                    obj_attr.append(cls.load_from_dict(elem_dict))
            else:
                setattr(obj, attr, val)
        return obj

    def save_to_config(self, name=None):
        if name is None:
            name = self.name
        measurement_config[name] = self._JSONEncoder()

    @property
    def next_measurement(self):
        """ Get measurement after self.measurement. In case there is no next
        measurement, returns None """
        if self.measurement is None:
            return None
        msmt_idx = self.measurements.index(self.measurement)
        if msmt_idx + 1 == len(self.measurements):
            # No next measurement
            return None
        else:
            return self.measurements[msmt_idx + 1]
