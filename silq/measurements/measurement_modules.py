from qcodes import config

from silq.measurements.measurement_types import *
from silq.tools.general_tools import JSONEncoder
measurement_config = config['user'].get('measurement', {})


class MeasurementSequence:
    def __init__(self, name=None, measurements=None, condition_sets=None,
                 set_parameters=None, acquisition_parameter=None,
                 silent=True):
        self.set_parameters = set_parameters
        self.acquisition_parameter = acquisition_parameter

        self.measurements = [] if measurements is None else measurements
        for measurement in measurements:
            if measurement.acquisition_parameter is None:
                measurement.acquisition_parameter = acquisition_parameter

        self.name = name
        self.condition_sets = [] if condition_sets is None else condition_sets

        self.silent = silent

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.measurements[index]
        elif isinstance(index, str):
            measurements = [m for m in self.measurements if m.name == index]
            assert len(measurements) == 1, \
                "Found more than one measurement with name {}".format(index)
            return measurements[0]

    def __iter__(self):
        self.measurement = None
        self.num_measurements = 0
        self.datasets = []
        return self

    def __next__(self):
        if self.next_measurement is None:
            if not self.silent:
                print('Finished measurements')
            raise StopIteration
        else:
            self.measurement = self.next_measurement
        self.measurement.silent = self.silent

        # Perfom measurement
        self.num_measurements += 1
        if not self.silent:
            print('Performing measurement {}'.format(self.measurement))
        dataset = self.measurement()
        self.datasets.append(dataset)

        # Check condition sets and update parameters accordingly
        condition_set = self.measurement.check_condition_sets(
            *self.condition_sets)
        self.measurement.update_set_parameters()

        # Return result of the final condition set
        # Either this was the first successful condition, or if none were
        # successful, this would be the final condition set
        self.result = condition_set.result
        return self.result

    def __call__(self):
        # Perform measurements iteratively, collecting their results
        self.results = [result for result in self]
        # Choose last measurement result
        result = self.results[-1]
        if result['action'] is None:
            # TODO better way to discern success from fail
            result['action'] = 'success' if result['is_satisfied'] else 'fail'
        elif result['action'][:4] == 'next':
            # Action is of form 'next_{cmd}, meaning that if there is a next
            # measurement, it would have performed that measurement.
            # However, since this is the last measurement, action should be cmd.
            result['action'] = result['action'][5:]

        # Optimal vals
        self.optimal_set_vals, self.optimal_val = self.measurement.get_optimum()

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

        station = qc.station.Station.default
        if isinstance(obj.acquisition_parameter, str):
            obj.acquisition_parameter = getattr(station,
                                                obj.acquisition_parameter)
        obj.set_parameters = [parameter if type(parameter) != str
                              else getattr(station, parameter)
                              for parameter in obj.set_parameters]
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
            return self.measurements[0]
        elif self.result['action'] is not None and \
                        self.result['action'][:4] != 'next':
            return None
        else:
            msmt_idx = self.measurements.index(self.measurement)
            if msmt_idx + 1 == len(self.measurements):
                # No next measurement
                return None
            else:
                return self.measurements[msmt_idx + 1]
