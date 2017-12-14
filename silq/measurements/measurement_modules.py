import logging
import qcodes as qc

from silq.tools.general_tools import JSONEncoder

__all__ = ['MeasurementSequence']

logger = logging.getLogger(__name__)

measurement_config = qc.config['user'].get('measurement', {})


class MeasurementSequence:
    def __init__(self, name=None, measurements=None, condition_sets=None,
                 set_parameters=None, acquisition_parameter=None,
                 silent=True, set_active=False, continuous=False,
                 base_folder=None):
        self.set_parameters = set_parameters
        self.acquisition_parameter = acquisition_parameter

        self.measurements = [] if measurements is None else measurements
        for measurement in measurements:
            if measurement.acquisition_parameter is None:
                measurement.acquisition_parameter = acquisition_parameter

        self.name = name
        self.condition_sets = [] if condition_sets is None else condition_sets

        self.silent = silent
        self.set_active = set_active
        self.continuous = continuous
        self.base_folder = base_folder

        self.optimal_set_vals = None
        self.optimal_val = None

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
            logger.debug('Finished measurements')
            raise StopIteration
        else:
            self.measurement = self.next_measurement
        self.measurement.silent = self.silent
        self.measurement.base_folder = self.base_folder

        # Perfom measurement
        self.num_measurements += 1
        self.measurement.silent = self.silent
        # Performing measurement also checks for condition sets, and updates
        # set parameters accordingly
        self.measurement.single_settings(condition_sets=self.condition_sets)
        logger.info(f'Performing measurement {self.measurement}')
        dataset, self.result = self.measurement.get(set_active=self.set_active)
        self.datasets.append(dataset)

        # Return result of the final condition set
        # Either this was the first successful condition, or if none were
        # successful, this would be the final condition set
        return self.result

    def __call__(self):
        if self.continuous:
            self.acquisition_parameter.temporary_settings(continuous=True)

        try:
            # Perform measurements iteratively, collecting their results
            self.results = [result for result in self]
            logger.info(f'Consecutive measurement actions: '
                        f'{[result["action"] for result in self.results]}')
        finally:
            self.acquisition_parameter.layout.stop()
            # Clear settings such as continuous=True
            self.acquisition_parameter.clear_settings()

        # Choose last measurement result
        result = self.results[-1]
        if result['action'] is None:
            # TODO better way to discern success from fail
            result['action'] = 'success' if result['is_satisfied'] else 'fail'
            logger.info(f'Final action is None,  but since "is_satisfied" is '
                        f'{result["is_satisfied"]}, '
                        f'action is {result["action"]}')
        elif result['action'][:4] == 'next':
            # Action is of form 'next_{cmd}, meaning that if there is a next
            # measurement, it would have performed that measurement.
            # However, since this is the last measurement, action should be cmd.
            logger.info(f"Final action is {result['action']}, so this becomes "
                        f"{result['action'][5:]}")
            result['action'] = result['action'][5:]

        # Optimal vals
        self.optimal_set_vals, self.optimal_val = self.measurement.get_optimum()

        # TODO correct return
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
