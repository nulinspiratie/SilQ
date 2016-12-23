import numpy as np

import qcodes as qc
from qcodes.data import hdf5_format, io

from silq.tools import data_tools
from silq.tools.parameter_tools import create_set_vals
from silq.tools.general_tools import SettingsClass, get_truth, \
    clear_single_settings, JSONEncoder


class Condition:
    def _JSONEncoder(self):
        """
        Converts to JSON encoder for saving metadata

        Returns:
            JSON dict
        """
        return JSONEncoder(self)

    @classmethod
    def load_from_dict(cls, load_dict):
        obj = cls()
        for attr, val in load_dict.items():
            if attr == '__class__':
                continue
            setattr(obj, attr, val)
        return obj

class TruthCondition(Condition):
    def __init__(self, attribute=None, relation=None, target_val=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.attribute = attribute
        self.relation = relation
        self.target_val = target_val

    def check_satisfied(self, dataset):
        test_arr = getattr(dataset, self.attribute).ndarray
        # Determine which elements satisfy condition
        satisfied_arr = get_truth(test_arr, self.target_val, self.relation)
        is_satisfied = np.any(satisfied_arr)
        return is_satisfied, satisfied_arr


class ConditionSet:
    def __init__(self, *conditions, on_success=None, on_fail=None):
        self.on_success = on_success
        self.on_fail = on_fail

        self.conditions = []
        for condition in conditions:
            self.add_condition(condition)

    def _JSONEncoder(self):
        """
        Converts to JSON encoder for saving metadata

        Returns:
            JSON dict
        """
        return JSONEncoder(self, ignore_attrs=['loc_provider'])

    @classmethod
    def load_from_dict(cls, load_dict):
        obj = cls()
        for attr, val in load_dict.items():
            if attr == '__class__':
                continue
            elif attr == 'conditions':
                obj.conditions = []
                for condition in val:
                    # Load condition class from globals
                    cls = globals()[condition['__class__']]
                    obj.conditions.append(cls.load_from_dict(condition))
            else:
                setattr(obj, attr, val)
        return obj

    def add_condition(self, condition):
        if isinstance(condition, Condition):
            self.conditions.append(condition)
        elif isinstance(condition, (list, tuple)) and len(condition) == 3:
            self.conditions.append(TruthCondition(*condition))
        else:
            raise Exception('Could not decode condition {}'.format(condition))

    def check_satisfied(self, dataset):
        """
        Checks if a dataset satisfies a set of conditions
        Args:
            dataset:

        Returns:

        """
        # Determine dimensionality from discriminant array
        dims = getattr(dataset, self.discriminant).ndarray.shape

        # Start of with all set points satisfying conditions
        satisfied_arr = np.ones(dims)

        for condition in self.conditions:
            satisfied_single_arr = condition.check_satisfied(dataset)
            # Update satisfied elements with those satisfying current condition
            satisfied_arr = np.logical_and(self.satisfied_arr,
                                           satisfied_single_arr)

        is_satisfied = np.any(satisfied_arr)
        action = self.on_success if is_satisfied else self.on_fail

        return {'is_satisfied': is_satisfied,
                'action': action,
                'satisfied_arr': satisfied_arr}


class Measurement(SettingsClass):
    def __init__(self, name=None, condition_sets=None,
                 acquisition_parameter=None,
                 set_parameters=None, set_vals=None,
                 steps=None, points=None, discriminant=None):
        SettingsClass.__init__(self)

        self.name = name
        self.acquisition_parameter = acquisition_parameter
        self.discriminant = discriminant
        self.steps = steps
        self.points = points
        self.set_parameters = set_parameters
        self._set_vals = set_vals
        self.condition_sets = [] if condition_sets is None else condition_sets
        self.dataset = None
        self.satisfied_arr = None

        self.loc_provider = qc.data.location.FormatLocation(
            fmt='#{counter}_{name}_{time}')

    def __repr__(self):
        return '{} measurement'.format(self.name)

    def __call__(self, *args):
        if len(args) == 0:
            return self.get()
        else:
            self.set(*args)

    def _JSONEncoder(self):
        """
        Converts to JSON encoder for saving metadata

        Returns:
            JSON dict
        """
        return JSONEncoder(self, ignore_attrs=['loc_provider'],
                           ignore_vals=[None, {}, []])

    @classmethod
    def load_from_dict(cls, load_dict):
        obj = cls()
        for attr, val in load_dict.items():
            if attr == '__class__':
                continue
            elif attr == 'condition_sets':
                obj.condition_sets = []
                for condition_set in val:
                    # Load condition class from globals
                    cls = globals()[condition_set['__class__']]
                    obj.conditions.append(cls.load_from_dict(condition_set))
            else:
                setattr(obj, attr, val)
        return obj

    @property
    def disk_io(self):
        return io.DiskIO(data_tools.get_latest_data_folder())

    @property
    def set_vals(self):
        if self._set_vals is not None:
            return self._set_vals
        elif self.points is not None:
            return create_set_vals(set_parameters=self.set_parameters,
                                   steps=self.steps, points=self.points,
                                   silent=True)

    def check_condition_sets(self, *condition_sets):
        condition_sets = condition_sets + self.condition_sets
        for condition_set in condition_sets:
            condition_result = condition_set.check_satisfied(self.dataset)
            if condition_result['action'] is not None:
                break

        condition_result['measurement'] = self.name
        return condition_result

    def get_optimal_val(self, dataset, satisfied_arr=None):
        discriminant_vals = getattr(dataset, self.discriminant)

        # Convert arrays to 1D
        measurement_vals_1D = np.ravel(discriminant_vals)

        if satisfied_arr is not None:
            # Filter 1D arrays by those satisfying conditions
            satisfied_arr_1D = np.ravel(satisfied_arr)
            satisfied_idx = np.nonzero(satisfied_arr_1D)[0]

            measurement_vals_1D = np.take(measurement_vals_1D, satisfied_idx)

        max_idx = np.argmax(measurement_vals_1D)
        self.optimal_set_vals = self.set_vals_from_idx(max_idx)
        self.optimal_val = measurement_vals_1D[max_idx]
        return self.optimal_set_vals, self.optimal_val


class Loop0DMeasurement(Measurement):
    def __init__(self, name=None, acquisition_parameter=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         **kwargs)

    def set_vals_from_idx(self, idx):
        """
        Return set vals that correspond to the acquisition idx.
        In this case it returns an empty dict, since there are no set parameters
        Args:
            idx: Acquisition idx, in this case always zero

        Returns:
            Dict of set vals
        """
        assert idx == 0, "Optimal idx must be zero"
        return {}

    @clear_single_settings
    def get(self):
        """
        Performs a measurement at a single point using qc.Measure
        Returns:
            Dataset
        """
        self.measurement = qc.Measure(self.acquisition_parameter)
        self.dataset = self.measurement.run(
            name='{}_{}'.format(self.name, self.acquisition_parameter.name),
            data_manager=False,
            io=self.disk_io, location=self.loc_provider)

        return self.dataset


class Loop1DMeasurement(Measurement):
    def __init__(self, name=None, set_parameter=None, acquisition_parameter=None,
                 set_vals=None, steps=None, points=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         set_parameters=[set_parameter], set_vals=set_vals,
                         steps=steps, points=points, **kwargs)
        self.set_parameter = set_parameter

    def set_vals_from_idx(self, idx):
        """
        Return set vals that correspond to the acquisition idx.
        Args:
            idx: Acquisition idx

        Returns:
            Dict of set vals (in this case contains one element)
        """
        return {self.set_parameter.name: self.set_vals[idx]}

    @clear_single_settings
    def get(self):
        """
        Performs a 1D measurement loop
        Returns:
            Dataset
        """
        # Set data saving parameters
        self.measurement = qc.Loop(
            self.set_parameter[self.set_vals]).each(
                self.acquisition_parameter)
        self.dataset = self.measurement.run(
            name='{}_{}_{}'.format(self.name, self.set_parameter.name,
                                   self.acquisition_parameter.name),
            background=False, data_manager=False,
            io=self.disk_io, location=self.loc_provider)

        return self.dataset

    def set(self, val):
        self.set_vals = val


class Loop2DMeasurement(Measurement):
    def __init__(self, name=None, set_parameters=None, acquisition_parameter=None,
                 set_vals=None, steps=None, points=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         set_parameters=set_parameters, set_vals=set_vals,
                         steps=steps, points=points, **kwargs)

    def set_vals_from_idx(self, idx):
        """
        Return set vals that correspond to the acquisition idx.
        Args:
            idx: Acquisition idx

        Returns:
            Dict of set vals (in this case contains two elements)
        """
        len_inner = len(self.set_vals[1])
        idxs = (idx // len_inner, idx % len_inner)
        return {self.set_parameters[0].name: self.set_vals[0][idxs[0]],
                self.set_parameters[1].name: self.set_vals[1][idxs[1]]}

    @clear_single_settings
    def get(self):
        """
        Performs a 2D measurement loop
        Returns:
            Dataset
        """
        # Set data saving parameters
        self.measurement = qc.Loop(
            self.set_parameters[0][self.set_vals[0]]).loop(
            self.set_parameters[1][self.set_vals[1]]).each(
            self.acquisition_parameter)

        self.dataset = self.measurement.run(
            name='{}_{}_{}_{}'.format(self.name, self.set_parameters[0].name,
                                      self.set_parameters[1].name,
                                      self.acquisition_parameter.name),
            background=False, data_manager=False,
            io=self.disk_io, location=self.loc_provider)

        return self.dataset

    def set(self, val):
        self.set_vals = val

