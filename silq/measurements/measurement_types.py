import numpy as np
import logging
from functools import partial
import warnings

import qcodes as qc
from qcodes import BreakIf
from qcodes.instrument.parameter import _BaseParameter
from qcodes.data.data_set import DataSet

from silq.tools.parameter_tools import create_set_vals
from silq.tools.general_tools import SettingsClass, get_truth, \
    clear_single_settings, JSONEncoder

__all__ = ['Condition', 'TruthCondition', 'ConditionSet', 'Measurement',
           'Loop0DMeasurement', 'Loop1DMeasurement', 'Loop2DMeasurement']

logger = logging.getLogger(__name__)

_dummy_parameter = qc.Parameter(name='msmt_idx',
                                set_cmd=None,
                                label='Measurement idx')

class Condition:
    def __init__(self, **kwargs):
        pass

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

    def check_satisfied(self, data):
        if isinstance(data, DataSet):
            test_arr = getattr(data, self.attribute).ndarray
        else:
            test_arr = np.array([data[self.attribute]])
        # Determine which elements satisfy condition
        with warnings.catch_warnings():
            # Suppress warnings when comparing to NaN
            warnings.simplefilter("ignore", category=RuntimeWarning)
            satisfied_arr = get_truth(test_arr, self.target_val, self.relation)

        is_satisfied = np.any(satisfied_arr)
        return is_satisfied, satisfied_arr

    def __repr__(self):
        return f'({self.attribute} {self.relation} {self.target_val})'


    class ModCondition(Condition):
        def __init__(self, num, start=False, **kwargs):
            super().__init__(**kwargs)
            self.num = num

            if start:
                self.idx = 0
            else:
                self.idx = 1

        def check_satisfied(self, *args, **kwargs):
            return (not self.idx % self.num),

        def __repr__(self):
            return f'(idx: {self.idx} % {self.num} == 0)'


class ConditionSet:
    """
    A ConditionSet represents a set of conditions that a dataset can be
    tested against. The ConditionSet also contains information on what action
    should be performed if the dataset satisfies the conditions (success) or
    does not (fail). These actions can then be performed by a
    MeasurementSequence. Possible actions are:

        :'success': Finish measurement sequence successfully
        :'fail': Finish measurement sequence unsuccessfully
        :'next_{cmd}': Go to next measurement if it exists, else it is cmd,
            where cmd can be either 'success' or 'fail'.
        :None: Go to next measurement if it exists. If there is no next
            measurement, the action is 'success' if the last measurement
            satisfies the condition_set, else 'fail'. Note that this is not a
            string.

    Parameters:
        on_success (str): action to perform if some points satisfy conditions.
        on_fail (str): action to perform if no points satisfy conditions.
        update (bool): Values should be updated if dataset satisfies conditions.
        result (dict): result after testing a dataset for conditions.
            items are:

            :is_satisfied (bool): Dataset has points that satisfy conditions
            :action (str): action to perform, taken from
                self.on_success if is_satisfied, else from self.on_fail.
            :satisfied_arr (bool arr): array of dataset dimensions,
                where each element indicates if that value satisfies
                conditions.
    """
    def __init__(self, *conditions, on_success=None, on_fail=None,
                 update=False):
        self.on_success = on_success
        self.on_fail = on_fail
        self.update = update
        self.result = None

        self.conditions = []
        for condition in conditions:
            self.add_condition(condition)

    def __repr__(self):
        conditions = [repr(condition) for condition in self.conditions]
        if len(conditions) > 1:
            conditions_str = f'[{", ".join(conditions)}]'
        else:
            conditions_str = f' {conditions[0]}'
        return f'ConditionSet{conditions_str}, update: {self.update}'

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
                # Ignore attr since it is used to determine class
                continue
            elif attr == 'conditions':
                obj.conditions = []
                for condition in val:
                    # Load condition class from globals
                    condition_cls = globals()[condition['__class__']]
                    obj.conditions.append(
                        condition_cls.load_from_dict(condition))
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

    def add_condition(self, condition):
        if isinstance(condition, Condition):
            self.conditions.append(condition)
        elif isinstance(condition, (list, tuple)) and len(condition) == 3:
            self.conditions.append(TruthCondition(*condition))
        else:
            raise Exception(f'Could not decode condition {condition}')

    def check_satisfied(self, data):
        """
        Checks if a dataset satisfies a set of conditions

        Args:
            dataset: Dataset to check against conditions

        Returns:
            Dict[str, Any]: Dictionary containing:

            :is_satisfied (bool): If the conditions are satisfied
            :action (string): Action to perform
            :satisfied_arr (bool arr): array where each element corresponds to a
                combination of set vals, and whose value specifies if those
                set_vals satisfies conditions
        """
        # Determine dimensionality from attribute of first condition
        attr = self.conditions[0].attribute

        if isinstance(data, DataSet):
            dims = getattr(data, attr).ndarray.shape
        else:
            dims = (1, )

        # Start of with all set points satisfying conditions
        satisfied_arr = np.ones(dims)

        for condition in self.conditions:
            _, satisfied_single_arr = condition.check_satisfied(data)
            # Update satisfied elements with those satisfying current condition
            satisfied_arr = np.logical_and(satisfied_arr,
                                           satisfied_single_arr)

        is_satisfied = np.any(satisfied_arr)
        action = self.on_success if is_satisfied else self.on_fail

        self.result = {'is_satisfied': is_satisfied,
                       'action': action,
                       'satisfied_arr': satisfied_arr}

        return self.result


class Measurement(SettingsClass):
    def __init__(self, name=None, base_folder=None, condition_sets=None,
                 acquisition_parameter=None,
                 set_parameters=None, set_vals=None, step=None,
                 step_percentage=None, points=None,
                 discriminant=None, silent=True,
                 break_if=False):
        # Initialize SettingsClass, specifying that if self.condition_sets is
        # not None, using single_settings or temporary_settings won't change
        # its value.
        SettingsClass.__init__(self, ignore_if_set=['condition_sets'])

        self.name = name
        self.base_folder = base_folder
        self.acquisition_parameter = acquisition_parameter
        self.discriminant = discriminant
        self.step = step
        self.step_percentage = step_percentage
        self.points = points
        self.set_parameters = set_parameters
        self.set_vals = set_vals
        self.condition_sets = [] if condition_sets is None else condition_sets
        self.dataset = None
        self.silent = silent
        self.initial_set_vals = None
        self.break_if = break_if
        self.measurement = None

    def __repr__(self):
        return f'{self.name} measurement'

    def __call__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return self.get()
        else:
            self.set(*args, **kwargs)

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
                    condition_cls = globals()[condition_set['__class__']]
                    obj.conditions.append(
                        condition_cls.load_from_dict(condition_set))
            else:
                setattr(obj, attr, val)
        return obj

    @property
    def loc_provider(self):
        if self.base_folder is None:
            fmt = '{date}/#{counter}_{name}_{time}'
        else:
            fmt = self.base_folder + '/#{counter}_{name}_{time}'
        return qc.data.location.FormatLocation(fmt=fmt)

    @property
    def set_vals(self):
        if self._set_vals is not None:
            return self._set_vals
        elif self.points is not None and (self.step is not None
                                          or self.step_percentage is not None):
            self._set_vals = create_set_vals(
                set_parameters=self.set_parameters, step=self.step,
                step_percentage=self.step_percentage, points=self.points,
                silent=True)
            return self._set_vals
        else:
            return None

    @set_vals.setter
    def set_vals(self, set_vals):
        self._set_vals = set_vals
        if set_vals is not None:
            self.set_parameters = [set_val.parameter for set_val in set_vals]

    def check_condition_sets(self, data, *condition_sets):
        """
        Tests dataset for condition sets.
        Condition sets are tested until the result of a condition set has an
        'action' key that is not equal to None. After this, self.condition_set
        is updated to this condition set. If no condition sets have an action,
        self.condition_set will equal the last condition set
        Args:
            *condition_sets: condition sets to be tested, to be tested before
            self.condition_sets

        Returns:
            self.condition_set: condition set that has an 'action', or the
            last condition set if none have an action.
        """

        if not condition_sets:
            logger.warning(f'No condition sets provided')
            return None

        if isinstance(data, _BaseParameter):
            data = data.results

        for condition_set in condition_sets:
            condition_set.check_satisfied(data)
            if not self.silent:
                logger.debug(f'{condition_set} satisfied: '
                             f'{condition_set.result["is_satisfied"]}, '
                             f'action: {condition_set.result["action"]}')

            if condition_set.result['action'] is not None:
                break

        logger.info(f'Using condition set {condition_set}, '
                    f'is_satisfied: {condition_set.result["is_satisfied"]}, '
                    f'action: {condition_set.result["action"]}')

        condition_set.result['measurement'] = self.name
        return condition_set

    def satisfies_condition_set(self, data, action=None):
        condition_set = self.check_condition_sets(data, *self.condition_sets)
        if not condition_set.result['is_satisfied']:
            return False
        elif action is not None:
            return condition_set.result['action'] == action
        else:
            return True

    def get_optimum(self, dataset=None, condition_set=None):
        """
        Get the optimal value from the possible set vals.
        If satisfied_arr is not provided, it will first filter the set vals
        such that only those that satisfied self.condition_sets are satisfied.

        Args:
            dataset (Optional): Dataset to test. Default is self.dataset

        Returns:
            self.optimal_set_vals (dict): Optimal set val for each set parameter
                The key is the name of the set parameter. Returns None if no
                set vals satisfy condition_set.
            self.optimal_val (val): Discriminant value at optimal set vals.
                Returns None if no set vals satisfy condition_set
        """
        if dataset is None:
            dataset = self.dataset

        discriminant_vals = getattr(dataset, self.discriminant)

        # Convert arrays to 1D
        measurement_vals_1D = np.ravel(discriminant_vals)

        if condition_set is not None:
            if not condition_set.result['is_satisfied']:
                # No values satisfy condition sets
                self.optimal_set_vals = self.set_vals_from_idx(-1)
                self.optimal_val = np.nan
                return self.optimal_set_vals, self.optimal_val

            # Filter 1D arrays by those satisfying conditions
            satisfied_arr_1D = np.ravel(
                condition_set.result['satisfied_arr'])
            satisfied_idx, = np.nonzero(satisfied_arr_1D)

            measurement_vals_1D = np.take(measurement_vals_1D, satisfied_idx)
            max_idx = satisfied_idx[np.nanargmax(measurement_vals_1D)]
        else:
            max_idx = np.nanargmax(measurement_vals_1D)

        # TODO more adaptive way of choosing best value, not always max val
        self.optimal_val = np.nanmax(measurement_vals_1D)
        self.optimal_set_vals = self.set_vals_from_idx(max_idx)

        logger.info(f'Optimal value: {self.optimal_val:.5f}, '
                    f'set values: {self.optimal_set_vals}')

        return self.optimal_set_vals, self.optimal_val

    def update_set_parameters(self, condition_set):
        # Determine if values need to be updated
        if condition_set is None:
            update = False
        elif condition_set.result['is_satisfied']:
            update = condition_set.update
        else:
            update = False

        if update:
            logger.info('Updating set parameters to optimal values: '
                       f'{self.optimal_set_vals}')
            for set_parameter in self.set_parameters:
                set_parameter(self.optimal_set_vals[set_parameter.name])
        else:
            logger.info('Resetting set parameters to initial values: '
                         f'{self.initial_set_vals}')
            for set_parameter in self.set_parameters:
                set_parameter(self.initial_set_vals[set_parameter.name])

    def initialize_measurement(self):
        raise NotImplementedError('Must be implemented in subclass')

    def initialize(self):
        for condition_set in self.condition_sets:
            condition_set.result = None

        if self.points is not None and (self.step is not None or
                                        self.step_percentage is not None):
            # Reset set_vals if step and points is given
            self._set_vals = None

        self.initial_set_vals = {
            p.name: p() for p in self.set_parameters}
        logger.info(f'Initial set values: {self.initial_set_vals}')

        self.measurement = self.initialize_measurement()

        # Create dataset
        self.dataset = self.measurement.get_data_set(
            name=self.measurement_name,
            io=DataSet.default_io,
            location=self.loc_provider)

        self.acquisition_parameter.base_folder = self.dataset.location

    def set_vals_from_idx(self, idx):
        raise NotImplementedError('Must be implemented in subclass')


class Loop0DMeasurement(Measurement):
    def __init__(self, name=None, acquisition_parameter=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         **kwargs)
        self.set_parameters = []

    def set_vals_from_idx(self, idx):
        """
        Return set vals that correspond to the acquisition idx.
        In this case it returns an empty dict, since there are no set parameters
        Args:
            idx: Acquisition idx, in this case always zero

        Returns:
            Dict of set vals
        """
        return {}

    def initialize_measurement(self):
        self.measurement = qc.Measure(self.acquisition_parameter)
        return self.measurement

    @property
    def measurement_name(self):
        return f'{self.name}_{self.acquisition_parameter.name}'

    @clear_single_settings
    def get(self, set_active=True):
        """
        Performs a measurement at a single point using qc.Measure
        Returns:
            Dataset
        """
        self.initialize()

        logger.info(f'Performing 0D measurement {self.measurement_name}')
        try:
            self.measurement.run(quiet=True, set_active=set_active)
        finally:
            self.acquisition_parameter.base_folder = None


        # Test condition sets until a condition_set is found that has an action
        condition_set = self.check_condition_sets(self.dataset,
                                                  *self.condition_sets)

        # Find optimal values satisfying condition_sets.
        self.get_optimum(condition_set=condition_set)

        return self.dataset, condition_set.result


class Loop1DMeasurement(Measurement):
    def __init__(self, name=None, set_parameter=None, set_parameters=None,
                 acquisition_parameter=None, set_vals=None, step=None,
                 step_percentage=None, points=None, **kwargs):

        if set_parameters is None and set_parameter is not None:
            set_parameters = [set_parameter]

        if set_vals is not None:
            set_vals = [set_vals]

        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         set_parameters=set_parameters, set_vals=set_vals,
                         step=step, step_percentage=step_percentage,
                         points=points, **kwargs)

    @property
    def set_parameter(self):
        if self.set_vals is not None:
            return self.set_vals[0].parameter
        else:
            return _dummy_parameter

    def set_vals_from_idx(self, idx):
        """
        Return set vals that correspond to the acquisition idx.
        Args:
            idx: Acquisition idx. If equal to -1, returns nan for each element

        Returns:
            Dict of set vals (in this case contains one element)
        """
        if idx == -1:
            return {set_parameter.name: np.nan
                    for set_parameter in self.set_parameters}
        elif self.set_vals is not None:
            return {self.set_parameter.name: self.set_vals[0][idx]}
        else:
            # No set vals specified. This means that a dummy loop is
            # performed, and so set vals must be extracted from dataset
            return {set_parameter.name: getattr(self.dataset,
                                                set_parameter.name)[idx]
                    for set_parameter in self.set_parameters}

    def initialize_measurement(self):
        # Start with an empty set of actions in the loop
        actions = []

        if self.set_vals is not None:
            set_loop = qc.Loop(self.set_vals[0])
        elif self.points is not None:
            # No set vals specified, but points are, so create dummy parameter
            set_loop = qc.Loop(_dummy_parameter[0:self.points:1])

            # Also measure the set_parameters, as we are going to update them
            actions += self.set_parameters
        else:
            raise RuntimeError('Either set_vals or point must be defined')

        # Add measurement of acquisition parameter
        actions.append(self.acquisition_parameter)

        if self.break_if:
            actions.append(BreakIf(partial(self.satisfies_condition_set,
                                           self.acquisition_parameter,
                                           action=self.break_if)))

        self.measurement = set_loop.each(*actions)
        return self.measurement

    @property
    def measurement_name(self):
        return f'{self.name}_{self.set_parameter.name}_' \
               f'{self.acquisition_parameter.name}'

    @clear_single_settings
    def get(self, set_active=True):
        """
        Performs a 1D measurement loop
        Returns:
            Dataset
        """
        self.initialize()

        logger.info(f'Performing 1D measurement {self.measurement_name}')
        try:
            self.measurement.run(quiet=True, set_active=set_active)
        finally:
            self.acquisition_parameter.base_folder = None

        # Test condition sets until a condition_set is found that has an action
        condition_set = self.check_condition_sets(self.dataset,
                                                  *self.condition_sets)

        # Find optimal values satisfying condition_sets.
        self.get_optimum(condition_set=condition_set)

        # Update set parameter values, either to optimal values or to initial
        #  values. This depends on condition_set.update
        self.update_set_parameters(condition_set)

        return self.dataset, condition_set.result

    def set(self, set_vals=None, step=None, points=None):
        if set_vals is not None:
            self.step = None
            self.points = None
            self._set_vals = [self.set_parameter[list(set_vals)]]
        else:
            self._set_vals = None
            self.step = step
            if points is not None:
                self.points = points
            self._set_vals = self.set_vals


class Loop2DMeasurement(Measurement):
    def __init__(self, name=None, set_parameters=None,
                 acquisition_parameter=None, set_vals=None, step=None,
                 step_percentage=None, points=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         set_parameters=set_parameters, set_vals=set_vals,
                         step_percentage=step_percentage, step=step,
                         points=points, **kwargs)

    def set_vals_from_idx(self, idx):
        """
        Return set vals that correspond to the acquisition idx.
        Args:
            idx: Acquisition idx

        Returns:
            Dict of set vals (in this case contains two elements)
        """
        if idx == -1:
            return {p.name: np.nan for p in self.set_parameters}
        else:
            len_inner = len(self.set_vals[1])
            idxs = (idx // len_inner, idx % len_inner)
            return {self.set_parameters[0].name: self.set_vals[0][idxs[0]],
                    self.set_parameters[1].name: self.set_vals[1][idxs[1]]}

    def initialize_measurement(self):
        if self.break_if is False:
            self.measurement = qc.Loop(
                self.set_vals[0]).loop(
                    self.set_vals[1]).each(
                        self.acquisition_parameter)
        else:
            break_action = BreakIf(partial(self.satisfies_condition_set,
                                           self.acquisition_parameter,
                                           action=self.break_if))
            self.measurement = qc.Loop(
                self.set_vals[0]).each(
                    qc.Loop(
                        self.set_vals[1]).each(
                        self.acquisition_parameter,
                        break_action),
                    break_action)
        return self.measurement

    @property
    def measurement_name(self):
        return f'{self.name}_{self.set_parameters[0].name}_' \
               f'{self.set_parameters[1].name}_' \
               f'{self.acquisition_parameter.name}'

    @clear_single_settings
    def get(self, set_active=True):
        """
        Performs a 2D measurement loop
        Returns:
            Dataset
        """
        self.initialize()

        logger.info(f'Performing 2D measurement {self.measurement_name} ')
        logger.info(
            f'set_vals: {self.set_parameters[0].name}[{self.set_vals[0][:]}], '
            f'{self.set_parameters[1].name}[{self.set_vals[1][:]}')

        try:
            self.measurement.run(quiet=True, set_active=set_active)
        finally:
            self.acquisition_parameter.base_folder = None

        # Test condition sets until a condition_set is found that has an action
        condition_set = self.check_condition_sets(self.dataset,
                                                  *self.condition_sets)

        # Find optimal values satisfying condition_sets.
        self.get_optimum(condition_set=condition_set)

        # Update set parameter values, either to optimal values or to initial
        # values. This depends on condition_set.update
        self.update_set_parameters(condition_set)

        return self.dataset, condition_set.result

    def set(self, set_vals=None, step=None, points=None):
        if set_vals is not None:
            self.step = None
            self.points = None
            self._set_vals = [set_parameter[set_val] for set_parameter, set_val
                              in zip(self.set_parameters, set_vals)]
        else:
            self._set_vals = None
            if step is not None:
                self.step = step
            if points is not None:
                self.points = points
