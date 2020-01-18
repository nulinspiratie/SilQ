import numpy as np
from typing import List, Tuple, Union

from qcodes.data.data_set import new_data
from qcodes.data.data_array import DataArray
from qcodes.instrument.sweep_values import SweepValues
from qcodes import Parameter, ParameterNode


class Measurement:
    # Context manager
    running_measurement = None

    def __init__(self, name: str):
        self.name = name

        self.loop_dimensions: Tuple[int] = None  # Total dimensionality of loop

        self.loop_indices: Tuple[int] = None  # Current loop indices

        self.action_indices: Tuple[int] = None  # Index of action

        # contains data groups, such as ParameterNodes and nested measurements
        self._data_groups = {}

        self.active: bool = False  # Only become active when used as context manager

    @property
    def data_groups(self):
        return running_measurement()._data_groups

    def __enter__(self):
        self.active = True

        if Measurement.running_measurement is None:
            # Register current measurement as active primary measurement
            Measurement.running_measurement = self

            # Initialize dataset
            self.dataset = new_data(name=self.name)
        else:
            # Primary measurement is already running. Add this measurement as
            # a data_group of the primary measurement
            msmt = Measurement.running_measurement
            msmt.data_groups[msmt.action_indices] = self

        self.loop_dimensions = ()
        self.loop_indices = ()
        self.action_indices = (0,)

        self.data_arrays = {}
        self.set_arrays = {}

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if Measurement.running_measurement is self:
            Measurement.running_measurement = None
            self.dataset.finalize()

        self.active = False

    # Data array functions

    def _create_data_array(
        self,
        parameter: Union[Parameter, str],
        result,
        action_indices: Tuple[int],
        ndim: int = None,
        is_setpoint: bool = False,
    ):
        """

        Args:
            parameter: Parameter for which to create a DataArray
            result: Result returned by the Parameter
            action_indices: Action indices for which to store parameter
            ndim: Number of dimensions. If not provided, will use length of
                action_indices
            is_setpoint: Whether the Parameter is used for sweeping or measuring

        Returns:

        """

        if ndim is None:
            ndim = len(action_indices)

        array_kwargs = {
            "is_setpoint": is_setpoint,
            "action_indices": action_indices,
            "shape": self.loop_dimensions,
        }

        if is_setpoint or isinstance(result, (np.ndarray, list)):
            array_kwargs["shape"] += np.shape(result)

        if isinstance(parameter, Parameter):
            array_kwargs["parameter"] = parameter
        else:
            array_kwargs["name"] = parameter
            array_kwargs["label"] = parameter[0].capitalize() + parameter[1:].replace(
                "_", " "
            )
            array_kwargs["unit"] = ""

        # Add setpoint arrays
        if not is_setpoint:
            array_kwargs["set_arrays"] = self._add_set_arrays(
                parameter, result, action_indices, ndim
            )

        data_array = DataArray(**array_kwargs)

        data_array.array_id = data_array.full_name
        data_array.array_id += "_" + "_".join(str(k) for k in action_indices)

        data_array.init_data()

        self.dataset.add_array(data_array)

        # Add array to set_arrays or to data_arrays of this Measurement
        if is_setpoint:
            self.set_arrays[action_indices] = data_array
        else:
            self.data_arrays[action_indices] = data_array

        return data_array

    def _add_set_arrays(
        self,
        parameter: Union[Parameter, str],
        result,
        action_indices: Tuple[int],
        ndim: int,
    ):
        set_arrays = []
        for k in range(1, ndim):
            sweep_indices = action_indices[:k]
            set_arrays.append(self.set_arrays[sweep_indices])
            # TODO handle grouped arrays (e.g. ParameterNode, nested Measurement)

        # Create new set array(s) if parameter result is an array or list
        if isinstance(result, (np.ndarray, list)):
            if isinstance(result, list):
                result = np.ndarray(result)

            # TODO handle if the parameter contains attribute setpoints

            # parameter can also be a string, in which case we don't use parameter.name
            name = getattr(parameter, "name", parameter)

            for k, shape in enumerate(result.shape):
                arr = np.arange(shape)
                # Add singleton dimensions
                arr = np.broadcast_to(arr, result.shape[: k + 1])

                set_array = self._create_data_array(
                    parameter=f"{name}_set{k}",
                    result=arr,
                    action_indices=action_indices + (0,) * k,
                    is_setpoint=True,
                )
                set_arrays.append(set_array)

        return tuple(set_arrays)

    def _create_data_array_group(self, action_indices, parameter_node):
        # TODO: Finish this function
        # self.data_arrays[action_indices] = dict()
        pass

    def _process_parameter_result(
        self, action_indices, parameter, result, ndim=None, store: bool = True
    ):
        # Get parameter data array (either create new array or choose existing)
        if action_indices not in self.data_arrays:
            # Create array based on first result type and shape
            data_array = self._create_data_array(
                parameter, result, action_indices, ndim=ndim
            )
        else:
            # Select existing array
            data_array = self.data_arrays[action_indices]

        # Ensure an existing data array has the correct name
        # parameter can also be a string, in which case we don't use parameter.name
        name = getattr(parameter, "name", parameter)
        if not data_array.name == name:
            raise SyntaxError(
                f"Existing DataArray '{data_array.name}' differs from result {name}"
            )

        data_to_store = {data_array.array_id: result}

        # If result is an array, update set_array elements
        if isinstance(result, list):  # Convert result list to array
            result = np.ndarray(result)
        if isinstance(result, np.ndarray):
            ndim = len(self.loop_indices)
            if len(data_array.set_arrays) != ndim + result.ndim:
                raise RuntimeError(
                    f"Wrong number of set arrays for {data_array.name}. "
                    f"Expected {ndim + result.ndim} instead of "
                    f"{len(data_array.set_arrays)}."
                )

            for k, set_array in enumerate(data_array.set_arrays[ndim:]):
                # Successive set arrays must increase dimensionality by unity
                arr = np.arange(result.shape[k])
                # Add singleton dimensions
                arr = np.broadcast_to(arr, result.shape[: k + 1])
                data_to_store[set_array.array_id] = arr

        if store:
            self.dataset.store(self.loop_indices, data_to_store)

        return data_to_store

    def _store_dict_results(
        self,
        action_indices: Tuple[int],
        group_name: str,
        results: dict,
        create: bool = True,
    ):
        if action_indices not in self.data_arrays:
            if not create:
                raise RuntimeError(
                    f"Data array group {group_name} "
                    f"does not exist, but not allowed to create"
                )
            self._create_data_array_group(self.action_indices, group_name)

        if not isinstance(results, dict):
            raise SyntaxError(
                f"Results from {group_name} is not a dict." f"Results are: {results}"
            )

        data_to_store = {}
        for k, (key, result) in enumerate(results.items()):
            data_to_store.update(
                **self._process_parameter_result(
                    action_indices=action_indices + (k,),
                    parameter=key,
                    result=result,
                    ndim=len(action_indices),
                    store=False,
                )
            )

        # Add result to data array
        self.dataset.store(self.loop_indices, data_to_store)

    # Measurement-related functions
    def _measure_parameter(self, parameter):
        # Get parameter result
        result = parameter()

        self._process_parameter_result(self.action_indices, parameter, result)

        return result

    def _measure_parameter_node(self, parameter_node):
        action_indices = self.action_indices

        results = parameter_node.get()

        self._store_dict_results(action_indices, parameter_node.name, results)

        return results

    def _measure_callable(self, callable):
        action_indices = self.action_indices

        results = callable()

        self._store_dict_results(action_indices, callable.__name__, results)

        return results

    def measure(self, measurable):
        if not self.active:
            raise RuntimeError("Must use the Measurement as a context manager, "
                               "i.e. 'with Measurement(name) as msmt:'")

        if self != Measurement.running_measurement:
            # Since this Measurement is not the running measurement, it is a
            # DataGroup in the running measurement. Delegate measurement to the
            # running measurement
            return Measurement.running_measurement.measure(measurable)

        # Get corresponding data array (create if necessary)
        if isinstance(measurable, Parameter):
            result = self._measure_parameter(measurable)
        elif isinstance(measurable, ParameterNode):
            result = self._measure_parameter_node(measurable)
        elif callable(measurable):
            result = self._measure_callable(measurable)
        else:
            raise RuntimeError(f"Cannot measure {measurable} as it cannot be called")

        # Increment last action index by 1
        action_indices = list(self.action_indices)
        action_indices[-1] += 1
        self.action_indices = tuple(action_indices)

        return result


def running_measurement() -> Measurement:
    return Measurement.running_measurement


class Sweep:
    def __init__(self, sequence, name=None, unit=None):
        if running_measurement() is None:
            raise RuntimeError("Cannot create a sweep outside a Measurement")

        # Properties for the data array
        self.name = name
        self.unit = unit

        self.sequence = sequence
        self.dimension = len(running_measurement().loop_dimensions)
        self.loop_index = None
        self.iterator = None

        if running_measurement().action_indices in running_measurement().set_arrays:
            self.set_array = running_measurement().set_arrays[
                running_measurement().action_indices
            ]
        else:
            self.set_array = self.create_set_array()

    def __iter__(self):
        running_measurement().loop_dimensions += (len(self.sequence),)
        running_measurement().loop_indices += (0,)
        running_measurement().action_indices += (0,)

        # Create a set array if necessary

        self.loop_index = 0
        self.iterator = iter(self.sequence)

        return self

    def __next__(self):
        msmt = running_measurement()

        # Increment loop index of current dimension
        loop_indices = list(msmt.loop_indices)
        loop_indices[self.dimension] = self.loop_index
        msmt.loop_indices = tuple(loop_indices)

        try:  # Perform loop action
            sweep_value = next(self.iterator)
            # Remove last action index and increment one before that by one
            action_indices = list(msmt.action_indices)
            action_indices[-1] = 0
            msmt.action_indices = tuple(action_indices)
        except StopIteration:  # Reached end of iteration
            msmt.loop_dimensions = msmt.loop_dimensions[:-1]
            msmt.loop_indices = msmt.loop_indices[:-1]

            # Remove last action index and increment one before that by one
            action_indices = list(msmt.action_indices[:-1])
            action_indices[-1] += 1
            msmt.action_indices = tuple(action_indices)
            raise StopIteration

        if isinstance(self.sequence, SweepValues):
            self.sequence.set(sweep_value)

        self.set_array[msmt.loop_indices] = sweep_value

        self.loop_index += 1

        return sweep_value

    def create_set_array(self):
        if isinstance(self.sequence, SweepValues):
            return running_measurement()._create_data_array(
                parameter=self.sequence.parameter,
                result=self.sequence,
                action_indices=running_measurement().action_indices,
                is_setpoint=True,
            )
        else:
            return running_measurement()._create_data_array(
                name=self.name or "iterator",
                unit=self.unit,
                result=self.sequence,
                action_indices=running_measurement().action_indices,
                is_setpoint=True,
            )
