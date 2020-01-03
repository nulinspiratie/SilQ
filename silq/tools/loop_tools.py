import numpy as np
from typing import List, Tuple

from qcodes.data.data_set import new_data
from qcodes.data.data_array import DataArray
from qcodes.instrument.sweep_values import SweepValues
from qcodes import Parameter, ParameterNode


def running_measurement():
    return Measurement.running_measurement


class Measurement:
    # Context manager
    running_measurement = None

    def __init__(self, name: str, run: bool = True):
        self.name = name

        self.run = run

        self.loop_dimensions: Tuple[int] = None  # Total dimensionality of loop

        self.loop_indices: Tuple[int] = None  # Current loop indices

        self.action_indices: Tuple[int] = None  # Index of action

    def __enter__(self):
        self.dataset = new_data(name=self.name)

        # Register current measurement as active measurement
        if Measurement.running_measurement is not None:
            raise RuntimeError("Currently cannot handle multiple measurements")
        Measurement.running_measurement = self

        self.loop_dimensions = ()
        self.loop_indices = ()
        self.action_indices = (0,)

        self.data_arrays = {}
        self.set_arrays = {}

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Measurement.running_measurement = None
        self.dataset.finalize()

    # Data array functions

    def create_data_array(
        self,
        parameter: Parameter,
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
            array_kwargs["set_arrays"] = self.add_set_arrays(
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

    def add_set_arrays(self, parameter, result, action_indices, ndim):
        set_arrays = []
        for k in range(1, ndim):
            sweep_indices = action_indices[:k]
            set_arrays.append(self.set_arrays[sweep_indices])

        # Create new set array(s) if parameter result is an array or list
        if isinstance(result, (np.ndarray, list)):
            if isinstance(result, list):
                result = np.ndarray(result)

            set_arrays.append(self.create_data_array())


            raise RuntimeError("No support yet for parameters returning Array")
            # array_kwargs['set_arrays'] += self.create_array_setpoints(
            #     parameter, result)

        return tuple(set_arrays)


    def create_data_array_group(self, action_indices, parameter_node):
        # TODO: Finish this function
        # self.data_arrays[action_indices] = dict()
        pass

    def store_parameter_result(self, action_indices, parameter, result):
        # Get parameter data array (either create new array or choose existing)
        if action_indices not in self.data_arrays:
            # Create array based on first result type and shape
            data_array = self.create_data_array(parameter, result, action_indices)
        else:
            # Select existing array
            data_array = self.data_arrays[action_indices]

        self.dataset.store(self.loop_indices, {data_array.array_id: result})

    def store_dict_results(
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
            self.create_data_array_group(self.action_indices, group_name)

        if not isinstance(results, dict):
            raise SyntaxError(
                f"Results from {group_name} is not a dict." f"Results are: {results}"
            )

        data_to_store = {}
        for k, (key, result) in enumerate(results.items()):
            if action_indices + (k,) not in self.data_arrays:
                data_array = self.create_data_array(
                    key, result, action_indices + (k,), ndim=len(action_indices)
                )
            else:
                data_array = self.data_arrays[action_indices + (k,)]

            # Ensure an existing data array has the correct name
            if not data_array.name == key:
                raise SyntaxError(
                    f"Existing DataArray '{data_array.name}' differs from "
                    f"ParameterNode result key {key}"
                )

            data_to_store[data_array.array_id] = result

        # Add result to data array
        self.dataset.store(self.loop_indices, data_to_store)

    # Measurement-related functions
    def measure_parameter(self, parameter):
        # Get parameter result
        result = parameter()

        self.store_parameter_result(self.action_indices, parameter, result)

        return result

    def measure_parameter_node(self, parameter_node):
        action_indices = self.action_indices

        results = parameter_node.get()

        self.store_dict_results(action_indices, parameter_node.name, results)

        return results

    def measure_callable(self, callable):
        action_indices = self.action_indices

        results = callable()

        self.store_dict_results(action_indices, callable.__name__, results)

        return results

    def measure(self, measurable):
        # Get corresponding data array (create if necessary)
        if isinstance(measurable, Parameter):
            result = self.measure_parameter(measurable)
        elif isinstance(measurable, ParameterNode):
            result = self.measure_parameter_node(measurable)
        elif callable(measurable):
            result = self.measure_callable(measurable)
        else:
            raise RuntimeError(f"Cannot measure {measurable}, it cannot be called")

        # Increment last action index by 1
        action_indices = list(self.action_indices)
        action_indices[-1] += 1
        self.action_indices = tuple(action_indices)

        return result


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
            return running_measurement().create_data_array(
                parameter=self.sequence.parameter,
                result=self.sequence,
                action_indices=running_measurement().action_indices,
                is_setpoint=True,
            )
        else:
            return running_measurement().create_data_array(
                name=self.name or "iterator",
                unit=self.unit,
                result=self.sequence,
                action_indices=running_measurement().action_indices,
                is_setpoint=True,
            )
