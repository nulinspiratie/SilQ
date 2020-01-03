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
    def create_data_array(self, parameter, result, action_indices, is_setpoint=False):
        array_kwargs = {
            "is_setpoint": is_setpoint,
            "action_indices": action_indices,
        }

        # Potentially allow for 1D arrays in inner loops
        # Determine array shape
        # if is_setpoint:
        #     array_kwargs["shape"] = (len(result),)
        # else:
        array_kwargs["shape"] = self.loop_dimensions or (1, )
        if is_setpoint or isinstance(result, (np.ndarray, list)):
            array_kwargs["shape"] += np.shape(result)

        if isinstance(parameter, Parameter):
            array_kwargs["parameter"] = parameter
        else:
            array_kwargs["name"] = parameter
            array_kwargs["label"] = (
                parameter[0].capitalize() + parameter[1:].replace("_", " "),
            )
            array_kwargs["unit"] = ""

        # Add setpoints
        if not is_setpoint:
            array_kwargs['set_arrays'] = []
            for k in range(1, len(action_indices)):
                sweep_indices = action_indices[:k]
                array_kwargs['set_arrays'].append(self.set_arrays[sweep_indices])
            array_kwargs['set_arrays'] = tuple(array_kwargs['set_arrays'])

        print(array_kwargs)
        data_array = DataArray(**array_kwargs)

        data_array.array_id = data_array.full_name + "_".join(
            str(k) for k in action_indices
        )

        data_array.init_data()

        self.dataset.add_array(data_array)

        if is_setpoint:
            self.set_arrays[action_indices] = data_array
        else:
            self.data_arrays[action_indices] = data_array

        return data_array

    def create_data_array_group(self, parameter_node, action_indices):
        self.data_arrays[action_indices] = dict()

    def store_parameter_result(self, action_indices, parameter, result):
        # Get parameter data array (either create new array or choose existing)
        if action_indices not in self.data_arrays:
            # Create array based on first result type and shape
            data_array = self.create_data_array(parameter, result, action_indices)
        else:
            # Select existing array
            data_array = self.data_arrays[action_indices]

        # Add parameter result to data array
        # data_array[self.loop_indices] = result

        loop_indices = self.loop_indices or (0,)  # Allow for non-loop measurement

        self.dataset.store(loop_indices, {data_array.array_id: result})

    def store_parameter_node_results(
        self, action_indices, parameter_node, results, create: bool = True
    ):
        if action_indices not in self.data_arrays:
            if not create:
                raise RuntimeError(
                    f"Data array group for node {parameter_node} "
                    f"does not exist, but not allowed to create"
                )
            self.create_data_array_group(self.action_indices, parameter_node)

        if not isinstance(results, dict):
            raise SyntaxError(
                f"Results from node {parameter_node} is not a dict."
                f"Results are: {results}"
            )

        # Ensure there is a
        data_to_store = {}
        for k, (key, result) in enumerate(results.items()):
            # TODO this line is not right
            if key not in self.data_arrays:
                data_array = self.create_data_array(key, result, action_indices + (k,))
            else:
                data_array = self.data_arrays[action_indices + (k,)]

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
        action_indices = self.action_indices + (parameter_node.name,)

        results = parameter_node.get()

        self.store_parameter_node_results(action_indices, parameter_node, results)

        return results

    def measure(self, measurable):
        # Get corresponding data array (create if necessary)
        if isinstance(measurable, Parameter):
            result = self.measure_parameter(measurable)
        elif isinstance(measurable, ParameterNode):
            result = self.measure_parameter_node(measurable)

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
            self.set_array = running_measurement().set_arrays[running_measurement().action_indices]
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
        # Increment loop index of current dimension
        loop_indices = list(running_measurement().loop_indices)
        loop_indices[self.dimension] = self.loop_index
        running_measurement().loop_indices = tuple(loop_indices)

        try:  # Perform loop action
            sweep_value = next(self.iterator)
            # Remove last action index and increment one before that by one
            action_indices = list(running_measurement().action_indices)
            action_indices[-1] = 0
            running_measurement().action_indices = tuple(action_indices)
        except StopIteration:  # Reached end of iteration
            running_measurement().loop_dimensions = running_measurement().loop_dimensions[:-1]
            running_measurement().loop_indices = running_measurement().loop_indices[:-1]

            # Remove last action index and increment one before that by one
            action_indices = list(running_measurement().action_indices[:-1])
            action_indices[-1] += 1
            running_measurement().action_indices = tuple(action_indices)
            raise StopIteration

        if isinstance(self.sequence, SweepValues):
            self.sequence.set(sweep_value)

        self.set_array[running_measurement().loop_indices] = sweep_value

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
