import sys
import numpy as np
from typing import List, Tuple, Union, Sequence, Dict, Any
import threading
from time import sleep

from qcodes.data.data_set import new_data
from qcodes.data.data_array import DataArray
from qcodes.instrument.sweep_values import SweepValues
from qcodes import Parameter, ParameterNode
from qcodes.utils.helpers import using_ipython, directly_executed_from_cell


class Measurement:
    """
    Args:
        name: Measurement name, also used as the dataset name
        force_cell_thread: Enforce that the measurement has been started from a
            separate thread if it has been directly executed from an IPython
            cell/prompt. This is because a measurement is usually run from a
            separate thread using the magic command `%%new_job`.
            An error is raised if this has not been satisfied.
            Note that if the measurement is started within a function, no error
            is raised.


    Notes:
        When the Measurement is started in a separate thread (using %%new_job),
        the Measurement is registered in the user namespace as 'msmt', and the
        dataset as 'data'

    """

    # Context manager
    running_measurement = None
    measurement_thread = None

    # Default names for measurement and dataset, used to set user namespace
    # variables if measurement is executed in a separate thread.
    _default_measurement_name = "msmt"
    _default_dataset_name = "data"

    def __init__(self, name: str, force_cell_thread: bool = True):
        self.name = name

        # Total dimensionality of loop
        self.loop_dimensions: Union[Tuple[int], None] = None

        # Current loop indices
        self.loop_indices: Union[Tuple[int], None] = None

        # Index of current action
        self.action_indices: Union[Tuple[int], None] = None

        # contains data groups, such as ParameterNodes and nested measurements
        self._data_groups: Dict[Tuple[int], "Measurement"] = {}

        self.is_context_manager: bool = False  # Whether used as context manager
        self.is_paused: bool = False  # Whether the Measurement is paused
        self.is_stopped: bool = False  # Whether the Measurement is stopped

        self.force_cell_thread = force_cell_thread and using_ipython()

    @property
    def data_groups(self) -> Dict[Tuple[int], "Measurement"]:
        if running_measurement() is not None:
            return running_measurement()._data_groups
        else:
            return self._data_groups

    def __enter__(self):
        self.is_context_manager = True

        # Encapsulate everything in a try/except to ensure that the context
        # manager is properly exited.
        try:
            if Measurement.running_measurement is None:
                # Register current measurement as active primary measurement
                Measurement.running_measurement = self
                Measurement.measurement_thread = threading.current_thread()

                # Initialize dataset
                self.dataset = new_data(name=self.name)
                self.dataset.add_metadata({"measurement_type": "Measurement"})

                # Initialize attributes
                self.loop_dimensions = ()
                self.loop_indices = ()
                self.action_indices = (0,)
                self.data_arrays = {}
                self.set_arrays = {}

            else:
                if threading.current_thread() is not Measurement.measurement_thread:
                    raise RuntimeError(
                        'Cannot run a measurement while another measurement '
                        'is already running in a different thread.'
                    )

                # Primary measurement is already running. Add this measurement as
                # a data_group of the primary measurement
                msmt = Measurement.running_measurement
                msmt.data_groups[msmt.action_indices] = self
                msmt.action_indices += (0,)

                # Nested measurement attributes should mimic the primary measurement
                self.loop_dimensions = msmt.loop_dimensions
                self.loop_indices = msmt.loop_indices
                self.action_indices = msmt.action_indices
                self.data_arrays = msmt.data_arrays
                self.set_arrays = msmt.set_arrays

            # Perform measurement thread check, and set user namespace variables
            if self.force_cell_thread and Measurement.running_measurement is self:
                # Raise an error if force_cell_thread is True and the code is run
                # directly from an IPython cell/prompt but not from a separate thread
                is_main_thread = threading.current_thread() == threading.main_thread()
                if is_main_thread and directly_executed_from_cell():
                    raise RuntimeError(
                        "Measurement must be created in dedicated thread. "
                        "Otherwise specify force_thread=False"
                    )

                # Register the Measurement and data as variables in the user namespace
                # Usually as variable names are 'msmt' and 'data' respectively
                from IPython import get_ipython

                shell = get_ipython()
                shell.user_ns[self._default_measurement_name] = self
                shell.user_ns[self._default_dataset_name] = self.dataset

            return self
        except:
            # An error has occured, ensure running_measurement is cleared
            if Measurement.running_measurement is self:
                Measurement.running_measurement = None
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        msmt = Measurement.running_measurement
        if msmt is self:
            Measurement.running_measurement = None
            self.dataset.finalize()
        else:
            # This is a nested measurement.
            # update action_indices of primary measurements
            msmt.action_indices = msmt.action_indices[:-1]

        self.is_context_manager = False

    # Data array functions
    def _create_data_array(
        self,
        parameter: Union[Parameter, str],
        result,
        action_indices: Tuple[int],
        ndim: int = None,
        is_setpoint: bool = False,
        label: str = None,
        unit: str = None,
    ):
        """Create a data array from a parameter and result.

        The data array shape is extracted from the result shape, and the current
        loop dimensions.

        The data array is added to the current data set.

        Args:
            parameter: Parameter for which to create a DataArray. Can also be a
                string, in which case it is the data_array name
            result: Result returned by the Parameter
            action_indices: Action indices for which to store parameter
            ndim: Number of dimensions. If not provided, will use length of
                action_indices
            is_setpoint: Whether the Parameter is used for sweeping or measuring
            label: Data array label. If not provided, the parameter label is
                used. If the parameter is a name string, the label is extracted
                from the name.
            unit: Data array unit. If not provided, the parameter unit is used.

        Returns:
            Newly created data array

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
            if label is None:
                label = parameter[0].capitalize() + parameter[1:].replace("_", " ")
            array_kwargs["label"] = label
            array_kwargs["unit"] = unit or ""

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
            if sweep_indices in self.set_arrays:
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

    def get_arrays(self, action_indices: Sequence[int] = None) -> List[DataArray]:
        """Get all arrays belonging to the current action indices
        If the action indices corresponds to a group of arrays (e.g. a nested
        measurement or ParameterNode), all the arrays in the group are returned

        Args:
            action_indices: Action indices of arrays.
                If not provided, the current action_indices are chosen

        Returns:
            List of data arrays matching the action indices
        """
        if action_indices is None:
            action_indices = self.action_indices

        if not isinstance(action_indices, Sequence):
            raise SyntaxError("parent_action_indices must be a tuple")

        num_indices = len(action_indices)
        return [
            arr
            for action_indices, arr in self.data_arrays.items()
            if action_indices[:num_indices] == action_indices
        ]

    def _add_measurement_result(
        self,
        action_indices,
        parameter,
        result,
        ndim=None,
        store: bool = True,
        label: str = None,
        unit: str = None,
    ):
        # Get parameter data array, creating a new one if necessary
        if action_indices not in self.data_arrays:
            # Create array based on first result type and shape
            self._create_data_array(
                parameter, result, action_indices, ndim=ndim, label=label, unit=unit
            )

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

    # Measurement-related functions
    def _measure_parameter(self, parameter):
        # Get parameter result
        result = parameter()

        self._add_measurement_result(self.action_indices, parameter, result)

        return result

    def _measure_callable(self, callable, name=None):

        results = callable()

        # Check if the callable already performed a nested measurement
        # In this case, the nested measurement is stored as a data_group, and
        # has loop indices corresponding to the current ones.
        msmt = Measurement.running_measurement
        data_group = msmt.data_groups.get(self.action_indices)
        if getattr(data_group, "loop_indices", None) == self.loop_indices:
            # Measurement has already been performed by a nested measurement
            return results
        else:
            # No nested measurement has been performed in the callable.
            # Add results, which should be dict, by creating a nested measurement
            if not isinstance(results, dict):
                raise SyntaxError(f"{name} results must be a dict, not {results}")

            with Measurement(name) as msmt:
                for key, val in results.items():
                    msmt.measure(val, name=key)

        return results

    def measure(self, measurable, name=None):
        # TODO add label, unit, etc. as kwargs
        if not self.is_context_manager:
            raise RuntimeError(
                "Must use the Measurement as a context manager, "
                "i.e. 'with Measurement(name) as msmt:'"
            )
        elif self.is_stopped:
            raise SystemExit("Measurement.stop() has been called")
        elif threading.current_thread() is not Measurement.measurement_thread:
            raise RuntimeError(
                'Cannot measure while another measurement is already running '
                'in a different thread.'
            )

        if self != Measurement.running_measurement:
            # Since this Measurement is not the running measurement, it is a
            # DataGroup in the running measurement. Delegate measurement to the
            # running measurement
            return Measurement.running_measurement.measure(measurable, name=name)

        # Code from hereon is only reached by the primary measurement,
        # i.e. the running_measurement

        # Wait as long as the measurement is paused
        while self.is_paused:
            sleep(0.1)

        if isinstance(measurable, Parameter):
            result = self._measure_parameter(measurable)
        elif callable(measurable):
            if name is None:
                if hasattr(measurable, "__self__") and isinstance(
                    measurable.__self__, ParameterNode
                ):
                    name = measurable.__self__.name
                elif hasattr(measurable, "__name__"):
                    name = measurable.__name__
                else:
                    action_indices_str = "_".join(
                        str(idx) for idx in self.action_indices
                    )
                    name = f"data_group_{action_indices_str}"
            result = self._measure_callable(measurable, name=name)
        elif isinstance(measurable, (float, int, bool, np.ndarray)):
            if name is None:
                raise RuntimeError(
                    "A name must be provided when measuring an int, float, bool, or array"
                )
            result = measurable
            self._add_measurement_result(
                action_indices=self.action_indices, parameter=name, result=result
            )
        else:
            raise RuntimeError(
                f"Cannot measure {measurable} as it cannot be called, and it "
                f"is not an int, float, bool, or numpy array."
            )

        # Increment last action index by 1
        action_indices = list(self.action_indices)
        action_indices[-1] += 1
        self.action_indices = tuple(action_indices)

        return result

    # Functions relating to measurement flow
    def pause(self):
        """Pause measurement at start of next parameter sweep/measurement"""
        self.is_paused = True

    def resume(self):
        """Resume measurement after being paused"""
        self.is_paused = False

    def stop(self):
        self.is_stopped = True


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
        if threading.current_thread() is not Measurement.measurement_thread:
            raise RuntimeError(
                'Cannot create a Sweep while another measurement '
                'is already running in a different thread.'
            )

        running_measurement().loop_dimensions += (len(self.sequence),)
        running_measurement().loop_indices += (0,)
        running_measurement().action_indices += (0,)

        # Create a set array if necessary

        self.loop_index = 0
        self.iterator = iter(self.sequence)

        return self

    def __next__(self):
        msmt = running_measurement()

        if not msmt.is_context_manager:
            raise RuntimeError(
                "Must use the Measurement as a context manager, "
                "i.e. 'with Measurement(name) as msmt:'"
            )
        elif msmt.is_stopped:
            raise SystemExit

        # Wait as long as the measurement is paused
        while msmt.is_paused:
            sleep(0.1)

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
                parameter=self.name or "iterator",
                unit=self.unit,
                result=self.sequence,
                action_indices=running_measurement().action_indices,
                is_setpoint=True,
            )
