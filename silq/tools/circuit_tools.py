from typing import Union, List
from pathlib import Path

import numpy as np
import pygsti
from pygsti.protocols.gst import GateSetTomographyDesign
from pygsti.protocols.rpe import RobustPhaseEstimationDesign
from pygsti.circuits import Circuit

from qcodes import DataArray

def convert_circuit(circuit, target_type: Union[str, List[str], Circuit] = str,
                    include_state_space_labels=True,
                    ):
    """Convert a circuit to a target type (e.g. list, str or Circuit)

    Args:
        circuit: Circuit that should be converted
            Can be a str, list, or pygsti Circuit
        target_type: Target circuit type
            can be str, list, pygsti Circuit
        include_state_space_labels: If True, every gate in the circuit will be
            represented by the operation and the target qubit (e.g. 'Gx:0'),
            if False the target qubit is dropped.

    Examples:
        >>> convert_circuit('GxGi', target_type=list)
        ['Gx', 'Gi']
        >>> convert_circuit('Gx:0Gi:1', target_type=Circuit)
        Circuit(Gx:0Gi:1@(0,1))

    Note: For all target types, the input circuit will be first converted to a
          pygsti Circuit and then converted to the final output type. For 500
          circuits this takes on average 5 ms. If this is found to be too slow,
          a direct conversion may be implemented.
    """
    assert target_type in [Circuit, str, list], \
        "Target type not understood, must be either Circuit, string or list."

    # No conversion necessary
    if isinstance(circuit, target_type):
        return circuit

    if isinstance(circuit, Circuit):
        output_circuit = []
        for gate in circuit:
            if include_state_space_labels:
                output_circuit.append(str(gate))
            else:
                output_circuit.append(gate.name)

        if target_type == str:
            output_circuit = "".join(output_circuit)

    if isinstance(circuit, (str, list)):
        # Convert to a Circuit first
        output_circuit = Circuit(circuit)
        output_circuit = convert_circuit(output_circuit, target_type,
                                             include_state_space_labels)

    return output_circuit


def _get_experiment_dir(filepath: Union[str, Path],
                        config_entry: str='properties.circuits_folder'):
    """For relative paths, attempts to find the `filepath` in the nominated
    circuits folder (default config.properties.circuits_folder) or in the current
    working directory.

    Search priority: absolute path, if path is relative then config circuits
                     folder, then local directory

    Args:
        filepath: The path to a pyGSTi generated folder which includes the
                  "edesign" and "data" subfolders
        config_entry: The location in the silq config that specifies the default
                    path for pyGSTi experiments.

    Returns:
        Absolute path to specified folder.

    Raises:
        FileNotFoundError
    """
    # Defer importing of silq config as circuit_tools is typically imported
    # before the config is created.
    from silq import config

    assert config_entry in config, f'config.{config_entry} does not exist.'
    assert isinstance(config[config_entry], str), \
        f"Config entry for experiment designs path must be a string and not " \
        f"{type(config[config_entry])}"

    filepath = Path(filepath)
    if not filepath.is_absolute():
        config_path = Path(config[config_entry]) / filepath
        if config_path.exists():
            filepath = config_path
        elif not filepath.exists():
            raise FileNotFoundError(
                f'Cannot find specified file "{filepath}" in'
                f' config.{config_entry} or in '
                f'the current directory.')

    return filepath.absolute()

def load_experiment_design(
        filepath: Union[str, Path],
        **kwargs
):
    """Loads a pyGSTi experiment design from file.

    Args:
        filepath: A relative or absolute path to a pyGSTi generated folder which
                  includes the "edesign" and "data" subfolders.

    Returns:
        An experiment design.
    """

    filepath = _get_experiment_dir(filepath, **kwargs)

    # If absolute file does not exist, error will be raised below.
    return pygsti.io.read_edesign_from_dir(filepath, **kwargs)

def load_GST_circuits(
        exp_design: Union[str, GateSetTomographyDesign,
                                  RobustPhaseEstimationDesign, Path],
        circuit_list: str='all_circuits_needing_data',
        **kwargs
):
    """Loads a pyGSTi experiment design and converts the circuits to
        a simple format, a list of circuits which

    Args:
        exp_design: An experiment design or the path to an experiment design folder.

        circuit_list: The name of the circuit list to load from the experiment design.

    Returns:
        A list of circuits, where each circuit is a list of gates represented in string format.
    """
    if isinstance(exp_design, (GateSetTomographyDesign, RobustPhaseEstimationDesign)):
        pass
    else:
        exp_design = load_experiment_design(exp_design)

    circuits = getattr(exp_design, circuit_list)

    # Perform any necessary conversion, see defaults for convert_circuit
    circuits = [convert_circuit(circuit, **kwargs) for circuit in circuits]

    return circuits

def load_dataset(path: Union[str, Path], **kwargs):
    path = Path(path)
    if path.suffix == '':  # Treat as path to experiment design
        path = _get_experiment_dir(path, **kwargs) / 'data/dataset.txt'
    elif path.suffix == 'txt' and not path.is_absolute():
        # Relative path to .txt dataset, try config first then local directory
        try:
            config_path = _get_experiment_dir(path.parents[1], **kwargs) / \
                          path.parents[0] / path.name

            if config_path.exists():
               path = config_path
        except IndexError:
            # Occurs if relative path does not have two parent directories.
            pass

    # Else, path is an absolute path to a .txt file, or a relative path in which
    # case the dataset will attempt to be loaded from the current directory.
    return pygsti.io.read_dataset(path, **kwargs)

def create_dataset(results, circuits:List[Circuit], path=None):
    """Create a pyGSTi DataSet from a results array.

    Args:
        results: A nested dictionary of circuit results, with the summed counts
                for each possible outcome. The dict value for each circuit is a
                set of outcome_label-count pairs.
        circuits: An ordered list of the circuits, must be the same length as
                the number of values in each dictionary entry.
        path: If set, write the resulting dataset to file at the specified path.

    Returns:
        Populated pyGSTi dataset.
    """

    dataset = pygsti.data.DataSet()

    for k, circuit in enumerate(circuits):
        dataset.add_count_dict(circuit, results[circuit])

    if path is not None:
        path = Path(path)
        if path.is_dir():
            path /= 'data/dataset.txt'
        pygsti.io.write_dataset(path, dataset)

    return dataset

def analyse_circuit_results(
    exp_design,
    outcomes_arr:DataArray,
    circuits:List[Circuit]=None,
    circuits_axis=None,
    outcomes_axis=None,
    outcomes_mapping=None
):
    """

    Args:
        exp_design: A pyGSTi experiment design or path to one.
        outcomes_arr: A QCodes DataArray which has all of the resulting counts
                     from running the given circuits.
        circuits: A list of pyGSTi circuits. If unspecified, the experiment design
        'all_circuits_needing_data' is used.
        circuits_axis: Specifies which axis distinguishes each circuit.
        outcomes_axis: Specifies which axis distinguishes each circuit outcome.
        outcomes_mapping: Optional, a mapping between each outcome and the index
                    into the outcomes_arr on the outcomes_axis. This is useful if
                    the recorded results for each outcome aren't in binary order.
                    Defualt mapping is from index i to its binary string
                    representation, e.g. index 3 corresponds to outcome '11'.

    Returns:
        A dictionary of circuit results, with the summed counts for each
        possible outcome.

    """
    if isinstance(exp_design, (RobustPhaseEstimationDesign, GateSetTomographyDesign)):
        pass
    else:
        exp_design = load_experiment_design(exp_design)

    if circuits is None:
        circuits = exp_design.all_circuits_needing_data



    target_model = exp_design.create_target_model()
    # Get unique set of outcome labels for this model (normally '0', '1', etc.)
    outcome_labels = set(
        [key for povm in target_model.povms.values() for key in povm.keys()])
    n_outcomes = len(outcome_labels)

    if outcomes_mapping is None:
        outcomes_mapping = {label: k for k, label in enumerate(outcome_labels)}

    if outcomes_axis is None:
        # Find which axis hosts the distinct circuit outcomes
        n_matching_axes = np.count_nonzero(
            np.array(outcomes_arr.shape) == n_outcomes)
        assert n_matching_axes == 1,\
        f"{n_matching_axes} axes were found with dimension that matches the number of " \
        "expected outcomes. Please specify which axis distinguishes the circuit" \
        "outcomes."

        outcomes_axis = np.argmin(np.array(outcomes_arr.shape), n_outcomes)


    if circuits_axis is None:
        # Find which axis hosts the distinct circuit outcomes
        n_matching_axes = np.count_nonzero(
            np.array(outcomes_arr.shape) == len(circuits))
        assert n_matching_axes == 1,\
        f"{n_matching_axes} axes were found with dimension that matches the number of " \
        "circuits. Please specify which axis distinguishes the circuits."

        circuits_axis = np.argmin(np.array(outcomes_arr.shape), len(circuits))

    # Move the circuits and outcomes axis to the final two dimensions
    outcomes_arr = np.swapaxes(outcomes_arr, outcomes_axis, -1)
    outcomes_arr = np.swapaxes(outcomes_arr, circuits_axis, -2)

    # Assume all other dimensions are repetitions of the circuit and sum all counts
    outcomes_arr = np.sum(outcomes_arr.reshape(-2, *outcomes_arr.shape[-2:]), axis=0)

    return {circuit: {outcome_label: outcomes_arr[circuit_idx][outcomes_mapping[outcome_label]]
                      for outcome_label in outcome_labels}
            for circuit_idx, circuit in enumerate(circuits)}

# deprecated
def analyse_circuit_results_old(
    flips_arr,
    possible_flips_arr,
    circuits=None,
    output_filename=None,
    insert_identity=False,
):
    # Each circuit is repeated if the up_proportion filtering was unsuccessful.
    # This happens a maximum of five times, though the majority is already successful at the first attempt.
    # Here we remove this final axis.
    flips_flattened_arr = np.nansum(flips_arr, axis=2)
    # If all five attempts are unsuccessful, we set this array element back to nan
    all_attempts_nan = np.all(np.isnan(flips_arr), axis=2)
    flips_flattened_arr[all_attempts_nan] = np.nan

    possible_flips_flattened_arr = np.nansum(possible_flips_arr, axis=2)
    possible_flips_flattened_arr[all_attempts_nan] = np.nan

    successful_flips = np.nansum(flips_flattened_arr, axis=0)
    unsuccessful_flips = np.nansum(possible_flips_flattened_arr, axis=0) - successful_flips

    results = {
        '1': successful_flips,
        '0': unsuccessful_flips
    }

    if circuits is not None:
        assert len(successful_flips) == len(circuits)
        results['circuits'] = {
            circuit: [flips, no_flips] for circuit, flips, no_flips in zip(circuits, successful_flips, unsuccessful_flips)
        }

        if output_filename is not None:
            output_filename = Path(output_filename)
            assert output_filename.is_absolute()

            with open(output_filename, 'w') as f:
                f.write('## Columns = 0 count, 1 count\n')

                if insert_identity:
                    # First circuit (identity operation) has not been performed.
                    max_flips = np.nanmax(possible_flips_flattened_arr)
                    f.write(f'{{}} {int(max_flips)} 0\n')

                for key, (flips, no_flips) in results['circuits'].items():
                    f.write(f'{key} {int(no_flips)} {int(flips)}\n')

            results['filepath'] = str(output_filename)

    return results

def save_circuits(circuits, filepath):
    """Save list of circuits to a .txt file"""
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.txt')

    with filepath.open('w') as f:
        for circuit in circuits:
            circuit_str = convert_circuit(circuit, target_type=str)
            f.write(circuit_str + '\n')

def load_circuits(
        filepath: Union[str, Path],
        target_type: type = list,
        load_probabilities: bool = False
):
    filepath = Path(filepath)
    with open(filepath.with_suffix('.txt')) as file:
        lines = [line.strip() for line in file]

    # Check if line already contains data
    if lines[0].startswith('##'):
        # Remove header line
        lines = lines[1:]

    # Remove any existing results
    circuit_strings = [line.split(' ')[0] for line in lines]

    # Ensure all strings are of the target type
    circuits = [
        convert_circuit(circuit, target_type=target_type)
        for circuit in circuit_strings
    ]

    if load_probabilities:
        # Ignore when there is more than one space
        lines = [' '.join(line.split()) for line in lines]
        state_events = np.array([
            [int(elem) for elem in line.split(' ')[1:]]
            for line in lines
        ])
        state_probabilities = state_events / state_events.sum(axis=1)[:, None]
        return circuits, state_probabilities
    else:
        return circuits