from typing import Type
from pathlib import Path
import numpy as np
import regex as re
pi = np.pi

from pygsti.objects.circuit import Circuit


def convert_circuit(circuit, target_type: Type = str):
    # First convert all types to string
    if isinstance(circuit, Circuit):
        expanded_circuit = []
        for gate in circuit.get_labels():
            if gate.issimple():
                expanded_circuit.append(gate.name)
            elif str(gate) == '[]':
                gate.append('Gi')
            else:
                raise RuntimeError(f'Cannot understand gate {gate}')

    elif isinstance(circuit, str):
        # Replace [] by identity gate
        expanded_circuit = circuit.replace('[]', 'Gi')

        # Expand any expressions ({gates})^exponent
        expand_subcircuit = lambda match: match.group(0).replace(
            match.group(0), match.group(1) * int(match.group(2))
        )
        expanded_circuit = re.sub(r'\(([A-Za-z]+)\)\^([0-9])', expand_subcircuit, expanded_circuit)

        # Remove all single parentheses without exponent
        expanded_circuit = re.sub(r"\(|\)", "", expanded_circuit)

        # Newer circuits end with @(qubit idxs), which should be removed
        expanded_circuit = expanded_circuit.split('@')[0]

        if expanded_circuit == '{}':
            expanded_circuit = ''

        # Split by every letter G
        expanded_circuit = ['G'+ gate for gate in expanded_circuit.split('G')[1:]]
    else:
        raise RuntimeError("circuit type not understood")

    if target_type == str:
        if expanded_circuit:
            return ''.join(expanded_circuit)
        else:
            return '{}'
    elif target_type == list:
        return expanded_circuit
    else:
        raise NameError(f'Cannot select target type {target_type}')


def save_circuits(circuits, filepath):
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.txt')

    with filepath.open('w') as f:
        for circuit in circuits:
            circuit_str = convert_circuit(circuit, target_type=str)
            f.write(circuit_str + '\n')


def load_circuits(filepath, target_type=list):
    filepath = Path(filepath)
    with open(filepath.with_suffix('.txt')) as file:
        lines = [line.strip() for line in file]

        # Check if line already contains data
        if lines[0].startswith('##'):
            # Remove two headers
            lines = lines[1:]

        # Remove any existing results
        lines = [line.split(' ')[0] for line in lines]

        circuits = [
            convert_circuit(line, target_type=target_type) for line in lines
        ]

    return circuits


def analyse_circuit_results(
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