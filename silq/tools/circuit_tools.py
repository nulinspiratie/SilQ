from typing import Type
from pathlib import Path
import numpy as np
import regex as re
pi = np.pi

from pygsti.objects.circuit import Circuit


def convert_circuit(circuit, target_type: Type = str):
    # First convert all types to string
    if isinstance(circuit, Circuit):
        expanded_circuit = [gate.name for gate in circuit.get_labels()]
    elif isinstance(circuit, str):
        expand_subcircuit = lambda match: match.group(0).replace(
            match.group(0), match.group(1) * int(match.group(2))
        )
        expanded_circuit = re.sub(r'\(([A-Za-z]+)\)\^([0-9])', expand_subcircuit, circuit)

        # Remove all single parentheses without exponent
        expanded_circuit = re.sub(r"\(|\)", "", expanded_circuit)

        # Split by every letter G
        expanded_circuit = ['G'+ gate for gate in expanded_circuit.split('G')[1:]]
    else:
        raise RuntimeError("circuit type not understood")

    if target_type == str:
        return ''.join(expanded_circuit)
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
    with open(filepath.with_suffix('.txt')) as file:
        lines = [line.strip() for line in file]

        # Check if line already contains data
        if lines[0].startswith('##'):
            # Remove two headers
            lines = lines[2:]

            # Remove results
            lines = [line.split(' ')[0] for line in lines]

        circuits = [
            convert_circuit(line, target_type=target_type) for line in lines
        ]

    return circuits