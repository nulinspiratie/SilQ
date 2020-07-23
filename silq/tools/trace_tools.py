"""Modules related to manipulation of traces saved with datasets"""

import os
import numpy as np
import h5py
import logging
import qcodes as qc
from typing import Sequence, Dict

from silq.tools.general_tools import slice_length

log = logging.getLogger(__name__)


def load_traces(dataset: qc.DataSet, name: str = None, mode: str = "r"):
    """Load traces HDF5 file from a dataset

    Args:
        dataset: Dataset from which to load traces.
        name: Optional name to specify traces file. Should be used if more than
            one parameter is used in the measurement that saves traces.
        mode: Open mode (default is 'r' for read-only)

    raises:
        FileNotFoundError if no trace file can be found in dataset folder
        SyntaxError if more than one trace file is found, and no unique file
            can be determined from the ``name`, if provided
    """
    data_path = dataset.io.to_path(dataset.location)
    traces_path = os.path.join(data_path, "traces")
    trace_filenames = os.listdir(traces_path)
    trace_filenames = [
        filename for filename in trace_filenames if filename.endswith(".hdf5")
    ]

    if not trace_filenames:
        raise FileNotFoundError(f"No trace files found in {traces_path}")

    if name is None and len(trace_filenames) == 1:
        trace_filename = trace_filenames[0]
    else:  # Multiple traces files found or specific name provided
        if name is None:
            raise SyntaxError(
                f"No unique trace file found: {trace_filenames}. "
                f"Trace filename must be provided"
            )
        filtered_trace_filenames = [
            filename for filename in trace_filenames if filename.startswith(name)
        ]
        if len(filtered_trace_filenames) != 1:
            raise SyntaxError(f"No unique trace file found: {trace_filenames}.")

        trace_filename = filtered_trace_filenames[0]

    trace_filepath = os.path.join(traces_path, trace_filename)
    trace_file = h5py.File(trace_filepath, mode)
    return trace_file


def extract_pulse_slices_from_trace_file(traces_file: h5py.File,
                                         sample_rate=None):
    """Extract the slices in trace arrays for each pulse.

    Newer trace files (March 2020 onwards) contain the key ``pulse_slices``,
    which is then returned.


    Older trace files do not have this key, and so the start and stop idx is
    retrieved from the ``pulse_sequence``, which is stored as a key.

    Args:
        traces_file: Opened hdf5 file containing traces and information about
            pulse sequence
    """
    if "pulse_slices" in traces_file:
        pulse_slices = dict(traces_file["pulse_slices"].attrs)
        for key, val in pulse_slices.items():
            pulse_slices[key] = slice(*val)
        return pulse_slices

    elif "pulse_sequence" in traces_file:
        capture_full_traces = traces_file.attrs["capture_full_trace"]
        if sample_rate is None:
            sample_rate = traces_file.attrs["sample_rate"]

        pulses = traces_file["pulse_sequence"]["pulses"].values()

        pulse_slices = {}
        for k, pulse in enumerate(pulses):
            if not pulse.attrs["acquire"]:  # Pulse was not acquired
                continue

            # Create pulse name
            name = pulse.attrs["name"]
            if "None" not in str(pulse.attrs["id"]):
                name += f"[{pulse.attrs['id']}]"

            # Extract pulse start time
            if "t_start" in pulse.attrs:
                t_start = pulse.attrs["t_start"]
            elif "t_start (s)" in pulse.attrs:
                t_start = pulse.attrs["t_start (s)"]
            else:
                raise RuntimeError(f"Error: pulse {name} has no known t_start.")
            start_idx = int(t_start * sample_rate)

            # Extract pulse stop time
            if "t_stop" in pulse.attrs:
                t_stop = pulse.attrs["t_stop"]
            elif "t_stop (s)" in pulse.attrs:
                t_stop = pulse.attrs["t_stop (s)"]
            else:
                raise RuntimeError(f"Error: pulse {name} has no known t_stop.")
            stop_idx = int(t_stop * sample_rate)

            pulse_slices[name] = slice(start_idx, stop_idx)

        # If capture_full_traces == False, acquisition was started from first
        # pulse with acquire == True onwards.
        if not capture_full_traces:
            # Find  start time of measurement
            min_pts = min(pulse_slice.start for pulse_slice in pulse_slices.values())

            # Subtract start of acquisition from each slice
            for pulse, pulse_slice in pulse_slices.items():
                pulse_slices[pulse] = slice(
                    pulse_slice.start - min_pts, pulse_slice.stop - min_pts
                )

        return pulse_slices

    else:
        raise RuntimeError(
            "Pulse sequence information not found in traces file. "
            "Could not segment traces."
        )


def load_pulse_traces(dataset: qc.DataSet = None,
                      traces_file: h5py.File = None,
                      name: str = None,
                      channels: Sequence[str] = ('output', ),
                      array_slices: tuple = (),
                      maximum_array_size: float = 50e6) -> Dict[str, np.ndarray]:
    """Load segmented pulse traces from a trace file stored with a dataset

    Args:
        dataset: QCoDeS dataset from which to retrieve traces.
            If not provided, must provide traces_file
        traces_file: HDF5 file containing traces.
            If not provided, must provide traces_file
        name: Name of traces file. Only relevant if a dataset is passed, and
            it contains multiple trace files.
        channels: List of digitizer channels from which to retrieve traces.
            Must be a subset of all saved channels
        array_slices: Optional array slices along dimensions.
            Useful for not loading the entire traces array at once as the 
            total size can exceed available memory.
            Each element in the tuple corresponds to the respective dimension,
            and can either be an index, or a `slice`.
        maximum_array_size: Maximum entries of all pulse trace arrays.
            If size exceeds this value, an error is raised

    Returns:
        Dictionary with key for each pulse.
        If only one channel is passed, the corresponding value is a numpy array
        with the corresponding traces. If multiple channels are passed, each
        value is a dictionary with a key for each channel and corresponding
        trace array.

    Raises:
        SyntaxError: If a channel is passed that has not been stored.
        OverflowError: If the size of the trace arrays exceed maximum_array_size.
        RuntimeError: If the pulse slices could not be extracted from the file.
    """
    assert dataset is not None or traces_file is not None

    if traces_file is None:
        traces_file = load_traces(dataset, name=name)

    if not all(channel in traces_file['traces'] for channel in channels):
        raise SyntaxError('Could not find all channels. Available channels are '
                          f'{list(traces_file["traces"].keys())}')

    # Extract pulse sequence from traces file
    pulse_slices = extract_pulse_slices_from_trace_file(traces_file)

    # Verify that pulse trace segment sizes don't exceed a threshold
    total_pulse_pts = sum(pulse_slice.stop - pulse_slice.start
                          for pulse_slice in pulse_slices.values())
                          
    # The traces from all channels are assumed to have the same dimension (sampling rate etc.)
    traces = traces_file['traces'][channels[0]]

    # Determine size of array slices
    if array_slices:
        total_trace_pts = np.prod([
            slice_length(s, length) for s, length in zip(array_slices, traces.shape)
        ])
        total_trace_pts *= np.prod(traces.shape[len(array_slices):-1])
    else:
        total_trace_pts = np.prod(traces.shape[:-1], dtype=np.int64)

    total_trace_pts *= total_pulse_pts * len(channels)
    log.debug('Total trace points to retrieve: %d', total_trace_pts)
    if total_trace_pts > maximum_array_size:
        raise OverflowError(f'Total size of traces {total_trace_pts/1e6}M '
                            f'exceeds limit {maximum_array_size/1e6}M. '
                            f'Total trace_shape: {traces.shape}. '
                            f'Please increase maximum_array_size or pass array_slice')

    # Segment traces into pulse trace segments
    # If there is only one channel, don't make a separate dict for each channel.
    pulses_traces = {}
    for pulse_name, pulse_slice in pulse_slices.items():
        idxs = (*array_slices, Ellipsis, pulse_slice)
        if len(channels) == 1:
            pulses_traces[pulse_name] = traces_file['traces'][channels[0]].__getitem__(idxs)
        else:
            pulses_traces[pulse_name] = {}
            for channel in channels:
                pulse_traces = traces_file['traces'][channel].__getitem__(idxs)
                pulses_traces[pulse_name][channel] = pulse_traces

    return pulses_traces
