**********************
Storing PulseSequences
**********************
Every time a `PulseSequence` is passed to the `Layout` to be targeted to the
experimental setup, a copy of the `PulseSequence` can be stored on the computer.
The `PulseSequence` is stored as a Python ``pickle``, along with a timestamp.
This can be a useful feature for logging, as the timestamp allows you to see
what `PulseSequence` was targeted at a given time.
To setup `PulseSequence` storage, the folder needs to be passed to the
`Layout` during its initialization, as such

>>> layout = Layout(store_pulse_sequences_folder={folder_path})

where ``{folder_path}`` is an absolute path to the folder.

Stored `PulseSequences <PulseSequence>` can be later retrieved either
directly using ``pickle``, or relative to a ``dataset`` using

>>> dataset.get_pulse_sequence()

By default the first pulse sequence will be retrieved that was targeted after
the dataset is created.
Additional arguments can be specified to retrieve a specific later
`PulseSequence`.

.. note::
   Due to the way pickling works, when a `PulseSequence` is retrieved, it
   will use the current version of SilQ QCoDeS to recreate the actual
   `PulseSequence` object.
   If the source code of either has changed significantly in the time between
   storage of the `PulseSequence` and its retrieval, it could be that the
   `PulseSequence` cannot be recreated and an error is raised.
   In this case it is recommended to revert to an earlier version of
   SilQ/QCoDeS, preferrably the version existing when the `PulseSequence` was
   stored.