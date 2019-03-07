**************************************
Guidelines for the InstrumentInterface
**************************************


=====================
Adding new interfaces
=====================
Adding a new instrument interface involves changing a handful of existing files
and creating some necessary new files.
This section gives a quick overview on how to add a new (empty) interface.

-----------------------
Changing existing files
-----------------------
``silq/instrument_interfaces/__init__.py``

Add a new line for importing the new interface class::

  from .<company>.<model>_interface import <model>Interface

Add a new entry in the ``instrument_interfaces`` dictionary::

  ``<instrument_class_name>``: <model>Interface,

This connects the new interface to the corresponding Qcodes driver of the instrument

------------------
Creating new files
------------------
The following files need to be created::

# ``silq/instrument_interfaces/<company>/<model>_interface.py``

This is the main file containing most of the interface functionality. This
file should contain a class specific for ``<model>` called ``<model>Interface``
which inherits from `InstrumentInterface` and should implement all abstract
methods.

# ``silq/instrument_interfaces/<company>/__init__.py``

This file is used for importing (possibly multiple) interface(s) of a specific
company. Add a new line to do so::

  from .<model>_interface import <model>Interface

--------------------------------
Instrument Interfaces guidelines
--------------------------------

::::::::::
Triggering
::::::::::
The duration of a trigger pulse is specified by the receiving instrument
interface. Therefore, if the interface needs to be or can be triggered by
another instrument, it should have a parameter called ``trigger_in_duration``.
This value can be set by the user, and is used when requesting additional
trigger pulses.


::::::::::::::::::::::
Acquisition interfaces
::::::::::::::::::::::
Should have the following parameters:

acquisition_channels
  Names of acquisition channels [chA, chB, etc.]. Set by the layout.

sample rate
  Acquisition sampling rate (Hz)

samples
  Number of times to acquire the pulse sequence.

capture_full_trace
  Capture from t=0 to end of pulse sequence. False by default, in which case
  start and stop times correspond to min(t_start) and max(t_stop) of all pulses
  with the flag acquire=True, respectively. Setting to True is useful for
  viewing/storing the full traces.

points_per_trace
  Number of points in a trace.

Should furthermore have the the attributes:

traces
  Dictionary of raw unsegemnted traces, of shape {channel_name: channel_traces}

pulse_traces
  Dictionary containing traces segmented per pulse.
  Shape is: {pulse_name: {channel_name: pulse_channel_traces}}
