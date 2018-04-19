
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
