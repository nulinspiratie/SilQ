******************
Brief QCoDeS guide
******************

Since QCoDeS is at the foundation of SilQ, we will first give a brief guide
on the main concepts of QCoDeS.
This page is intended to be a short guide that gets the reader introduced to
the most important principles of QCoDeS, and is by no means intended to
be a comprehensive guide.
For more information, please see the documentation website of `QCoDeS
<http://qcodes.github.io/Qcodes/>`_.
However, please remember that our group uses a version of QCoDeS that has
diverged from the main QCoDeS, and so some sections are not relevant to us
(in particular the new DataSet and measuring method).
Additionally, the `in-depth guides <in-depth guides/index>` section contains
information about specific parts of QCoDeS.

Instruments
===========
An experimental setup generally consists of instruments with specific functions.
QCoDeS has a plethora of drivers for different instruments, each of which is
a subclass of the `Instrument` class.
For a list of drivers, see the `instrument drivers webpage <instrument-drivers>`
or the QCoDeS source code.

The `Instrument` is the QCoDeS representation of the physical instrument that
facilitates communication.
Each `Instrument` generally has a connection mode (e.g. GPIB, ethernet) and a
connection address (e.g. ``GPIB0::5::INSTR``) to the physical instrument.

As an example throughout this notebook, we will

Parameters


measurement Loop

Plotting


Dataset

.. _instrument-drivers: http://qcodes.github.io/Qcodes/api/generated/qcodes.instrument_drivers.html
