.. SilQ documentation master file, created by
   sphinx-quickstart on Wed Dec 13 19:43:02 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SilQ's documentation!
================================

SilQ is a measurement software designed to control spin-based systems. It utilizes the QCoDeS data acquisition framework.



.. toctree::
   :maxdepth: 2
   :caption: Contents:


Documentation
-------------


.. toctree::
   :maxdepth: 2

   docs/setup
   specifications/index


# Setting up QCoDeS and SilQ

# Getting started with QCoDeS
  # `Parameter`
  # `Instrument`
  # Measurements: `Loop` and `Measure`
  # `DataSet`
  # Plotting
  # More documentation/examples

# Getting started with SilQ
  # `PulseSequence` and `Pulses`
  # `Layout`
  # `InstrumentInterface`
  # `Connections`
  # `AcquisitionParameter`


# Unsorted

  # Guidelines for an `InstrumentInterface`.
  # SIM GUI
  # Notebook widgets
  # MeasurementParameter
  # General parameters

    # CombinedParameter
    # AttributeParameter

  # SilQ magics

    # Loading data

  # SilQ config
  # PulseSequenceGenerator
  # Interactive plots



Autosummary
------------
.. currentmodule:: silq

.. autosummary::

   parameters.acquisition_parameters.AcquisitionParameter
   parameters.acquisition_parameters.DCParameter
   parameters.acquisition_parameters.DCSweepParameter

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
