************************************
Classes and functions index
************************************
.. currentmodule:: silq

Here we list the most important classes and functions.


Parameters
==========
`silq.parameters`

AcquisitionParameters
---------------------
`silq.parameters.acquisition_parameters`

.. autosummary::
   :nosignatures:

   ~parameters.acquisition_parameters
   ~parameters.acquisition_parameters.AcquisitionParameter
   ~parameters.acquisition_parameters.DCParameter
   ~parameters.acquisition_parameters.TraceParameter
   ~parameters.acquisition_parameters.VariableReadParameter
   ~parameters.acquisition_parameters.EPRParameter
   ~parameters.acquisition_parameters.ESRParameter
   ~parameters.acquisition_parameters.T2ElectronParameter
   ~parameters.acquisition_parameters.NMRParameter
   ~parameters.acquisition_parameters.FlipNucleusParameter
   ~parameters.acquisition_parameters.FlipFlopParameter
   ~parameters.acquisition_parameters.BlipsParameter
   ~parameters.acquisition_parameters.PulseSequenceAcquisitionParameter
   ~parameters.acquisition_parameters.DCSweepParameter

General Parameters
------------------
`silq.parameters.general_parameters`

.. autosummary::
   :nosignatures:

   ~silq.parameters.general_parameters.CombinedParameter
   ~silq.parameters.general_parameters.AttributeParameter

MeasurementParameters
---------------------
`silq.parameters.measurement_parameters`

.. autosummary::
   :nosignatures:

   ~silq.parameters.measurement_parameters.MeasurementParameter
   ~silq.parameters.measurement_parameters.CoulombPeakParameter
   ~silq.parameters.measurement_parameters.RetuneBlipsParameter
   ~silq.parameters.measurement_parameters.MeasureNucleusParameter
   ~silq.parameters.measurement_parameters.MeasureFlipNucleusParameter
   ~silq.parameters.measurement_parameters.DCMultisweepParameter
   ~silq.parameters.measurement_parameters.MeasurementSequenceParameter
   ~silq.parameters.measurement_parameters.SelectFrequencyParameter
   ~silq.parameters.measurement_parameters.TrackPeakParameter

Pulses and PulseSequences
=========================

Pulse types
-----------
`silq.pulses.pulse_types`

.. autosummary::
   :nosignatures:

   ~silq.pulses.pulse_types.Pulse
   ~silq.pulses.pulse_types.DCPulse
   ~silq.pulses.pulse_types.SinePulse
   ~silq.pulses.pulse_types.TriggerPulse
   ~silq.pulses.pulse_types.DCRampPulse
   ~silq.pulses.pulse_types.FrequencyRampPulse
   ~silq.pulses.pulse_types.MarkerPulse
   ~silq.pulses.pulse_types.MeasurementPulse

PulsesSequences
---------------
`silq.pulses.pulse_modules`

.. autosummary::
   ~silq.pulses.pulse_modules.PulseSequence

PulseSequence generators
------------------------
`silq.pulses.pulse_sequences`

.. autosummary::
   ~silq.pulses.pulse_sequences.PulseSequenceGenerator
   ~silq.pulses.pulse_sequences.ESRPulseSequence
   ~silq.pulses.pulse_sequences.T2ElectronPulseSequence
   ~silq.pulses.pulse_sequences.NMRPulseSequence
   ~silq.pulses.pulse_sequences.FlipFlopPulseSequence


Analysis
========
`silq.analysis`

Analysis functions
------------------
`silq.analysis.analysis`

.. autosummary::
   :nosignatures:

   ~silq.analysis.analysis.find_high_low
   ~silq.analysis.analysis.edge_voltage
   ~silq.analysis.analysis.find_up_proportion
   ~silq.analysis.analysis.count_blips
   ~silq.analysis.analysis.analyse_traces
   ~silq.analysis.analysis.analyse_EPR
   ~silq.analysis.analysis.analyse_flips


Fitting functions
-----------------
`silq.analysis.fit_toolbox`

.. autosummary::
   :nosignatures:

   ~silq.analysis.fit_toolbox.Fit
   ~silq.analysis.fit_toolbox.LinearFit
   ~silq.analysis.fit_toolbox.ExponentialFit
   ~silq.analysis.fit_toolbox.DoubleExponentialFit
   ~silq.analysis.fit_toolbox.SineFit
   ~silq.analysis.fit_toolbox.AMSineFit
   ~silq.analysis.fit_toolbox.ExponentialSineFit
   ~silq.analysis.fit_toolbox.RabiFrequencyFit


Tools
=====
`silq.tools`

General tools
-------------
`silq.tools.general_tools`

.. autosummary::
   :nosignatures:

   ~silq.tools.general_tools.run_code
   ~silq.tools.general_tools.SettingsClass
   ~silq.tools.general_tools.is_between

Notebook tools
--------------
`silq.tools.notebook_tools`

.. autosummary::
   :nosignatures:

   ~silq.tools.notebook_tools.create_cell
   ~silq.tools.notebook_tools.SilQMagics

Plot tools
----------
`silq.tools.plot_tools`

.. autosummary::
   :nosignatures:

   ~silq.tools.plot_tools.PlotAction
   ~silq.tools.plot_tools.SetGates
   ~silq.tools.plot_tools.MeasureSingle
   ~silq.tools.plot_tools.MoveGates
   ~silq.tools.plot_tools.SwitchPlotIdx
   ~silq.tools.plot_tools.InteractivePlot
   ~silq.tools.plot_tools.SliderPlot
   ~silq.tools.plot_tools.CalibrationPlot
   ~silq.tools.plot_tools.DCPlot
   ~silq.tools.plot_tools.ScanningPlot
   ~silq.tools.plot_tools.TracePlot
   ~silq.tools.plot_tools.DCSweepPlot


Detect peaks
------------
`silq.tools.detect_peaks`

.. autosummary::
   :nosignatures:

   ~silq.tools.detect_peaks.find_transitions
   ~silq.tools.detect_peaks.get_charge_transfer_information
   ~silq.tools.detect_peaks.plot_transitions
   ~silq.tools.detect_peaks.plot_transition_gradient


