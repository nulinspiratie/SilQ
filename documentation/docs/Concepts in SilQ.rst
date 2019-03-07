****************
Concepts in SilQ
****************
This page describes the main concept in SilQ.
The first section describes the `Main classes` that build up the layers of
abstraction.
The second section describes how all these classes interact with one another
when `Targeting a pulse sequence` to a specific experimental setup.
The final section describes the `AcquisitionParameter`, which is the final layer
of abstraction, and whose description

Main classes
============
The classes described here are ordered by how they control one another.
In general, classes later on control the classes described earlier.
Every class described here is a QCoDeS `ParameterNode`, and their properties
are `Parameters <Parameter>`.

InstrumentInterface
---------------------
Each instrument has a corresponding QCoDeS :class:`Instrument` driver that
facilitates
communication with the user via Python. These drivers usually directly copy the
commands described in the instrument manual, and occasionally add some features.
As a result, each Instrument is controlled slightly differently.
On top of this, different instruments that are meant to perform similar tasks,
such as arbitrary waveform generators (AWGs), can have a completely different
way of controlling.

To be able to start with a setup-independent pulse sequence, an interface is
needed that can convert generic instructions, such as outputting a pulse, to
instructions specific for the particular instrument. This is exactly what the
`InstrumentInterface` does.
When a pulse sequence is being setup (targeted), the `InstrumentInterface`
receives a list of pulses that it should output when the experiment starts.
This will usually be a subset of the whole pulse sequence, plus potentially
ancillary pulses.
The `InstrumentInterface` then converts these pulses into specific instrument
instructions and sets up the instrument.
An `InstrumentInterface` has a list of pulses that it can program, defined
in its attribute ``pulse_implementations``, and if it receives any pulse that
is not defined here, it will raise an error.

Furthermore, the `InstrumentInterface` may request additional pulses, such
as triggering pulses or modulation pulses.
These get sent back to the `Layout` (described next), which will then direct
those to the appropriate `InstrumentInterface`.

.. note::
    While an `InstrumentInterface` is supposed to set all the parameters of the
    `Instrument` relevant to outputting a pulse, often there are simply too many
    parameters, and some are not included.
    However, these are usually parameters that are rarely modified.
    Additionally, the `InstrumentInterface` itself has parameters that can be
    set by the user, and will influence how it programs the `Instrument`.


Connection
------------
In an experimental setup, instruments are physically connected to one another
by cables.
This physical connection is represented by the `Connection`, which has an input
and output instrument and channel.
It can additionally have flags, such as being a trigger connection, or having
a scale (attenuation).
Connections can also be combined into a `CombinedConnection`, which can be useful
when you want a single pulse to be sent to multiple connections.

It is convenient to identify a connection by a label.
This way, a pulse can be passed the same connection_label to ensure it is passed
to that specific connection.
Surprisingly, this helps keeping the pulse sequence setup-independent.
For example, a pulse having the connection_label ``output`` can be directed to
completely different connections in different setups, as the `Connection` having
label ``output`` can differ.


Layout
--------
The `Layout` is at the heart of the experimental setup.
Its basis is being a layout of all the instruments and the connectivity between
them.
A `PulseSequence` is passed onto the `Layout`, which will then use its knowledge
of the experimental setup to direct each of the pulses to the appropriate
`InstrumentInterface`.
If an `InstrumentInterface` requests additional pulses, the `Layout` can find
the appropriate `InstrumentInterface` using its knowledge of the connectivity
between instruments.
The `Layout` also communicates with the data acquisition instrument (via its
interface) to perform data acquisition.

.. note::
    The `Layout` never directly communicates with an `Instrument`, but always
    via the corresponding `InstrumentInterface`.


Pulse
-----
A `Pulse` is a representation of a physical pulse sent in an experiment.
There are many different `Pulse` subclasses, common ones are `DCPulse`,
`SinePulse`, `TriggerPulse`.
These pulses usually have several attributes, such as a ``name``, ``amplitude``,
``duration``, and ``frequency``.
In an experiment, a pulse is always attached to a particular connection
(e.g. an AWG outputting a pulse from one of its channels to the input of a gate
on your device sample).
This is reflected in the `Pulse`, which is linked to a specific `Connection`.
This means that when the pulse is targeted by the `Layout`, the `Connection`'s
output `InstrumentInterface` will be instructed to program the `Pulse` to that
particular connection.
A `Pulse` also has ancillary properties, such as ``acquire``, which signals the
`Layout` that the signal during this pulse should be acquired by the data
acquisition instrument.

Often, pulses with a specific name are reused, either in a `PulseSequence`, or
in different `PulseSequences <PulseSequence>`.
Instead of having to specify all the `Pulse` properties every time,
properties belonging to a `Pulse` with a specific name can be stored in the
config.
This way, any time a new `Pulse` with that name is created, it will use those
properties by default.

For more information on the `Pulse`, see `Pulses and PulseSequences`.

PulseSequence
-------------
An experiment usually consists of a sequence of pulses being output by different
instruments at precise timings.
In SilQ, this is represented by the `PulseSequence`, which contains
`Pulses <Pulse>` that
have specific start times, durations, and `Connections <Connection>`.
A `PulseSequence` can be passed onto the `Layout`, which then targets the
`PulseSequence` to the particular experimental setup by passing its `Pulses
<Pulse>`
along to the `InstrumentInterfaces <InstrumentInterface>`, which then set up
their instruments.

If the properties of the `InstrumentInterfaces <InstrumentInterface>` and
`Layout` have been
configured, passing a `PulseSequence` to the `Layout` is sufficient to execute
the pulse sequence, and obtain the resulting traces from the data acquisition
interface.

For more information on the `PulseSequence`, see `Pulses and PulseSequences
<in-depth guides/Pulses and PulseSequences>`.

.. note::
    Incorporating feedback routines into the pulse sequence is one of the
    future goals.

Targeting a pulse sequence
=============================

The diag

.. image:: images/Pulse\ sequence\ targeting.jpg
  :alt: Alternative text



AcquisitionParameter
----------------------
