****************
Concepts in SilQ
****************
This page describes the main concept in SilQ.
The first section describes the `Main classes <main-classes>` that build up the
layers of
abstraction.
The second section describes how all these classes interact with one another
when `Targeting a pulse sequence <targeting-pulsesequence>` to a specific
experimental setup.
The final section describes the `AcquisitionParameter`, which is the final layer
of abstraction, and whose description

.. _main-classes:
Main classes
============
The classes described here are ordered by how they control one another.
In general, classes later on control the classes described earlier.
Every class described here is a QCoDeS `ParameterNode`, and their properties
are `Parameters <Parameter>`, and so they benefit from all the features
provided by these.
See `Parameter guide <in-depth guides/Parameter guide>` and `ParameterNode
guide <in-depth guides/ParameterNode guide>` for more information.

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

For more information on the `Pulse`, see `Pulses and PulseSequences
<in-depth guides/Pulses and PulseSequences>`.

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


.. _targeting-pulsesequence:

Targeting a pulse sequence
==========================
There are several steps happening when a `PulseSequence` is targeted by the
`Layout` to a specific experimental setup.
To understand the processes that happen behind the scenes, the most important
piece of information is knowing which classes interact with each other, and
if it's a one-way interaction or two-way interaction.
Below is a figure containing a very simple experimental setup (top), and the
corresponding representation in SilQ.

.. image:: images/Pulse\ sequence\ targeting.jpg
  :alt: Alternative text

The experiment shown above is a simplified version of a typical
experimental setup.
It only contains three instruments, and for simplicity we ignore any sample
being experimented on.
A trigger instrument (left) handles the timing of the system by sending
periodic triggers to the other instruments to indicate an event.
The waveform generator (middle) can output waveforms (pulses).
It receives triggers from the trigger instrument to indicate that it should
output the next pulse.
Alternatively, a trigger can indicate that it should output the entire pulse
sequence and wait for the next trigger (this is usually the case for
experiments requiring nanoscale precision).
The waveform generator emits the pulses to the acquisition card, which
is programmed to record a fixed-duration digitized signal when it receives a
trigger from the trigger instrument.
By programming the instruments correctly, the acquisition card can be
setup to record specific pulses from the waveform generator.

Even such a simple measurement as the one described above requires many
commands to be sent to the different instruments.
In SilQ, this is handled by the `Layout` targeting a `PulseSequence` to the
particular experimental setup.
The bottom of the figure shows how the different SilQ objects interact with
one another when targeting a `PulseSequence`.
The arrows indicate the direction of communication, a round dot indicates
being a property of the class the line originates from.
The dotted line indicates there is a `Connection` between the
`InstrumentInterfaces <InstrumentInterface>` (there is also a connection between
the left-most and right-most interface).

Targeting a `PulseSequence` is actually a two-step process.
However, step zero is having preprogrammed all the `Instruments
<Instrument>`, `InstrumentInterfaces <InstrumentInterface>`, and `Layout`.
This does not mean manually sending all the commands to output the pulse
sequence, but specifying the parameters that are freely configurable,
such as the``sample rate``.

The first step is invoked by setting the `Layout` `PulseSequence`:

>>> layout.pulse_sequence = pulse_sequence

In the first step, no instruments are actually configured, but instead the
`Layout` passes the `Pulses <Pulse>` around to the different
`InstrumentInterfaces <InstrumentInterface>`.
These then verify that they can program their instrument to output the pulse,
and optionally request ancillary pulses from the `Layout` (such as trigger
pulses).
If any `InstrumentInterface` is not able to program its instrument to output
all the required pulses, an error is raised.

If the first does not raise any errors, then each of the `InstrumentInterfaces
<InstrumentInterface>` will have its ``InstrumentInterface.pulse_sequence``
filled with the pulses it should output.
Additionally, ``InstrumentInterface.input_pulse_sequence`` contains a list of
pulses that it receives.
This is a good moment to see if the `InstrumentInterfaces
<InstrumentInterface>` have pulse sequences that actually make sense.

The second step consists of programming the `Instruments <Instrument>`.
This is invoked by calling

>>> layout.setup()

At this point the `Layout` signals all the `InstrumentInterfaces
<InstrumentInterface>` to program their `Instruments <Instrument>`.
Each `InstrumentInterface` will convert its `PulseSequence` into `Instrument`
commands, and execute them.
At this stage, errors may also be raised.
This is often the case when an instrument command cannot be executed by the
instrument.

TODO:

>>> layout.acquire()
>>> layout.start()
>>> layout.stop()


AcquisitionParameter
----------------------
