****************
Concepts in SilQ
****************
This page describes the main concept in SilQ.
The first section describes the :ref:`Main classes` that build up the layers of
abstraction.
The second section describes how all these classes interact with one another
when :ref:`Targeting a pulse sequence` to a specific experimental setup.


Main classes
============
The classes described here are ordered by how they control one another.
In general, classes later on control the classes described earlier.

`InstrumentInterface`
---------------------
Each instrument has a corresponding QCoDeS `Instrument` driver that facilitates
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


`Connection`
------------
In an experimental setup, instruments are physically connected to one another
by cables.
This physical connection is represented by the `Connection`, which has an input
and output instrument and channel.
It can additionally have flags, such as being a trigger connection, or having
a scale (attenuation).
Connections can also be combined into a `MultiConnection`, which can be useful
when you want a single pulse to be sent to multiple connections.

It is convenient to identify a connection by a label.
This way, a pulse can be passed the same connection_label to ensure it is passed
to that specific connection.
Surprisingly, this helps keeping the pulse sequence setup-independent.
For example, a pulse having the connection_label ``output`` can be directed to
completely different connections in different setups, as the `Connection` having
label ``output`` can differ.


`Layout`
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


`PulseSequence` and `Pulse`
---------------------------

`AcquisitionParameter`
----------------------



Targeting a pulse sequence
=============================
