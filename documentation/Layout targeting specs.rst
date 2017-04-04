===============================
Layout targeting specifications
===============================
When the Layout receives an abstract PulseSequence (usually setup-independent),
it needs to targeted the PulseSequence to the experimental setup. It does
this by distributing the pulses over the interfaces. Each interface then
implements the pulses it has received on its associated instrument, such that
the PulseSequence is played once the acquisition is started.

Current targeting scheme
************************
Currently, targeting of an abstract PulseSequence is performed as follows.
Once the Layout receives the PulseSequence, it will loop over the pulses.

-. For each pulse, it will check ``interface.pulse_implementations`` to see
   which of the interfaces can output a given pulse. Exactly one interface
   must be able to handle a given pulse, or it will raise an error.
-. If a unique interface is found, it checks which connections can be used to
   output a pulse. If more than one connection is found, it will check if any
   of them have attribute ``connection.default = True``. If this is the case,
   or only one connection is found, it will choose this connection, else it
   will raise an error.
-. If a unique connection is found, the connection will first target the pulse.
   This consists of creating a copy of the pulse (needed to keep the original
   pulse abstract), and attaching itself to ``pulse.connection``. It can
   further perform operations, such as modifying ``pulse.amplitude`` by
   ``connection.scale``. If the connection is a CombinedConnection, this will
    furthermore result in several pulses, targeted to each of the underlying
    connections.
-. Next the interface transforms the pulses into PulseImplementations that
   are specific to the interface. First, a PulseImplementation is created,
   after which properties from the pulse are copied over to the
   PulseImplementation. At this stage, the PulseImplementation can require
   additional pulses such as triggering. These pulses are then placed in
   ``PulseImplementation.additional_pulses``. The PulseImplementation has a
   method ``PulseImplementation.implement()``, which implements the pulse for
    a specific interface. The PulseImplementation is then added to
    ``interface.pulse_sequence``.
-. Any additional pulses are also targeted in the same way before looping to
   the next pulse.

Desired features
****************
The current targeting scheme has worked for the relatively simple
measurements we have performed so far in SilQ, but is not very practical for
more complicated measurements. This is especially the case when considering the
upcoming improvements on the PulseSequence (see PulseSequence specifications).

- Easily direct similar pulses to different connections
- Targeting of logic operations
- Dealing with nested PulseSequences
- Handling PulseSequences which have repetitions


Easily direct similar pulses to different connections
-----------------------------------------------------
Figuring out which pulse should go to which connection is currently done
almost entirely autonomous. While this is in principle a good thing, it
breaks down when

- The pulse should go to a connection that is not default
- When there are multiple possible connections, none of which are default
- When there are multiple interfaces that can implement the given pulse

This can be circumvented by giving kwargs to ``pulse.connection_requirements``
that specify the connection it should go to, but this is awkward, and more
importantly counters the whole idea of generalization that SilQ is meant to
achieve. While the user should still be able to specify the connection in
``pulse.connection_requirements``, there should be more convenient
alternatives that keep the code general.

As an example for when we would want to easily specify the connection a pulse
should go to, we look at a multi-qubit system. In this case, pulses are
usually directed to one qubit, each of which has its own associated connections.
If you could therefore connect a pulse to a qubit, you would be able to go
through its connections to extract the connection it should be targeted to.
Having some sort of label/tag associated to a pulse
This label can furthermore be used to extract other parameters, such as the
ESR frequency etc.