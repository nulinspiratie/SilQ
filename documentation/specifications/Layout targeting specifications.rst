===============================
Layout targeting specifications
===============================
When the Layout receives an abstract PulseSequence (usually setup-independent),
it needs to target each pulse in the PulseSequence to the experimental setup. It does
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
Having some sort of label/tag associated to a pulse would allow this.
This label can furthermore be used to extract sample/setup-dependent
parameters such as the ESR frequency.


Targeting strategy
******************
When we assume that the PulseSequence will be upgraded according to the
specifications laid out in PulseSequence specifications, a good strategy
needs to be used to target a PulseSequence. The proposed strategy is as follows:

The config maintains a list of ``environments`` (needs better word). Each
environment has a name/label, and with it a list of associated connections.
These connections each have a label within the environment, and can be linked to
certain pulses (i.e. connection ``qubit1.DC`` corresponds to connection ``DC``
in environment ``qubit1``). This environment furthermore can contain
pulse-specific implementations, such as the duration and frequency of a
pi-pulse. The idea is that the label of such an environment could be
something like ``qubit1``, which then contains information about the
connections used to send pulses to qubit 1, and also information on the
pulses, such as the resonance frequency.

At the first stage of targeting, the Layout will target the PulseSequence to
connections. To this end, the PulseSequence is first copied, such that the
original PulseSequence remains untargeted. Each pulse has an associated
connection label, which is linked to a connection within an environment. This
allows each pulse to be mapped to a unique connection. At the end of this stage,
a copy of the original PulseSequence is generated where all pulses are
targeted to connections. All pulse properties are now also hard properties of
the pulse, i.e. they are no longer extracted from the config.

Next, the pulses are distributed to the interfaces. Instead of looping
through the pulses, interfaces are looped over. This looping should be done
hierarchically (starting with instruments that are not triggering
instruments, then their triggering instruments etc.). For each interface, a
skeleton of the PulseSequence is created, and pulses whose connection
includes the interface is added. The skeleton means everything excluding
the actual pulses. this therefore includes the nested PulseSequence with
relevant properties (such as duration), the analysis, the logic operations,
etc. Transferring the skeleton allows the interfaces to verify if they can
actually implement the PulseSequence.

For each interface, additional pulses may be required, such as triggers. This
is then returned to the original Layout as a secondary PulseSequence skeleton
containing these additional pulses. When the Layout reaches interfaces that
need to implement these additional pulses, it will add these pulses to the
original PulseSequence skeleton.

This strategy allows straightforward implementation of the desired features
mentioned above. First of all, nested PulseSequences are also passed along,
as they are part of the PulseSequence skeleton. The same holds for logic
operations and PulseSequences with repetitions, as these also belong to the
skeleton. Furthermore, the ``environment`` facilitates directing similar pulses
to different connections.