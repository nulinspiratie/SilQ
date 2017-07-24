====================
Pulse specifications
====================

In a measurement, a PulseSequence is composed of Pulses. Each of these Pulses
can either be untargeted (generic) or targeted (specific to a connection and
interface). The user is supposed to write untargeted pulses, which then are
converted to targeted pulses through the Layout. Currently, the Layout tries
to match each Pulse to a specific connection by searching through the
available interfaces and their PulseImplementations.
However, there are situations where you want a pulse to be directed to a
specific connection, in which case the Layout fails. A solution is proposed
in `Layout targeting strategy`, where each Pulse has an associated
`environment` and `connection_label`. An environment is a dictionary stored
in the config, which contains among others a dictionary of
`connection_label: connection` pairs. This way, a pulse gets sent to a
specific connection depending on the environment it is using.

You can think of the `environment` as something that gives context to the
pulse from which the pulse can gather environment-specific information and
properties.

Desired attributes
******************
name
    The name of a Pulse is a unique identifier that is used to select the
    Pulse from a PulseSequence. If the name contains a dot (.), this denotes
    that the pulse is in a nested PulseSequence. In this case the form is
    '{PulseSequence}.{Pulse}'

label(s)
    Measurements often share common pulses (such as a `read` pulse). These
    pulses should share common properties, meaning that changing the property
    globally should change it for all instances. This is done by linking
    Pulses to the config. A Pulse can retrieve its properties from the config,
    and changing the value in the config is reflected in all related pulses.
    A label can be used to specify where in the config a pulse should retrieve
    its properties from. For instance, a pulse with label 'pi' should
    retrieve the frequency, duration, and amplitude of a pi-pulse. There
    could potentially be multiple labels attached, in which case properties
    from multiple parts of the config can be used. However, this may make
    things more complicated.

environment
    The environment of a Pulse defines where the Pulse can retrieve certain
    properties from. Examples are the connection it should
    be targeted to (defined in connection_attribute), and properties such as
    its frequency. The environment is defined in the config.

connection_label
    The connection_label is a label that is a key in the
    environment's connections, along with an associated connection value.
    This connection is a direct representation of a connection defined in the
    Layout. This way, each Pulse has a connection_label. Depending on the
    environment, this Pulse gets sent to a certain connection.


Open questions
**************

How should a Pulse retrieve properties from the config?
    Something along the lines of a label should be used to link a Pulse with
    a set of properties defined in the config. An example is a `read` pulse,
    which should retrieve the duration and voltage level of a `read` pulse.
    However, there are more complicated cases where a single label might not
    be sufficient.