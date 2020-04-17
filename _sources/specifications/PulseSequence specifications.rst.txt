.. role:: red

============================
PulseSequence specifications
============================

The PulseSequence is a class that combines Pulses, and is able to define a
measurement. Its current features are as follows:

Current features
================
- Ability to add/remove/address pulses
- Fixed total duration, determined from the pulses it contains
- Handle enabling/disabling of individual pulses
- Restrictions on adding (un)targeted pulses
- Find pulse(s) based on certain pulse or connection properties
- Find connection based on certain pulse or connection properties
- Check if pulses overlap
- Get transition voltage between two pulses (useful for triggering)
- Initialization stage for PulseSequence (before t=0). To be replaced.


Desired features
================
Although the PulseSequence suits our current needs for performing a measurement,
it lacks some useful features, and is in need of an upgrade. The following
sections highlight the features that are lacking, including a consideration
of issues and implementation.


Nesting PulseSequence within another PulseSequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ability to nest a PulseSequence in another PulseSequence allows a
natural way to break up a complex measurement into different layers.
For example, a PulseSequence could consists of three stages:

#. Initialization - Perform the necessary pulses to initialize the system
#. Manipulation - Manipulate the qubit through pulses
#. Read - Read out the state of the qubit

Each of these stages could be an independent PulseSequence, which are joined
together in the larger PulseSequence. If you combine this with the idea of
registering a PulseSequence in the config, it becomes especially useful. For
instance, the Read stage can be identical for different measurements, and so
you only have to program the Read PulseSequence once, and it can then be
called from all measurements.

Issues
******
One of the main issues is that targeting of a PulseSequence currently
targets individual pulses. Now instead a PulseSequence could consist of not
only Pulses, but also PulseSequences.

One solution is to roll-out the inner PulseSequences, effectively
concatenating their pulses together into one long PulseSequence. This can
become a problem when you combine it with logic operations, as these may
point towards an entire PulseSequence instead of a single pulse.

Another solution is to create an encompassing class, which itself
contains PulseSequences. You can then make the distinction that
PulseSequences have a fixed duration. Combining PulseSequences and
performing logic operations, such as conditional operations, can only be
implemented at the level of the encompassing class. However, this only shifts
the problem, as you could also want to join such encompassing classes together.

The actual problem may well be that the whole targeting process needs an
upgrade.

Implementation
**************
If we assume that targeting is taken care of, nesting PulseSequences should
have the following features:

- Adding a nested PulseSequences should happen the same as adding a pulse.
- A name, which can be used to identify the PulseSequence when it is nested.
- The duration, if it exists. This may not be the case if the PulseSequence
  contains logic operations.
- Each nested PulseSequence should start at ``t=0``. This would probably mean
  that each interface would need to know about all the nested PulseSequences,
  because if it would get a Pulse, it would need to know what PulseSequence
  it belongs to.
- **Optional** If a PulseSequence contains at least one PulseSequence, there
  should always be an active PulseSequence. If a Pulse is then added, it is
  automatically added to a PulseSequence. This way you ensure that a
  PulseSequence contains either Pulses or PulseSequences, but never both. One
  could even create a separate PulseSequence class for PulseSequences that
  contains other PulseSequences, although I'm not sure if its necessary.
- **Optional** Some sort of output data, which can be treated as input variables
  for other PulseSequences. This is related to logic operations.
- **Optional** Analysis within the PulseSequence. For instance, check for blips.


Loop over pulses
~~~~~~~~~~~~~~~~
There are situations where you want to loop over a pulse, or perhaps even over a
PulseSequence. An example is the CPMG sequence, which consists of many
identical pi pulses. Instead of having to add 8192 pi-pulses to a
PulseSequence, which can significantly slow down the measurement, it is much
easier to add a single pulse with a command to apply it a number of times.
This idea could be extended to also iteratively modifying an attribute of the
Pulse being looped.

Features
********
- Loop over a Pulse or a PulseSequence
- The duration of the loop should take the repetitions into account.

Issues
******
- During targeting the loop should remain, i.e. the pulse should not be
  copied 8192 times.

Implementation
**************
Perhaps the easiest way to implement this is to add a
``PulseSequence.repetitions`` attribute, equal to 1 by default. When you then
want to loop over one or more pulses, you wrap them in a PulseSequence and
increase the repetitions. In this case it would be necessary to also somehow
pass along the PulseSequence (which has information on ``repetitions``) during
targeting.

Another solution is to create logic that says: Go back to beginning of pulse
8192 times. However, this feels less elegant than the previous solution, also
since passing along logic can be complicated.

A third solution is to create a LoopPulse, which loops over a pulse. This
also seems less elegant than the first solution, mainly because it becomes
awkward to loop over a PulseSequence (a PulseSequence within a pulse?).


Look up PulseSequence from config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Being able to efficiently store and load PulseSequences eases the reuse of
PulseSequences for different measurements, and also for different sessions. A
good option for this is in a config file.

Features
********
- Load and save PulseSequence to config,
- Easy creation of stored PulseSequence (preferably by creating
  PulseSequence with same name/label as the stored PulseSequence). Optionally
  there could be a flag ``load_from_config``, which only loads if set to True.
- Easy modification of stored PulseSequence. Preferably a file that can be
  easily modified. This could either be a JSON dictionary, or something that
  can be converted to/from a JSON dictionary.
- PulseSequence properties from a config can be overridden when creating an
  instance of the PulseSequence
- When a property in the PulseSequence config is updated, it is immediately
  reflected in the PulseSequence
- The pulses in a PulseSequence can also be loaded from a Pulse config.

Issues
******
If a PulseSequence coincidentally has the same name as one in the config, it
could automatically be loaded even though this is not wanted.

Implementation
**************
QCoDeS has a config file, and it may be good enough for our purposes.
However, modifying such a config is somewhat difficult, and so we might opt
for a separate SilQ config. In this case, we need to ensure that it is also
stored during each measurement.

The name/label of a PulseSequence can be used to identify a PulseSequence in a
config.


Logic operations
~~~~~~~~~~~~~~~~
Advanced instruments, especially those containing an FPGA, can perform logic
operations. As a primary example, either PulseSequence A or PulseSequence B
is performed depending on the outcome of a measurement.

An existing example is the steered initialization, which remains idle until
no blips have been measured for a threshold duration. At the moment, this is
programmed in a hacky way, by letting a PulseSequence have an initialization
stage, after which the actual PulseSequence starts with ``t=0``. Ideally we
would want this to be replaced by a separate nested PulseSequence which ends
with a logic operation.


Issues
******
There are quite a few issues with adding logic to PulseSequences. The main
issue is probably how each of the interfaces will be informed about logic
operations, and how they will deal with it. In fact, most instruments are not
able to deal with general logic operations, but only in very specific cases.
As an example we consider an AWG that outputs pulses sequentially after
each trigger. It is not able to either output pulse A or pulse B depending on
the outcome of a measurement. However, it will be able to wait with
outputting a pulse until it receives a final trigger. The AWG can therefore
implement a subset of all logic operations.

Programming each interface how to discern if it can implement a PulseSequence
containing logic operations can become quite complicated. You would want
interfaces to raise errors if they cannot implement some sort of logic.
However, this means that all relevant interfaces would need to know about the
logic being used

Possible types of logic
-----------------------
- If/elif/else statements. In each of these cases, it should be able to point
  to a point in the PulseSequence, such as a different PulseSequence. It
  could also be other commands, such as ``continue`` and ``stop``.
- Perform analysis, such as check for blips. In this case, it would often
  need to point at the acquisition of data from a previous pulse.
- :red:`More types of logic?`

Implementation
**************
- If there is logic within a PulseSequence, there may not be a well-defined
  duration. For instance if an if statement can point to two different
  PulseSequences. In this case, PulseSequence.duration should either raise an
  error, or return None.
- Logic should not be a child of the Pulse class, but rather a separate class.
  Perhaps even another class for analysis?
- Analysis could be used in interfaces similar to PulseImplementations. In
  this case, instruments such as the Signadyne could have an
  AnalysisImplementation for finding blips.
- Implementing logic could be more tricky, as it clearly does not belong to a
  single interface, but rather affects all interfaces that are involved in
  the measurement.