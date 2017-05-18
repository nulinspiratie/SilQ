=====================
Config specifications
=====================

The QCoDeS config is a useful feature that facilitates the storage of
information such that it can be accessed by different objects. It furthermore
has the advantage that it is saved along with every measurement. Using the
config in SilQ is very useful, for instance because it allows pulse
properties to be stored globally, such that these properties can be shared by
different instances of the same pulse. Changing a value of a property in the
config is automatically reflected in all of these instances (see
PulseSequence specifications).

Desired features
****************
- Easy way of modifying data. Currently the two ways are either within python
  (by changing the dictionary and saving) or modifying the json file. Perhaps
  some sort of generator would be better.
- Environments. These should contain pulse properties, connections, and
  possibly more.
- Storing connections and loading them into the Layout
