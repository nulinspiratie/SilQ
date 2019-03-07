=======================
Experiment folder guide
=======================

^^^^^^^^^^^^^^^^^^^^
Initialization files
^^^^^^^^^^^^^^^^^^^^
An experimental setup (or personal computer) often requires code that should be
initialized at the start of your measurement session. This includes loading
often-used packages, initializing instruments, setting up the Layout, etc.

Instead of having all this code at the start of your measurement code,
SilQ provides an alternative, namely by creating a list of initialization files
that can be run easily. If the initialization belongs to a measurement, the
correct folder is ``experiments\{experiment_name}\init``, while for personal
initialization code the correct folder is
``personal\{username}\{whatever_you_prefer}``.
It is recommended that you create separate files for distinct types of
initialization code (e.g. package imports, instruments). The files will be run
in alphabetical order, so it is useful to add a numerical prefix to indicate the
loading order.

Furthermore, you should add a label to ``silq\configurations.py``, with
information on the location of the initialization folder. You can also add
additional ``modes``, which can for instance ignore some of the files (useful
when you want to operate in ``analysis`` mode, etc.). Once the initialization
code has been written and a label has been assigned, you can run the
initialization code using::

  import silq
  silq.initialize({label}, mode={mode})

where mode is optional.