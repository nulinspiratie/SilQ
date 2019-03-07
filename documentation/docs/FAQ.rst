**************************
Frequently asked questions
**************************


===================================================================
An error has occurred and I cannot run my experiment. What do I do?
===================================================================
The first step is to look at the traceback of the error. Is there any obvious
mistake? a tool that may help you navigate through the traceback is the `IPython
debugger <https://hasil-sharma.github.io/2017/python-ipdb/>`_, which can be
called by executig ``%ipdb``.

If you can't find any, the next step is to ensure SilQ and QCoDeS are
updated to the latest version. It could be that you've encountered a bug that
has been fixed in a newer version.

If this does not solve the problem, it is best to notify the developers.
You can either let us know in person, or if that's not an option, you can `raise
an issue in GitHub <https://github.com/nulinspiratie/SilQ/issues>`_.

Finally, if you want to get your hands dirty, you can browse the source code and
see if you can figure out what the cause is for the issue. If you find the cause
and a way to fix it, you can submit a pull request so that others may benefit
from your bugfix (see `Developer guide`).
Congratulations, you have just become a developer!


=================================================
Where do I find information about <insert thing here>?
=================================================
Direct information on any object/function can be found using its docstring (the
piece of text describing it).
This can be quickly accessed by placing a question mark before the object name::

    ?thing

In Jupyter notebook, you can additionally press ``shift+tab`` to open a docstring
popup. Pressing it multiple times creates larger and larger popups.
The docstring  can also be found on the documentation website.

Though we try to have a proper docstring for all classes and functions, we do
occasionally miss some.
In the case of objects, it could be that the parent class does have a proper
docstring.
So for example, if you have a `DCParameter` object, and it's missing a docstring
for its ``.acquire()`` method, its parent class (`AcquisitionParameter`)
does have a docstring (`AcquisionParameter.acquire`).

If a docstring cannot be found, it may be possible that there is a guide in the
documentation that describes how to use it. Searching in the documentation
website may lead to information.

Finally, if no information can be found from docstrings/guides, the remaining
options are to either ask the developers directly, or to look through the source
code.


=====================================
Can SilQ be used for our experiments?
=====================================
.. _can-silq-be-used-for-our-experiments:
SilQ has been developed with the goal of being generic enough to be usable by
systems other than donor qubits in silicon.
The degree to which this is possible will depend on how similar the system is.
For example, systems that have similar requirements as donor qubits and use the
same instruments will likely need next to no modifications.
Systems that have different constraints (such as nanoscale pulse timing) would
likely need some instrument interfaces to be modified, as the currently behaviour
is to require one trigger per pulse.
Systems are require significantly more advanced pulse sequences than ours, such
as requiring complex feedback algorithms, are not supported at the moment
(although this is one of the future goals of SilQ).
For systems using different instruments than the ones supported by SilQ, the
trickiest part is likely the development of Instrument interfaces. For more
information, see `InstrumentInterface guidelines <in-depth
guides/InstrumentInterface guidelines>`.