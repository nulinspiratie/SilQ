##########################
Setting up QCoDeS and SilQ
##########################

This guide explains the steps needed to install both SilQ and QCoDeS. In this
guide, the computer is assumed to not have Python installed.

.. warning::
  Our group uses a `QCoDeS <https://github.com/QCoDeS/qcodes>`_ fork, which
  contains some modifications, mostly for compatibility with SilQ. Please make
  sure that the main QCoDeS fork is not installed (neither through GitHub nor
  ``pip``).

==============
Install Python
==============
Both QCoDeS and SilQ are written in Python. To use either, Python must first be
installed. We recommend using Anaconda, which is a Python installation that
comes with many packages including Jupyter Notebook, as well as the Spyder IDE.
Anaconda can be downloaded and installed as follows:

1. Download `Anaconda 64bit python 3.6 <https://www.continuum.io/downloads>`_.
2. Install Anaconda (make sure to tick the options add to system path and
   register as python 3.6).

==============
Install Github
==============
GitHub is used as a source control repository for QCoDeS and SilQ. This means
that the source code, as well as the latest developer versions can be downloaded
using GitHub. The following instructions download GitHub Desktop, which is a
basic GUI for managing GitHub repositories.

1. Download and install `Github Desktop <https://desktop.github.com/>`_ and
   create a `Github account <https://github.com/>`_.
2. Give Serwan your account details to add you to the private git repository
   for `SilQ <https://github.com/nulinspiratie/silq>`_ (which includes our
   group's fork of `QCoDeS <https://github.com/nulinspiratie/qcodes>`_).

.. note::
  Any modifications of the SilQ/QCoDeS source code should be uploaded using
  `Pull Requests <https://help.github.com/articles/about-pull-requests/>`_ in
  GitHub.

.. note::
  It is not necessary to use GitHub Desktop. A good alternative is SourceTree
  by Atlassian, which has more functionality, but is also harder to get started.

=======================
Install QCoDeS and SilQ
=======================

--------------------
Download source code
--------------------
.. warning::
  Ensure QCoDeS is not installed on your computer and follow below instructions
  in correct order.

QCoDeS and SilQ should be installed from the source code. The source code can be
downloaded from GitHub using GitHub Desktop by `cloning <https://help.github
.com/desktop/guides/contributing-to-projects/cloning-a-repository-from-github-desktop/>`_ the repository
`SilQ <https://github.com/nulinspiratie/SilQ>`_ into your hard drive (e.g.
``User/Documents/Github/SilQ``).

.. note::
  The QCoDeS fork our group uses is a submodule of SilQ, and should be a folder
  in the ``SilQ`` folder. Please check that the folder is not empty. If it
  is empty, download the QCoDeS source code by entering the following command
  in cmd prompt while in the SilQ root folder::

    git submodule update --init --recursive

--------------
Install QCodes
--------------
After downloading the source code of both SilQ and QCoDeS, they can be installed
and registered in Python. To install our the QCoDeS fork of our group, follow
these steps:

1. Open command prompt (type cmd into start menu and right click to open as
   administrator)
2. Navigate to the Qcodes folder on your computer (SilQ/Qcodes),
3. run ``python setup.py develop``
4. If a notification pops up that packages are missing, install these:
   Run ``pip install {packages}``, where ``{packages}`` are the packages you
   want to install separated by spaces.

------------
Install SilQ
------------
After QCoDeS is installed, similar directions should be used to install SilQ:

1. Open Command prompt.
2. Navigate to SilQ folder on your computer (not SilQ/silq but one level up).
3. Run ``python setup.py develop``.
4. Install any packages that are missing.

After completing these steps, both SilQ and QCoDeS can be imported and used in
Python via::

  import silq
  import qcodes as qc

==========
Using SilQ
==========
The measurement software SilQ and QCodes can be imported as a package, just like
any other python packages. To import SilQ and QCoDeS, use the following lines::

  import qcodes as qc
  import silq

If there are no errors, both QCoDeS and SilQ should be installed correctly.

----------------
Jupyter Notebook
----------------
There are several ways to run measurement code with SilQ and QCoDeS.
One can use an iPython terminal, which is straightforward but has limited
capabilities. And alternative is to use an IDE such as Spyder or PyCharm.
The main advantage is that you have direct access to all the code and debugging
facilties.

This section will explain using a third method, namely `Jupyter Notebook
<http://jupyter.org/>`_. It is displayed in a web browser, and is intended to
mimick a physical notebook. Cells are used to run chunks of code, and also to
add text, titles, images, equations etc. One important advantage of using
Jupyter Notebook for SilQ is that it can use iPython widgets. These are
interactive elements in the browser (e.g. a button that runs code when pressed).
In SilQ, several widgets have been created for Jupyter Notebook, and more are on
the way. These are aimed at simplifying the use of SilQ.

To run Jupyter notebook, open a terminal, navigate to an appropriate root folder
and run::

  jupyter notebook

This will create a notebook server and open it in the browser. Here you can
navigate to folders and create notebooks. `Getting started with Jupyter Notebook
<https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter
-notebook-for-python-4e7082bd5d46>`_.

.. note::
  If using an `Experiments folder <setup-experiments-folder>`, this should be
  used as the root folder.

.. note::
  If you want to use an IDE instead, and want to execute a script, write your
  commands after::

    if __name__ == "__main__":

==============
Optional setup
==============

These parts of the setup are not necessary, but can be useful.

.. _setup-experiments-folder:

------------------------
Setup Experiments folder
------------------------
.. note::
  The Experiments folder is only used in SilQ, and so these steps are not
  necessary if you're planning on only using QCoDeS.

The Experiments folder is a folder on the PC that serves as the root folder for
all experiments. Each experiment should have its own Experiment folder within
the Experiments folder, and contains

Each Experiment folder should at least contain the following two folders:

:init: Initialization files, see `initialization-files`.
:config: SilQ config

Additional files/folders can be added to the experiment folder, such as
notebooks, images, etc.

Once the Experiments folder has been created, the following code should be run
once to point SilQ to the experiments folder::

    import silq
    silq.set_experiments_folder(r'{experiments_folder}')

where you should replace ``{experiments_folder}`` with the absolute path to your
Experiments folder. Note that you should write the path as a raw string:
``r'{experiments_folder}'``.

.. note::
  In the Morello group, a GitHub repository is used as a shared Experiments
  folder: `Experiments <https://github.com/nulinspiratie/experiments>`_.
  This folder includes our configurations and notebooks, but not our data.

.. todo::
  Provide more info on=config folders.

.. _initialization-files:

^^^^^^^^^^^^^^^^^^^^
initialization files
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

-------------------
Install PyCharm IDE
-------------------
PyCharm is a very powerful IDE and can simplify programming significantly.
To install it, follow these steps:

1. Download, install, and open PyCharm Community Edition (or use the
   professional version which is free for students).
2. Open SilQ project
3. Go to file -> default settings -> project interpreter
4. Set interpreter to the newly installed Python Anaconda 3.6
5. Restart PyCharm



============
Using GitHub
============

1.	Commit your changes to the master branch including a meaningful heading and commentary
2.	Sync with the master branch to get other people’s changes
