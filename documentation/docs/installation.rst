############
Installation
############

This page explains the steps needed to install both
`SilQ <https://github.com/nulinspiratie/silq>`_ and
`QCoDeS <https://github.com/QCoDeS/qcodes>`_. In this
guide, the computer is assumed to have a Windows OS without Python installed.

.. warning::
  Our group uses a QCoDeS_ fork, which
  is based on a previous version of QCoDeS
  . Please make sure that the main QCoDeS fork is **not** installed
  (not through GitHub nor pip).

==============
Install Python
==============
Both QCoDeS and SilQ are written in Python. To use either, Python must first be
installed. We recommend using Anaconda, which is a Python installation that
comes with many packages including Jupyter Notebook, as well as the Spyder IDE.
Anaconda can be downloaded and installed as follows:

1. Download `Anaconda 64bit python 3.6 <https://www.continuum.io/downloads>`_
   (Newer versions are probably also fine)
2. Install Anaconda (make sure to tick the options add to system path and
   register as python 3.6).

==============
Install Github
==============
GitHub is used as a source control repository for QCoDeS and SilQ. This means
that the source code, as well as the latest developer versions can be downloaded
using GitHub. The following instructions download GitHub Desktop, which is a
basic GUI for managing GitHub repositories.

1. Download and install a Git client, such as
   `Github Desktop <https://desktop.github.com/>`_
2. Create a `Github account <https://github.com/>`_ if you do not already have
   one.
3. Give Serwan your account details to add you to the private git repository
   for SilQ_ (which includes our group's fork of QCoDeS_).

.. note::
  Any modifications of the SilQ/QCoDeS source code should be uploaded using
  `Pull Requests <https://help.github.com/articles/about-pull-requests/>`_ in
  GitHub. See `Developer guide` for more information.

.. note::
  A good alternative to GitHub Desktop is
  `SourceTree <https://www.sourcetreeapp.com/>`_ by Atlassian, which has more
  functionality, but is also harder to get started.
  The die-hards can also directly use `git <https://git-scm.com/>`_ from the
  command line.

=======================
Install QCoDeS and SilQ
=======================

--------------------
Download source code
--------------------
.. warning::
  Ensure QCoDeS is not installed on your computer and follow the instructions
  below in correct order.

QCoDeS and SilQ should be installed from the source code. Both source codes can be
downloaded from GitHub using GitHub Desktop by `cloning <https://help.github
.com/desktop/guides/contributing-to-projects/cloning-a-repository-from-github-desktop/>`_
the repository SilQ_ into your hard drive (e.g. ``User/Documents/Github/SilQ``).

.. note::
  The QCoDeS fork our group uses is a submodule of SilQ, and should be a folder
  in the ``SilQ`` root folder. Please check that the folder is not empty. If it
  is empty, download the QCoDeS source code by entering the following command
  in cmd prompt while in the SilQ root folder::

    git submodule update --init --recursive

--------------
Install QCodes
--------------
After downloading the source code of both SilQ and QCoDeS, they can be installed
and registered in Python. To install the QCoDeS fork of our group, follow
these steps:

1. Open command prompt (type cmd into start menu and right click to open as
   administrator)
2. Navigate to the Qcodes folder on your computer (SilQ/Qcodes),
3. run ``python setup.py develop``
4. If a notification pops up that packages are missing, install these by running
   ``pip install {packages}``, where ``{packages}`` are the packages you
   want to install separated by spaces.

------------
Install SilQ
------------
After QCoDeS is installed, similar directions should be used to install SilQ:

1. In command prompt, navigate to SilQ folder on your computer
   (not ``SilQ/silq`` but one level up).
2. Run ``python setup.py develop``.
3. Install any packages that are missing.

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
capabilities. An alternative is to use an IDE such as Spyder or PyCharm.
The main advantage is that you have direct access to all the code and debugging
facilties.

This section will explain using a third method, namely `Jupyter Notebook
<http://jupyter.org/>`_. It is displayed in a web browser, and is intended to
mimick a physical notebook. Cells are used to run chunks of code, and also to
add text, titles, images, equations etc. One important advantage of using
Jupyter Notebook for SilQ is that it can use iPython widgets. These are
interactive GUI-like elements in the browser (e.g. a button that runs code when pressed).
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
  used as the root folder when starting Jupyter notebook.

.. note::
  If you want to use an IDE instead, and want to execute a script, write your
  commands after::

    if __name__ == "__main__":


=======================
Setup a SilQ Experiment
=======================
.. note::
  This section is only relevant if you are planning on using SilQ.

SilQ uses an Experiment to group together everything related to an experiment.
The Experiment is stored in an Experiment folder that includes initialization
code (e.g. connecting to instruments), and a config containing information and
settings (e.g. resonance frequencies, pulse settings).
It is not necessary to have an Experiment folder to use SilQ, but parts of SilQ
do rely on the existence of a config.

----------------------------------
Setup experiments root folder
----------------------------------
The experiments root folder is a folder on the PC that serves as the root
for all experiments. Each experiment should have its own Experiment folder within
the experiments root folder.

To specify the experiments root folder, run a Python session and execute::

    import silq
    silq.set_experiments_folder(r'{experiments_root_folder}')

where ``{experiments_root_folder}`` should be the absolute path to
your experiments root folder.
Do not forget the ``r`` before the apostrophe, indicating a raw string.

.. note::
  In the Morello group, the `Experiments <https://github.com/nulinspiratie/experiments>`_
  GitHub repository is used as a shared experiments root folder.
  This folder includes our configurations and notebooks, but not our data.
  Ask to be added to the repository.

---------------------------
Setup an Experiment folder
---------------------------
To create an Experiment, add a folder to the experiments root folder with the
name of your experiment.
The Experiment folder should contain at least the following two folders

:init: initialization scripts.
:config: SilQ config, folder contents will be converted to a Python dict.

Additional files/folders can be added to the experiment folder, such as
notebooks, images, etc.

The ``init`` folder contains all initializations python scripts.
They will be executed by alphanumeric order, so start the first script with
`0_`, and increase the index for subsequent files to fix the execution order.

The ``config`` folder contains all information on the experiment.
It can contain ``.json`` files, and folders containing `.json` files.
The config will be combined into a single dictionary, and all folders an files
will be keys and their contents will be values.

-----------------------
Register the Experiment
-----------------------
The final step is to register the experiment.
Create/open the file `configurations.json` in the experiments root
folder.
If the file does not yet exist, fill it with the following information::

    {
        "{Experiment_name}": {
            "folder": "{Experiment_folder_name}",
            "macs": [],
            "modes": {}
            }
        }
    }

If the file already exists, add the information above (minus the outer parentheses)
as an entry in the list.


--------------------------
Initializing an Experiment
--------------------------
After the Experiment folder has been created and the ``init`` and ``config``
folder have been created, the Experiment can be initialized by::

    import silq
    silq.initialize('{experiment_name}')

This will execute all scripts in ``init``, and the config can now be found in
``silq.config``

------------------------
Updating SilQ and QCoDeS
------------------------
As of the time of writing, both SilQ and QCoDeS are under active development.
This means that there are regular features/bugfixes being added
(and hopefully not many new ones introduced).
It is therefore recommended to regularly update your SilQ and QCoDeS to the
latest version.
For this, a Git client such as GitHub Desktop can be used.
To update either, first ensure that you don't have any new changes to the source
code. If you do and they could be useful for others, please see `Developer guide`.
Next, pull the latest changes from `master`, which is the main stable branch.


==============================
Optional setup and information
==============================

These parts of the setup are not necessary, but can be useful.

.. _setup-experiments-folder:



-------------------
Install PyCharm IDE
-------------------
`PyCharm <https://www.jetbrains.com/pycharm/>`_ is a very powerful IDE and can
simplify programming significantly.
To install it, follow these steps:

1. Download, install, and open PyCharm Community Edition (or use the
   professional version which is free for students).
2. Open the SilQ root folder
3. Go to file -> default settings -> project interpreter
4. Set interpreter to the newly installed Python Anaconda 3.6
5. Restart PyCharm