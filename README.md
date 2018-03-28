# SilQ

SilQ is a measurement software designed to control spin-based systems. It utilizes the QCoDeS data acquisition framework.


## Installation

### Install Python
1.	Download [Anaconda 64bit python 3.6](https://www.continuum.io/downloads)
2.	Install Anaconda (make sure to tick the options add to system path and register as python 3.6

### Install Github
1.	Download and install [Github Desktop](https://desktop.github.com/) and create a [Github account](https://github.com/)
2.	Give Serwan your account details to add you to the private git repository

### Install SilQ
<span style="color:red">Ensure QCoDeS is not installed on your computer and
follow below instructions in correct order</span>
1.	Clone the repository [SilQ](https://github.com/nulinspiratie/SilQ) (includes QCodes) into your hard drive
    (e.g. User/Documents/PythonSoftware) using the installed Github desktop
2.	QCodes: Open command prompt, Navigate to the Qcodes folder on your computer (SilQ/Qcodes),
    run ‘python setup.py develop’ and install missing packages (usually coverage and pyqtgraph)
3.	SilQ: Open Command prompt (type cmd into start menu and right click to open as administrator)
    1.	Navigate to SilQ folder on your computer (not SilQ/silq)
    3.	Run `python setup.py develop`
    4.	It will likely say that packages are missing
        Install these missing packages like instructed (`pip install {packages}`, where `{packages}` are the packages you want to install separated by spaces)

### Install Experiments repository
We store our group experiments in the GitHub repository Experiments, which includes our configurations and notebooks, but not our data.
1. Clone the repository [Experiments](https://github.com/nulinspiratie/experiments) into your hard drive.
2. Point SilQ to you experiments repository by typing:
```
import silq
silq.set_experiments_folder(r'{experiments_folder}')
```
where you should replace `{experiments_folder}` with the absolute path to your Experiments folder.
Note that you should write the path as a raw string: `r'{experiments_folder}'`.

### Optional: Install Jupyter Notebook extensions
A number of extensions have been written for Jupyter Notebooks, some of which are vey useful.
The extensions can be installed by running the following command in command prompt:
```
conda install -c conda-forge jupyter_contrib_nbextensions
```
You can see a list of extensions and enable/disable them through the [nbextension configurator](https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator). This can be accessed by opening a jupyter notebook instance, and clicking the `NBextensions` tab.

## Using SilQ
The measurement software SilQ and QCodes can be imported as a package, just like any other python packages.
To use the measurement software in your script, use the following line
```
import qcodes as qc
import silq
```


## Using GitHub
1.	Commit your changes to the master branch including a meaningful heading and commentary
2.	Sync with the master branch to get other people’s changes

## Choosing an IDE
### PyCharm
1.	Download, install, and open PyCharm Community Edition (or use the professional version which is free for students)
2.	Open SilQ project
3.	Go to file -> default settings -> project interpreter
4.	Set interpreter to the newly installed Python Anaconda 3.5
5.	Restart PyCharm

When using PyCharm, code must be executed in scripts. To run QCoDeS, write your commands after `if __name__ == "__main__":`

### Jupyter Notebook
should be already installed alongside Anaconda Python


## Creating initialization code
An experimental setup (or personal computer) often requires code that should be initialized at the start of your measurement session.
This includes loading often-used packages, initializing instruments, setting up the Layout, etc.
In SilQ this can be done by creating a list of initialization files that can easily be run.
If the initialization belongs to a measurement, the correct folder is `experiments\{experiment_name}\init`, while for personal initialization code the correct folder is `personal\{username}\{whatever_you_prefer}`.
It is recommended that you create separate files for distinct types of initialization code (e.g. package imports, instruments).
The files will be run in alphabetical order, so it is useful to add a numerical prefix to indicate the loading order.

Furthermore, you should add a label to `silq\configurations.py`, with information on the location of the initialization folder.
You can also add additional `modes`, which can for instance ignore some of the files (useful when you want to operate in `analysis` mode, etc.).
Once the initialization code has been written and a label has been assigned, you can run the initialization code using:

```
import silq
silq.initialize({label}, mode={mode})
```
where mode is optional.


## Adding new interfaces
Adding a new instrument interface involves changing a handful of existing files and creating some necessary new files.
This section gives a quick overview on how to add a new (empty) interface.
 
### Changing existing files
#### `silq/instrument_interfaces/__init__.py`

Add a new line for importing the new interface class 

```
from .<company>.<model>_interface import <model>Interface
```

Add a new entry in the `instrument_interfaces` dictionary. This connects the new interface to the corresponding Qcodes driver of the instrument. 

```
'<instrument_class_name>': <model>Interface,
```

### Creating new files
The following files need to be created:

#### `silq/instrument_interfaces/<company>/<model>_interface.py`

This is the main file containing most of the interface functionality. This file should contain a class specific for `<model>` called `<model>Interface` which inherits from `InstrumentInterface` and should implement all abstract methods.

#### `silq/instrument_interfaces/<company>/__init__.py`

This file is used for importing (possibly multiple) interface(s) of a specific company. Add a new line to do so.

```
from .<model>_interface import <model>Interface
```
