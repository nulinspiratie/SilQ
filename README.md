#SilQ

SilQ is a measurement software designed to control spin-based systems. It utilizes the QCoDeS data acquisition framework.


## Installation

### Install Python
1.	Download [Anaconda 64bit python 3.5](https://www.continuum.io/downloads)
2.	Install Anaconda (make sure to tick the options add to system path and register as python 3.5

### Install Github
1.	Download and install [Github Desktop]9https://desktop.github.com/) and create a [Github account](https://github.com/)
2.	Give Serwan your account details to add you to the private git repository

### Install SilQ+QCodes
1.	Clone the repository [SilQ](https://github.com/nulinspiratie/SilQ) (includes QCodes) into your hard drive
    (e.g. User/Documents/PythonSoftware) using the installed Github desktop
2.	QCodes: Open command prompt, Navigate to the Qcodes folder on your computer (SilQ/Qcodes),
    run ‘python setup.py develop’ and install missing packages (usually coverage and pyqtgraph)
3.	SilQ: Open Command prompt (type cmd into start menu and right click to open as administrator)
    1.	Navigate to SilQ folder on your computer (not SilQ/silq)
    2.  Run “pip install pythonnet”
    3.	Run “python setup.py develop”
    4.	It will likely say that packages are missing (usually lmfit and peakutils).
        Install these missing packages like instructed (conda (or pip) install lmfit and peakutils)

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
