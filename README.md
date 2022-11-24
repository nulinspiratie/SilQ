# SilQ

SilQ is a measurement software designed to control and measure quantum systems. It utilizes the [QCoDeS](https://github.com/nulinspiratie/Qcodes) data acquisition framework.

Full documentation can be found at https://nulinspiratie.github.io/SilQ/


## Installation

### Requirements

The key requirements for `SilQ` outside the default anaconda packages are the following:

```yml
name: silq
channels:
  - defaults
dependencies:
  - python=3.9
  - nodejs
  - sidecar
  - pyqtgraph
  - peakutils
  - lmfit
  - ipympl
  - slacker
  - nidaqmx-python
  - pyvisa=1.10.1
prefix: C:\Users\<USERNAME>\anaconda3
```

**N.B.** pay close attention to the pyvisa verion (1.10.1). SilQ and the qcodes fork that silq relies on will not work with newer pyvisa versions such as `1.12.0`.

It is recommend to install the required packages using `conda`, instead of `pip` via `conda install -c conda-forge <PACKAGE NAME>` as administrator.

Users will also need the following:

```yml
dependencies:
  - keysight-hvi
  - keysightSD1
```

The methods of installing the keysight packages are not elegant and are outlined below:

### Quick Installation Guide:

#### Windows

1. Clone the [SilQ](https://nulinspiratie.github.io/SilQ/) and relevant [Qcodes fork](https://github.com/nulinspiratie/Qcodes) repo

2. Open anaconda prompt window - **run as administrator**

3. cd into the SilQ repo directory and create an environment using the `spec-file.txt`. This will install all the required packages readily available via the `conda` package manager. 

```cmd.exe
conda create --name myenv --file spec-file.txt
```

4. Activate the newly created environment

```cmd.exe
conda activate myenv
```

5. Install Qcodes. `cd` into the Qcodes repo directory running the following in your anaconda prompt admin window

```cmd.exe
pip install -e .
```

6. Install SilQ. `cd` into the SilQ repo directory running the following in your anaconda prompt admin window

```cmd.exe
pip install -e .
```

7. Download [KeysightSD1 2.X software](https://www.keysight.com/us/en/lib/software-detail/instrument-firmware-software/sd1-2x-software-2784055.html), follow the wizard, use default settings.

8. Install `keysight-hvi`

```cmd.exe
cd C:\Program Files\Keysight\PathWave Test Sync Executive 2021\api\Python\Python38

pip install .
```

9. Install `keysightSD1`, by copying the the `keysightSD1.py` file from `C:\Program Files (x86)\Keysight\SD1\Libraries\Python` to `C:\Users\Scarlett\anaconda3\envs\<MY ENV>\Lib\site-packages`.


#### Linux

Unfortunately, there is not a prepared conda environment `spec-file.txt` file for Linux-based operating systems but one can create a new conda environment and install the [required packages](#requirements) in the following manner

```bash
conda create --name myenv

conda activate myenv

conda install -c forge nodejs sidecar pyqtgraph peakutils lmfit ipympl slacker nidaqmx-python pyvisa=1.10.1

```

Then continue the installation of `Qcodes`, `SilQ`, `keysight-hvi` and `keysightSD1` by carrying out the relevant steps outlined above and replacing the directories for the relevant modules with the Linux-based ones.
