from typing import List
import sys
import os
import warnings
import logging
import json
import h5py
import pickle
from pathlib import Path
from datetime import timedelta
from datetime import datetime

from .tools.config import DictConfig, ListConfig
from .tools.parameter_tools import SweepDependentValues
from .tools.general_tools import SettingsClass

import qcodes as qc

# Initially set the environment to None. Changing this will affect all
# environment config listeners (see `DictConfig`)
environment = None

# Initially set the experiments_folder to None.
experiments_folder = None

logger = logging.getLogger(__name__)

# Dictionary of SilQ subconfigs
config = DictConfig(name='config', save_as_dir=True, config={'properties': {}})
# Set qcodes.config.user.silq_config to the silq config
qc.config.user.update({'silq_config': config})

silq_env_var = 'SILQ_EXP_FOLDER'

if 'ipykernel' in sys.modules:
    # Load iPython magic (configured via qc.config.core.register_magic)
    from qcodes.utils.magic import register_magic_class

    register_magic = qc.config.core.get('register_magic', False)
    if register_magic is not False:
        from silq.tools.notebook_tools import SilQMagics

        register_magic_class(cls=SilQMagics,
                             magic_commands=register_magic)


def get_silq_folder() -> str:
    """Get root folder of silq source code."""
    return os.path.split(__file__)[0]


def get_SilQ_folder() -> str:
    """Get root folder of SilQ (containing source code)"""
    silq_folder = get_silq_folder()
    return os.path.join(silq_folder, r"../")


def set_experiments_folder(folder: str):
    """Sets experiments folder, used by `silq.initialize`.

    Setting the experiments should only be done once, as it creates the file
    ``experiments_folder.txt`` in the SilQ folder, which contains the
    experiments folder.

    The Experiments folder should contain one folder for each experiment.
    An Experiment folder contains startup scripts, configuration, and usually
    scripts and notebooks.
    An experiment folder contains the following folders:

    * **init**: Initialization .py scripts, executed in ascending order.
      Each file is should have form ``{idx}_{name}``, where ``idx`` is a
      zero-based that defines execution order, and ``name`` is the filename.
    * **config**: SilQ config folder, wherein each item is either a .JSON file
      or a folder containing .jSON files.

    Args:
        folder: Experiments folder
    """
    experiments_filepath = os.path.join(get_SilQ_folder(),
                                        'experiments_folder.txt')
    with open(experiments_filepath, 'w') as f:
        f.write(folder)


def get_experiments_folder():
    """Gets experiments folder

    Returns:
        experiments folder

    Raises:
        FileNotFoundError: No experiments_folder configured.
    """
    try:
        import silq
        experiments_folder = silq.experiments_folder
    except:
        experiments_folder = None

    if silq.experiments_folder is not None:
        return silq.experiments_folder
    else:
        experiments_folder = os.getenv(silq_env_var, None)
        if experiments_folder is not None:
            return experiments_folder
        else:
            logger.debug("Could not find experiments folder in system "
                         "environment variable 'SILQ_EXP_FOLDER'.")
            experiment_filepath = os.path.join(get_SilQ_folder(),
                                               'experiments_folder.txt')
            if os.path.exists(experiment_filepath):
                with open(experiment_filepath, 'r')as f:
                    experiments_folder = f.readline()
                return experiments_folder
            else:
                raise FileNotFoundError('No file "Silq/experiments_folder.txt" '
                                        'exists. Can be set via '
                                        'silq.set_experiments_folder()')


def get_experiment_configuration(name: str, experiments_folder: Path = None) -> (Path, dict):
    """Retrieves experiment folder and configuration from experiments folder.
    This contains all configurations that can be used by `silq.initialize`.

    Args:
        name: Case-insensitive experiment name. Can be either a relative
              folder path, or defined in configurations.json
        experiments_folder: Optional path for experiments_folder

    Returns: experiment folder and configuration if found.
             If not found, returns (False, {})

    Configuration file `configurations.json` can be either in experiment_folder
    or global experiments_folder/configurations.json

    Returns:
        dict of configurations
    """
    if experiments_folder is None:
        # Determine base folder by looking at the silq package
        experiments_folder = get_experiments_folder()
    if isinstance(experiments_folder, str):  # Convert to Path
        experiments_folder = Path(experiments_folder)

    experiment_folder = (experiments_folder / name).resolve()
    if experiment_folder.exists():
        # Check if there is a configuration file
        configuration_filepath = experiment_folder / 'configurations.json'
        if configuration_filepath.exists():
            configuration = json.load(configuration_filepath.open('r'))
        else:
            configuration = {}

    if not experiment_folder.exists() or not configuration:
        # Check if the configuration is specified in global configurations.json
        configurations_filepath = experiments_folder / 'configurations.json'
        try:
            configurations = json.load(configurations_filepath.open('r'))
            configuration = next(val for key, val in configurations.items()
                                 if key.lower() == name.lower())
            experiment_folder = experiments_folder / configuration['folder']
        except StopIteration:
            configuration = {}

    if not experiment_folder.exists():
        experiment_folder = False

    return experiment_folder, configuration


def execute_file(filepath: (str, Path), globals=None, locals=None):
    if globals is None and locals is None:
        # Register globals and locals of the above frame (for code execution)
        globals = sys._getframe(1).f_globals
        locals = sys._getframe(1).f_locals

    if isinstance(filepath, str):
        filepath = Path(filepath)

    assert filepath.suffix == '.py'

    execution_code = filepath.read_text() + "\n"
    try:
        exec(execution_code, globals, locals)
    except:
        raise RuntimeError(f'SilQ initialization error in {filepath} with code \n'
                           f'{execution_code}')


def run_scripts(name, mode: str = None, silent=False, globals=None, locals=None):
    if globals is None and locals is None:
        # Register globals and locals of the above frame (for code execution)
        globals = sys._getframe(1).f_globals
        locals = sys._getframe(1).f_locals

    experiment_folder, _ = get_experiment_configuration(name)

    script_folders = [experiment_folder / 'scripts']

    # Add script folders in current directory and parent directories thereof
    for k in range(4):
        relative_directory = Path('../' * k + '.')
        if relative_directory.absolute() == experiment_folder.absolute():
            # Reached experiment folder
            break

        script_folder = relative_directory / 'scripts'
        if script_folder.is_dir():
            script_folders.append(script_folder)

        # relative_directory = os.path.split(relative_directory)[0]
        # print(f'relative_directory: {relative_directory}')
    # print(script_folders)

    for script_folder in script_folders:
        if not script_folder.exists():
            continue

        for script_folder_element in script_folder.iterdir():
            # Only execute scripts in subfolders if folder name matches mode
            if script_folder_element.is_dir():
                if script_folder_element.stem != mode:
                    continue

                # Run each file in the subfolder
                for script_folder_subelement in script_folder_element.iterdir():
                    if not silent:
                        print(f'Running script {script_folder_element.stem}/{script_folder_subelement.stem}')
                    execute_file(script_folder_subelement, globals=globals, locals=locals)
            else:  # Execute file
                if not silent:
                    print(f'Running script {script_folder_element.stem}')

                execute_file(script_folder_element, globals=globals, locals=locals)


def initialize(name: str,
               mode: str = None,
               select: List[str] = [],
               ignore: List[str] = [],
               scripts=True,
               silent=False):
    """Runs experiment initialization .py scripts.

    The initialization scripts should be in the ``init`` folder in the
    experiment folder.
    Possible configurations are taken from the dictionary _configurations in
    the file configurations.py.

    Args:
        name: name of the configuration, used to find the folder
            from which to execute all init files.
        mode: mode that determines which subset of files should
            be executed. Possible modes can be specified in _configurations.
        select: Files to select, all others will be ignored.
        ignore: Files to ignore, all others will be selected.

    Notes:
        Scripts are run in the global namespace
    """
    # Register globals and locals of the above frame (for code execution)
    globals = sys._getframe(1).f_globals
    locals = sys._getframe(1).f_locals

    experiment_folder, configuration = get_experiment_configuration(name)
    assert experiment_folder, f"Could not find experiment '{name}', make sure " \
                              f"its a folder in the experiments_folder"

    if mode is not None:
        select += configuration['modes'][mode].get('select', [])
        ignore += configuration['modes'][mode].get('ignore', [])

    # TODO check if original config['folder'] had any weird double experiments_folder
    config.__dict__['folder'] = experiment_folder.absolute()
    if (experiment_folder / 'config').is_dir():
        config.load()

    # Run initialization files in ./init
    init_folder = experiment_folder / 'init'
    init_files = [f for f in init_folder.iterdir() if f.is_file()]

    for init_file in init_files:
        # Remove prefix
        filename_no_prefix = init_file.stem.split('_', 1)[1]
        if select and filename_no_prefix not in select:
            continue
        elif ignore and filename_no_prefix in ignore:
            continue
        else:
            if not silent:
                print(f'Initializing {filename_no_prefix}')
            execute_file(init_file, globals=globals, locals=locals)

    if scripts:
        run_scripts(name=name, mode=mode, globals=globals, locals=locals)

    if 'data_folder' in config.properties:
        qc.set_data_root_folder(config.properties.data_folder)

        location_provider = qc.data.data_set.DataSet.location_provider
        if os.path.split(config.properties.data_folder)[-1] == 'data' and \
                (location_provider.fmt == 'data/{date}/#{counter}_{name}_{time}'):
            logger.debug('Removing duplicate "data" from location provider')
            location_provider.fmt = '{date}/#{counter}_{name}_{time}'

    if not silent:
        print("Initialization complete")


### Override QCoDeS functions

# Add saving of config to qcodes DataSet
def _save_config(self, location=None):
    try:
        if location is None:
            location = self.location
        if not location and hasattr(self, '_location'):
            # Location is False, dataset created in qc.Measure, ignore
            return

        if not os.path.isabs(location):
            location = os.path.join(qc.DataSet.default_io.base_location, location)
        config.save(location)
    except Exception as e:
        logger.error(f'Datasaving error: {e.args}')
qc.DataSet.save_config = _save_config


def _load_traces(self, name: str = None, mode: str = 'r'):
    """Load traces HDF5 file from a dataset

    Args:
        name: Optional name to specify traces file. Should be used if more than
            one parameter is used in the measurement that saves traces.
        mode: Open mode (default is 'r' for read-only)
        """
    data_path = self.io.to_path(self.location)
    trace_path = os.path.join(data_path, 'traces')
    trace_filenames = os.listdir(trace_path)
    assert trace_filenames, f"No trace files found in {traces_path}"

    if name is None and len(trace_filenames) == 1:
        trace_filename = trace_filenames[0]
    else:
        assert name is not None, f"No unique trace file found: {trace_filenames}. " \
                                 "Trace filename must be provided"
        filtered_trace_filenames = [filename for filename in trace_filenames
                                    if name in filename]
        assert len(filtered_trace_filenames) == 1, \
            f"No unique trace file found: {trace_filenames}."
        trace_filename = filtered_trace_filenames[0]

    trace_filepath = os.path.join(trace_path, trace_filename)
    trace_file = h5py.File(trace_filepath, mode)
    return trace_file
qc.DataSet.load_traces = _load_traces



def _get_pulse_sequence(self, idx=0, pulse_name=None):
    """Load pulse sequence after measurement started

    Args:
        idx: index of pulse sequence after measurement started

    Returns:
        Pulse sequence
    """

    def get_next_pulse_sequence(date_time, max_date_delta=1):
        current_date = date_time
        while current_date <= date_time + timedelta(days=max_date_delta):
            date_str = date_time.strftime("%Y-%m-%d")
            pulse_sequence_path = os.path.join(self.default_io.base_location,
                                               r'pulse_sequences\data', date_str)
            pulse_sequence_files = os.listdir(pulse_sequence_path)
            # Sort by their idx (i.e. #095 at start of filename)
            pulse_sequence_files = sorted(pulse_sequence_files, key=lambda file: int(file.split('_')[0][1:]))
            for k, pulse_sequence_file in enumerate(pulse_sequence_files):
                pulse_sequence_date_time = datetime.strptime(f'{date_str}:{pulse_sequence_file[-15:-7]}', '%Y-%m-%d:%H-%M-%S')
                if pulse_sequence_date_time > current_date:
                    pulse_sequence_filepath = os.path.join(pulse_sequence_path,
                                                           pulse_sequence_file)
                    with open(pulse_sequence_filepath, 'rb') as f:
                        pulse_sequence = pickle.load(f)

                    # Pulse sequence duration needs to be reset
                    try:
                        duration = pulse_sequence['duration']._latest['raw_value']
                        pulse_sequence['duration']._latest['value'] = duration
                    except:
                        pass

                    current_date = pulse_sequence_date_time
                    yield pulse_sequence, f"{date_str}/{pulse_sequence_file}"
            else:
                # Goto next date
                current_date = datetime.combine(current_date.date() + timedelta(days=1),
                                                datetime.min.time())

    date, measurement_name = self.location.split('\\')
    measurement_date_time = datetime.strptime(f'{date}:{measurement_name[-8:]}', '%Y-%m-%d:%H-%M-%S')

    try:
        pulse_sequence_iterator = (pulse_sequence for pulse_sequence in
                                   get_next_pulse_sequence(measurement_date_time)
                                   if (not pulse_name or pulse_name in pulse_sequence[0]))
        for k in range(idx+1):
            pulse_sequence, pulse_sequence_filename = next(pulse_sequence_iterator)
    except StopIteration:
        raise StopIteration('No pulse sequences found')

    print(f'Pulse sequence file: {pulse_sequence_filename}')
    return pulse_sequence

qc.DataSet.get_pulse_sequence = _get_pulse_sequence


# parameter.sweep
def _sweep(self, start=None, stop=None, step=None, num=None,
          step_percentage=None, window=None, fix=True):
    if step_percentage is None and window is None:
        if start is None or stop is None:
            raise RuntimeError('Must provide start and stop')
        from qcodes.instrument.sweep_values import SweepFixedValues
        return SweepFixedValues(self, start=start, stop=stop,
                                step=step, num=num)
    else:
        return SweepDependentValues(parameter=self, step=step,
                                    step_percentage=step_percentage, num=num,
                                    window=window, fix=fix)
qc.Parameter.sweep = _sweep


# Override ActiveLoop._run_wrapper to stop the layout and clear settings of
# any accquisition parameters in the loop after the loop has completed.
_qc_run_wrapper = qc.loops.ActiveLoop._run_wrapper
def _run_wrapper(self, set_active=True, stop=True, *args, **kwargs):

    def clear_all_acquisition_parameter_settings(loop):
        from silq.parameters import AcquisitionParameter

        for action in loop:
            if isinstance(action, qc.loops.ActiveLoop):
                clear_all_acquisition_parameter_settings(action)
            elif isinstance(action, SettingsClass):
                logger.info(f'End-of-loop: clearing settings for {action}')
                action.clear_settings()

    try:
        _qc_run_wrapper(self, set_active=set_active, *args, **kwargs)
    finally:
        try:
            layout = qc.Instrument.find_instrument('layout')
            if stop:
                layout.stop()
                logger.info('Stopped layout at end of loop')
                if set_active:
                    layout.close_trace_files()
        except KeyError:
            logger.warning(f'No layout found to stop')

        # Clear all settings for any acquisition parameters in the loop
        clear_all_acquisition_parameter_settings(self)
qc.loops.ActiveLoop._run_wrapper = _run_wrapper