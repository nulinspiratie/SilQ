from typing import List
import sys
import os
import warnings
import logging
import json

from .tools.config import DictConfig, ListConfig
from .tools.parameter_tools import SweepDependentValues
from .tools.general_tools import SettingsClass

import qcodes as qc

# Initially set the environment to None. Changing this will affect all
# environment config listeners (see `DictConfig`)
environment = None


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


def get_configurations() -> dict:
    """Retrieves configurations folder from experiments folder.

    This contains all configurations that can be used by `silq.initialize`.
    Filepath should be {experiments_folder/configurations.json}

    Returns:
        dict of configurations
    """
    experiments_folder = get_experiments_folder()
    configurations_filepath = os.path.join(experiments_folder,
                                           'configurations.json')
    with open(configurations_filepath, 'r') as file:
        return json.load(file)


def initialize(name: str = None,
               mode: str = None,
               select: List[str] = [],
               ignore: List[str] = []):
    """Runs experiment initialization .py scripts.

    The initialization scripts should be in the ``init`` folder in the
    experiment folder.
    Possible configurations are taken from the dictionary _configurations in
    the file configurations.py.

    Args:
        name: name of the configuration, used to find the folder
            from which to execute all init files. If not given, the MAC address
            will be used to find the default configuration_name.
        mode: mode that determines which subset of files should
            be executed. Possible modes can be specified in _configurations.
        select: Files to select, all others will be ignored.
        ignore: Files to ignore, all others will be selected.

    Notes:
        Scripts are run in the global namespace
    """
    # Determine base folder by looking at the silq package

    globals = sys._getframe(1).f_globals
    locals = sys._getframe(1).f_locals

    experiments_folder = get_experiments_folder()
    configurations = get_configurations()

    if name is None:
        # Find init_name from mac address
        from uuid import getnode as get_mac
        mac = get_mac()
        for name, properties in configurations.items():
            if mac in properties.get('macs', []):
                name = name
                break

    try:
        configuration = next(val for key, val in get_configurations().items()
                             if key.lower() == name.lower())
    except StopIteration:
        raise NameError(f'Configuration {name} not found. Allowed '
                        f'configurations are {get_configurations().keys()}')

    if mode is not None:
        select += configuration['modes'][mode].get('select', [])
        ignore += configuration['modes'][mode].get('ignore', [])

    folder = os.path.join(experiments_folder, configuration['folder'])
    config.__dict__['folder'] = os.path.join(experiments_folder, folder)
    if os.path.exists(os.path.join(folder, 'config')):
        config.load()

    # Run initialization files in ./init
    init_folder = os.path.join(folder, 'init')
    init_filenames = os.listdir(init_folder)

    for filename in init_filenames:
        # Remove prefix
        filename_no_prefix = filename.split('_', 1)[1]
        # Remove .py extension
        filename_no_prefix = filename_no_prefix.rsplit('.', 1)[0]
        if select and filename_no_prefix not in select:
            continue
        elif ignore and filename_no_prefix in ignore:
            continue
        else:
            print(f'Initializing {filename_no_prefix}')
            filepath = os.path.join(init_folder, filename)
            with open(filepath, "r") as fh:
                exec_line = fh.read()
                try:
                    exec(exec_line+"\n", globals, locals)
                except:
                    raise RuntimeError(f'SilQ initialization error in '
                                       f'{filepath}')

    print("Initialization complete")

    if 'environment' in config.properties:
        logger.info(f'Initialization: environment set to '
                    f'{config.properties.environment}')

    if 'data_folder' in config.properties:
        logger.debug(f'using config data folder: '
                     f'{config.properties.data_folder}')
        qc.data.data_set.DataSet.default_io.base_location = \
            config.properties.data_folder

        location_provider = qc.data.data_set.DataSet.location_provider
        if os.path.split(config.properties.data_folder)[-1] == 'data' and \
                (location_provider.fmt ==
                     'data/{date}/#{counter}_{name}_{time}'):
            logger.debug('Removing duplicate "data" from location provider')
            location_provider.fmt = '{date}/#{counter}_{name}_{time}'


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


# parameter.sweep
def _sweep(self, start=None, stop=None, step=None, num=None,
          step_percentage=None, window=None):
    if step_percentage is None and window is None:
        if start is None or stop is None:
            raise RuntimeError('Must provide start and stop')
        from qcodes.instrument.sweep_values import SweepFixedValues
        return SweepFixedValues(self, start=start, stop=stop,
                                step=step, num=num)
    else:
        return SweepDependentValues(parameter=self, step=step,
                                    step_percentage=step_percentage, num=num,
                                    window=window)
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
        except KeyError:
            logger.warning(f'No layout found to stop')

        # Clear all settings for any acquisition parameters in the loop
        clear_all_acquisition_parameter_settings(self)
qc.loops.ActiveLoop._run_wrapper = _run_wrapper