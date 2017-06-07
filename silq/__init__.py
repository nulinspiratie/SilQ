import sys
import os
import warnings
from .configurations import _configurations
import json
from .tools.config import DictConfig, ListConfig

# Dictionary of SilQ subconfigs
config = DictConfig(name='config', save_as_dir=True, config={'properties': {}})

def get_silq_folder():
    import silq
    return os.path.split(silq.__file__)[0]

def get_SilQ_folder():
    silq_folder = get_silq_folder()
    return os.path.join(silq_folder, r"../")

def initialize(name=None, mode=None, select=None, ignore=None):
    """
    Initializes the global namespace by executing a list of files.
    Possible configurations are taken from the dictionary _configurations in
    the file configurations.py.
    If name is not given, the computer's MAC address is used to
    determine the default configuration_name.

    Args:
        name: name of the configuration, used to find the folder
            from which to execute all init files. If not given, the MAC address
            will be used to find the default configuration_name.
        mode: mode that determines which subset of files should
            be executed. Possible modes can be specified in _configurations.
        select: Files to select, all others will be ignored.
        ignore: Files to ignore, all others will be selected.

    Returns:

    """
    # Determine base folder by looking at the silq package

    globals = sys._getframe(1).f_globals
    locals = sys._getframe(1).f_locals

    if name is None:
        # Find init_name from mac address
        from uuid import getnode as get_mac
        mac = get_mac()
        for name, properties in _configurations.items():
            if mac in properties.get('macs', []):
                name = name
                break

    if mode is not None:
        select = _configurations[name]['modes'][mode].get('select', None)
        ignore = _configurations[name]['modes'][mode].get('ignore', None)

    folder = os.path.join(get_SilQ_folder(), _configurations[name]['folder'])
    config.__dict__['folder'] = folder
    if os.path.exists(os.path.join(folder, 'config')):
        config.load()

    # Run initialization files in ./init
    init_folder = os.path.join(folder, 'init')
    init_filenames = os.listdir(init_folder)

    for filename in init_filenames:
        # Remove prefix
        name = filename.split('_', 1)[1]
        # Remove .py extension
        name = name.rsplit('.', 1)[0]
        if select is not None and name not in select:
            continue
        elif ignore is not None and name in ignore:
            continue
        else:
            print('Initializing {}'.format(name))
            filepath = os.path.join(init_folder, filename)
            with open(filepath, "r") as fh:
                exec_line = fh.read()
                try:
                    exec(exec_line+"\n", globals, locals)
                except:
                    raise RuntimeError(f'SilQ initialization error in '
                                       f'{filepath}')

    print("Initialization complete")

    if not 'default_environment' in config.properties:
        warnings.warn("'default_environment' should be specified "
                      "in silq.config.properties")