import sys
import os
from .configurations import _configurations

def get_silq_folder():
    import silq
    return os.path.split(silq.__file__)[0]

def get_SilQ_folder():
    silq_folder = get_silq_folder()
    return os.path.join(silq_folder, r"../")


def initialize(name=None, mode=None, select=None, ignore=None,
               globals=None, locals=None):
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
        globals: The globals namespace, actual global namespace used by default.
        locals: The locals namespace, actual local namespace used by default.

    Returns:

    """
    # Determine base folder by looking at the silq package

    if globals is None:
        globals = sys._getframe(1).f_globals
    if locals is None:
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
        select = _configurations[
            name]['modes'][mode].get('select', None)
        ignore = _configurations[
            name]['modes'][mode].get('ignore', None)

    SilQ_folder = get_SilQ_folder()
    folder = os.path.join(SilQ_folder,
                          _configurations[name]['folder'])

    filenames = os.listdir(folder)

    for filename in filenames:
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
            filepath = os.path.join(folder, filename)
            with open(filepath, "r") as fh:
                exec(fh.read()+"\n", globals, locals)
    print("Initialization complete")