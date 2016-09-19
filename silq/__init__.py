import sys
import os
'''
This dictionary contains a list of initialization files.
An initialization file can be loaded using silq.initialize(init_name),
after which everything in the file will be loaded into the global namespace.
Each setting has a name, which can be given to silq.initialize.
The filename corresponds to the init file to be run, from bsae folder SilQ.
Additionally, a list of mac addresses can be added to each setting.
Adding a mac address to a setting will result in that settings file to be
instantiated upon calling intialize without specifying init_name
'''
_init_files = {
    'Serwan': {'filename': r"personal\Serwan\init.py",
               'macs': [113780617482]},
    'EWJN': {'filename': r"experiments\EWJN\init.py",
             'macs': []}
}

def initialize(init_name=None, globals=None, locals=None):
    # Determine base folder by looking at the silq package
    import silq
    silq_folder = os.path.split(silq.__file__)[0]
    SilQ_folder = os.path.join(silq_folder, r"../")

    if init_name is None:
        # Find init_name from mac address
        from uuid import getnode as get_mac
        mac = get_mac()
        for name, properties in _init_files.items():
            if mac in properties.get('macs', []):
                init_name = name
                break

    filename = os.path.join(SilQ_folder, _init_files[init_name]['filename'])

    if globals is None:
        globals = sys._getframe(1).f_globals
    if locals is None:
        locals = sys._getframe(1).f_locals
    with open(filename, "r") as fh:
        exec(fh.read()+"\n", globals, locals)