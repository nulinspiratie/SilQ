from blinker import signal
import json
import warnings

import qcodes as qc
from qcodes.config.config import DotDict, update


class SubConfig(DotDict):
    def __init__(self, name, filepath=None):
        self.name = name
        self.filepath = filepath


class DictConfig(DotDict):
    def __init__(self, name, filepath, config=None, item_class=None):
        self.__dict__['name'] = name
        self.__dict__['filepath'] = filepath

        if config is None:
            try:
                with open(filepath, "r") as fp:
                    config = json.load(fp)
            except FileNotFoundError:
                warnings.warn('Could not find config file for {}'.format(name))
        if item_class is None:
            update(self, config)
        else:
            for key, val in config.items():
                self[key] = item_class(name=key, **val)

        qc.config.user.update({name: self})

class ListConfig(list):
    def __init__(self, name, filepath, config=None):
        self.name = name
        self.filepath = filepath

        if config is None:
            try:
                with open(filepath, "r") as fp:
                    self += json.load(fp)
            except FileNotFoundError:
                warnings.warn('Could not find config file for {}'.format(name))

        qc.config.user.update({name: self})


class PulseConfig(DotDict):
    def __init__(self, name, **kwargs):
        self.__dict__['name'] = name
        self.__dict__['signal'] = signal('pulse_config:' + self.name)
        self.update(**kwargs)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        if isinstance(val, str) and 'config:' in val:
            val = qc.config['user'].__getitem__(val[7:])
        return val

    def __setitem__(self, key, val):
        current_val = super().__getitem__(key)
        if isinstance(current_val, str) and 'config:' in current_val:
            # Remove signal functionsignal

        super().__setitem__(key, val)

        # Get val after setting (can be different if val is depedent,
        # i.e. contains 'config:')
        get_val = self[key]
        self.signal.send(self, **{key: get_val})

    __setattr__ = __setitem__
