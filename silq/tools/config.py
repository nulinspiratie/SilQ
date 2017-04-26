from blinker import signal
import json
import warnings
from functools import partial

import qcodes as qc
from qcodes.config.config import DotDict, update


class SubConfig:
    def __init__(self, name, filepath=None, parent=None):
        # Set through __dict__ since setattr may be overridden
        self.__dict__['name'] = name
        self.__dict__['filepath'] = filepath
        self.__dict__['_connected_attrs'] = {}
        self.__dict__['parent'] = parent

        if self.parent is not None:
            # return f'{self.parent.config_path}.{self.name}'
            self.__dict__['config_path'] = '{}.{}'.format(
                self.parent.config_path, self.name)
        else:
            self.__dict__['config_path'] = 'config:{}'.format(self.name)
            # return f'config:{self.name}'


class DictConfig(SubConfig, DotDict):
    def __init__(self, name, filepath=None, parent=None, config=None):
        DotDict.__init__(self)
        SubConfig.__init__(self, name=name, filepath=filepath, parent=parent)

        if config is None and filepath is not None:
            try:
                with open(filepath, "r") as fp:
                    config = json.load(fp)
            except FileNotFoundError:
                warnings.warn('Could not find config file for {}'.format(name))

        update(self, config)

        qc.config.user.update({name: self})

    def __getitem__(self, key):
        val = DotDict.__getitem__(self, key)
        if isinstance(val, str) and 'config:' in val:
            val = qc.config['user'].__getitem__(val[7:])
        return val

    def __setitem__(self, key, val):
        try:
            current_val = DotDict.__getitem__(self, key)
            if isinstance(current_val, str) and 'config:' in current_val:
                config_path, attr = current_val.rsplit('.', 1)
                signal_function = self._connected_attrs.pop(key)
                signal(config_path).disconnect(signal_function)
        except KeyError:
            pass

        # Update item in dict (modified version of DotDict)
        if type(key)==str and '.' in key:
            myKey, restOfKey = key.split('.', 1)
            self.setdefault(myKey, DictConfig(name=myKey, config=val,
                                              parent=self))
            # target[restOfKey] = value
        else:
            if isinstance(val, dict) and not isinstance(val, SubConfig):
                val = DictConfig(name=key, config=val, parent=self)
            dict.__setitem__(self, key, val)

        if isinstance(val, str) and 'config:' in val:
            # Attach update function if file is in config
            config_path, attr = val.rsplit('.', 1)
            signal_function = partial(self._handle_config_signal, key, attr)
            signal(config_path).connect(signal_function)
            self._connected_attrs[key] = signal_function

        # Get val after setting, as it can be different if val is dependent,
        # (i.e. contains 'config:'). Using if because if val is dependent,
        # and the 'listened' property does not exist yet, hasattr=False.
        if hasattr(self, key):
            get_val = self[key]
            signal(self.config_path).send(self, **{key: get_val})

    def _handle_config_signal(self, dependent_attr,  listen_attr, _, **kwargs):
        """
        Sends signal when 'listened' property of dependent property is updated.
        Args:
            dependent_attr: name of dependent attribute
            listen_attr: name of attribute that is listened.
            _: sender object (not important)
            **kwargs: {'listened' attr: val}
                The dependent attribute mirrors the value of the 'listened' 
                attribute

        Returns:

        """
        sender_key, sender_val = kwargs.popitem()
        if sender_key == listen_attr:
            signal(self.config_path).send(self, **{dependent_attr: sender_val})

    __setattr__ = __setitem__



class ListConfig(SubConfig, list):
    def __init__(self, name, filepath=None, parent=None, config=None):
        list.__init__()
        SubConfig.__init__(self, name=name, filepath=filepath, parent=parent)

        if config is None and filepath is not None:
            try:
                with open(filepath, "r") as fp:
                    self += json.load(fp)
            except FileNotFoundError:
                warnings.warn('Could not find config file for {}'.format(name))

        qc.config.user.update({name: self})


# class PulseConfig(DotDict):
#     def __init__(self, name, **kwargs):
#         self.__dict__['name'] = name
#         self.__dict__['signal'] = signal('pulse_config:' + self.name)
#         self.update(**kwargs)
#
#
