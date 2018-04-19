import os
import collections
from blinker import signal
import json
from functools import partial
import copy

import qcodes as qc
from qcodes.config.config import DotDict

__all__ = ['SubConfig', 'DictConfig', 'ListConfig', 'update_dict']

class SubConfig:
    def __init__(self, name, folder=None, parent=None, save_as_dir=None):


        # Set through __dict__ since setattr may be overridden
        self.name = name
        self.folder = folder
        self._connected_attrs = {}
        self.parent = parent
        self.save_as_dir = save_as_dir

        qc.config.user.update({name: self})

    @property
    def config_path(self):
        if self.parent is None:
            return f'{self.name}:'
        else:
            parent_path = self.parent.config_path
            if parent_path[-1] != ':':
                parent_path += '.'
            return parent_path + self.name

    def load(self, folder=None, update=None):
        """
        Load config from folder.
        The folder must either contain a {self.name}.json file,
        or alternatively a folder containing config files/folders.
        In the latter case, a dict is created, and all the files/folders in
        the folder will be elements of the dict.

        If `self.save_as_dir is None`, it will be updated to either True or
        False depending if there is a subfolder or file to load from the folder,
        respectively.

        Note that load() returns a dict/list, which should then be added to
        the Subconfig object depending on the subclass. This should be
        implemented in the load() method of subclasses.

        Args:
            folder: folder to look for. If not provided, uses self.folder.

        Returns:
            dict/list config, to be added by the load() method of the subclass
        """
        if folder is None:
            folder = self.folder

        filepath = os.path.join(folder, f'{self.name}.json')
        folderpath = os.path.join(folder, self.name)

        if os.path.exists(filepath):
            # Load config from file
            # Update self.save_as_dir to False unless explicitly set to True
            if self.save_as_dir is None:
                self.save_as_dir = False

            # Load config from file
            with open(filepath, "r") as fp:
                config = json.load(fp)

        elif os.path.isdir(folderpath):
            # Config is a folder, and so each item in the folder is added to
            # a dict.
            config = {}

            # Update self.save_as_dir to False unless explicitly set to True
            if self.save_as_dir is None:
                self.save_as_dir = False

            for file in os.listdir(folderpath):
                filepath = os.path.join(folderpath, file)
                if '.json' in file:
                    with open(filepath, "r") as fp:
                        # Determine type of config
                        subconfig = json.load(fp)
                        if isinstance(subconfig, list):
                            config_class = ListConfig
                        elif isinstance(subconfig, dict):
                            config_class = DictConfig
                        else:
                            raise RuntimeError(f'Could not load config file '
                                               f'{filepath}')

                        subconfig_name = file.split('.')[0]
                        subconfig = config_class(name=subconfig_name,
                                                 folder=folderpath,
                                                 save_as_dir=False,
                                                 parent=self)
                elif os.path.isdir(filepath):
                    subconfig_name = file
                    subconfig = DictConfig(name=file,
                                           folder=folderpath,
                                           save_as_dir=True,
                                           parent=self)
                else:
                    raise RuntimeError(f'Could not load {filepath} to config')
                config[subconfig_name] = subconfig

        else:
            raise FileNotFoundError(
                f"No file nor folder found to load for {self.name}")

        return config

    def refresh(self, config=None):
        if config is None:
            config = self.load(update=False)

        if isinstance(config, dict) and isinstance(self, dict):
            for key, val in config.items():
                if key in self:
                    if isinstance(self[key], SubConfig):
                        self[key].refresh(config=config[key])
                    elif self[key] != val:
                        self[key] = val

                else:
                    self[key] = config[key]

            # Also remove any keys that are not in the new config
            for key in list(self):
                if key not in config:
                    self.pop(key)

        elif isinstance(config, list) and isinstance(self, list):
            if config != self:
                self.clear()
                self += config
        else:
            raise TypeError(f'{self.config_path} has different type as refreshed '
                            f'config {config}')

    def save(self, folder=None, save_as_dir=None, dependent_value=False):
        if folder == None:
            folder = self.folder
        if save_as_dir == None:
            save_as_dir = self.save_as_dir

        if not save_as_dir:
            filepath = os.path.join(folder, f'{self.name}.json')
            serialized_self = self.serialize(dependent_value=dependent_value)
            with open(filepath, 'w') as fp:
                json.dump(serialized_self, fp, indent=4)
        else:
            folderpath = os.path.join(folder, self.name)
            if not os.path.isdir(folderpath):
                os.mkdir((folderpath))
            for subconfig in self.values():
                subconfig.save(folder=folderpath)

    def serialize(self, dependent_value=False):
        raise NotImplementedError('Implement in subclass')


class DictConfig(SubConfig, DotDict):
    exclude_from_dict = ['name', 'folder', '_connected_attrs', 'parent',
                         'save_as_dir', 'config_path']
    def __init__(self, name, folder=None, parent=None, config=None,
                 save_as_dir=None):
        DotDict.__init__(self)
        SubConfig.__init__(self, name=name, folder=folder, parent=parent,
                           save_as_dir=save_as_dir)

        if config is not None:
            update_dict(self, config)
        elif folder is not None:
            self.load()

    def __contains__(self, key):
        if DotDict.__contains__(self, key):
            return True
        elif DotDict.__contains__(self, 'inherit'):
            if 'config:' in self.inherit:
                inherit_dict = qc.config['user'].__getitem__(self.inherit[7:])
            elif self.parent is not None:
                # Inherit is sibling of current dict item
                inherit_dict = self.parent[self.inherit]
            else:
                return False
            return key in inherit_dict
        else:
            return False

    def __getitem__(self, key):
        if DotDict.__contains__(self, key):
            val = DotDict.__getitem__(self, key)
            if key != 'inherit' and isinstance(val, str) and 'config:' in val:
                val = qc.config['user'].__getitem__(val[7:])
        elif 'inherit' in self:
            if 'config:' in self.inherit:
                inherit_dict = qc.config['user'].__getitem__(self.inherit[7:])
            elif self.parent is not None:
                # Inherit is sibling of current dict item
                inherit_dict = self.parent[self.inherit]
            else:
                raise KeyError
            val = inherit_dict[key]
        else:
            raise KeyError
        return val

    def __setitem__(self, key, val):

        # If previous value was dependent, remove connected function
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
            self.setdefault(myKey, DictConfig(name=myKey,
                                              config={restOfKey: val},
                                              parent=self))
        else:
            if isinstance(val, SubConfig):
                val.parent = self
            elif isinstance(val, dict):
                val = DictConfig(name=key, config=val, parent=self)
            elif isinstance(val, list):
                val = ListConfig(name=key, config=val, parent=self)
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
            # print(f'cfg: sending {(key, val)} to {self.config_path}')
            signal(self.config_path).send(self, **{key: get_val})

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self, dependent_value=True):
        if dependent_value:
            return {key: self[key] for key in self.keys()}.items()
        else:
            return {key: dict.__getitem__(self, key) for key in self.keys()}.items()

    def get(self, key, default=None):
        """
        Override dictionary get, because it otherwise does not call __getitem__
        Args:
            key: key to get
            default: default value if key not found. None by default

        Returns:
            value of key if in dictionary, else default value.
        """
        try:
            return self[key]
        except KeyError:
            return None

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

    def load(self, folder=None, update=True):
        if update:
            self.clear()
        config = super().load(folder=folder)
        if update:
            update_dict(self, config)
        return config

    def to_dict(self, dependent_value=True):
        d = {}
        for key, val in self.items(dependent_value=dependent_value):
            if isinstance(val, DictConfig):
                d[key] = val.to_dict(dependent_value=dependent_value)
            elif isinstance(val, ListConfig):
                d[key] = val.to_list(dependent_value=dependent_value)
            else:
                d[key] = val
        return d

    serialize = to_dict

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.to_dict())


class ListConfig(SubConfig, list):
    def __init__(self, name, folder=None, parent=None, config=None, **kwargs):
        list().__init__(self)
        SubConfig.__init__(self, name=name, folder=folder, parent=parent)

        if config is not None:
            self += config
        elif folder is not None:
            self.load()

    def load(self, folder=None, update=True):
        if update:
            self.clear()
        config = super().load(folder=folder)
        if update:
            self += config
        return config

    def to_list(self, dependent_value=True):
        l = []
        for val in self:
            if isinstance(val, DictConfig):
                l.append(val.to_dict(dependent_value=dependent_value))
            elif isinstance(val, ListConfig):
                l.append(val.to_list(dependent_value=dependent_value))
            else:
                l.append(val)
        return l

    serialize = to_list

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.to_list())

def update_dict(d, u):
    """
    Update dictionary recursively.
    this ensures that subdicts are also converted
    This is a modified version of the update function in qcodes config
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping) and k in d:
            # Update existing dict in d with dict v
            v = update_dict(d.get(k, {}), v)
        d[k] = v
    return d
