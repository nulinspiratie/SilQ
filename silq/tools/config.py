from typing import Any
import os
import collections
from blinker import signal, Signal
import json
from functools import partial
import copy

import qcodes as qc
from qcodes.config.config import DotDict
from qcodes.utils.helpers import SignalEmitter

import silq

__all__ = ['SubConfig', 'DictConfig', 'ListConfig', 'update_dict']

class SubConfig:
    """Extended config used within ``qcodes.config``.

    The SubConfig is a modified version of the qcodes config, the root being in
    ``qcodes.config.user``. It is used as the SilQ config (silq.config),
    initialized when silq is imported and populated during `silq.initialize`
    with the respective experiment config.

    SubConfigs can be nested, and there is a SubConfig child class for each
    type (`DictConfig` for dicts, `ListConfig` for lists, etc.). These should
    be automatically instantiated when adding a dict/list to a SubConfig.

    The SubConfig contains two main extensions over the qcodes config:

    1. Support for saving/loading the config as a JSON folder structure
       This simplifies editing part of the config in an editor.
       Each subconfig can be set to either save as a folder or as a file via the
       ``save_as_dir`` attribute.
    2. Emit a signal when a value changes. The signal  uses ``blinker.signal``,
       the signal name being ``config:{config_path}``, where ``config_path`` is
       a dot-separated path to of the config. For example, setting:

       >>> silq.config.environment1.key1 = val

       The config path is equal to ``qcodes.config.user.environment1.key1`` and
       emits the following signal:

       >>> signal('config:environment1').send(self, key1=val)

       This signal can then by picked up by other objects such as `Pulse`, to
       update its attribtues from the config.

    Parameters:
        name: Config name. SilQ config root is ``config``.
        folder: Absolute config folder path. Automatically set for child
            SubConfigs in the root SubConfig.
        parent: Parent SubConfig (None for root SubConfig).
        save_as_dir: Save SubConfig as dir. If False, SubConfig and all elements
            in it are saved as a JSON file. If True, SubConfig is saved as a
            folder, each dict key being a separate JSON file.
    """
    def __init__(self,
                 name: str,
                 folder: str = None,
                 parent: 'SubConfig' = None,
                 save_as_dir: bool = None):
        # Set through __dict__ since setattr may be overridden
        self.name = name
        self.folder = folder
        # TODO: modify
        self.parent = parent
        self.save_as_dir = save_as_dir

        qc.config.user.update({name: self})

    @property
    def config_path(self):
        """SubConfig path, e.g. ``config:dot.separated.path``"""
        if self.parent is None:
            return f'config:'
        else:
            parent_path = self.parent.config_path
            if parent_path == 'config:' and self.name == silq.environment:
                return 'environment:'
            else:
                # Ancestor of either config: or environment:
                if parent_path[-1] != ':':
                    # Not direct ancestor
                    parent_path += '.'
                return parent_path + self.name

    def load(self,
             folder: str = None):
        """Load config from folder.

        The folder must either contain a {self.name}.json file,
        or alternatively a folder containing config files/folders.
        In the latter case, a dict is created, and all the files/folders in
        the folder will be elements of the dict.

        If ``save_as_dir`` attribute is None, it will be updated to either True
        or False depending if there is a subfolder or file to load from the
        folder, respectively.

        Note that `SubConfig.load` returns a dict/list, which should then be
        added to the Subconfig object depending on the subclass. This should be
        implemented in the load method of subclasses.

        Args:
            folder: folder to look for. If not provided, uses ``self.folder``.

        Returns:
            dict/list config, to be added by the ``load`` method of the subclass
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

    def save(self,
             folder: str = None,
             save_as_dir: bool = None,
             dependent_value: bool = False):
        """Save SubConfig as JSON files in folder structure.

        Calling this method iteratively calls the same method on each of its
        elements. The folder structure is determined by the ``save_as_dir``
        attribute.

        Args:
            folder: Folder in which to save SubConfig. If ``None``, uses
                ``self.folder``. Automatically passed for child SubConfigs.
            save_as_dir: Save SubConfig as folder, in which each element is
                a key. If ``None``, uses ``self.save_as_dir``. Automatically set
                to ``None`` for all child SubConfigs.
        """
        if folder is None:
            folder = self.folder
        if save_as_dir is None:
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


class DictConfig(SubConfig, DotDict, SignalEmitter):
    """`SubConfig` for dictionaries, extension of ``qcodes.config``.

    This is a SubConfig child class for dictionaries.

    The DictConfig is a ``DotDict``, meaning that its elements can be accessed
    as attributes. For example, the following lines are identical:

    >>> dict_config['item1']['item2']
    >>> dict_config.item1.item2

    Args:
        name: Config name. SilQ config root is ``config``.
        folder: Absolute config folder path. Automatically set for child
            SubConfigs in the root SubConfig.
        parent: Parent SubConfig (None for root SubConfig).
        config: Pre-existing config to load into new DictConfig.
        save_as_dir: Save SubConfig as dir. If False, SubConfig and all elements
            in it are saved as a JSON file. If True, SubConfig is saved as a
            folder, each dict key being a separate JSON file.
    """
    exclude_from_dict = ['name', 'folder', '_mirrored_config_attrs', 'parent',
                         'signal', '_signal_chain',
                         'save_as_dir', 'config_path']

    signal = Signal()

    def __init__(self,
                 name: str,
                 folder: str = None,
                 parent: SubConfig = None,
                 config: dict = None,
                 save_as_dir: bool = None):
        self._mirrored_config_attrs = {}
        self._inherited_configs = []

        DotDict.__init__(self)
        SubConfig.__init__(self, name=name, folder=folder, parent=parent,
                           save_as_dir=save_as_dir)
        SignalEmitter.__init__(self, initialize_signal=False)


        if config is not None:
            update_dict(self, config)
        elif folder is not None:
            self.load()

    def __contains__(self, key):
        if DotDict.__contains__(self, key):
            return True
        elif DotDict.__contains__(self, 'inherit'):
            try:
                return key in self['inherit']
            except KeyError:
                return False
        else:
            return False

    def __getitem__(self, key):
        if key.startswith('config:'):
            if self.parent is None:
                return self[key.strip('config:')]
            else:
                # Pass config:path along to parent
                return self.parent[key]
        elif key.startswith('environment:'):
            if self.parent is None:
                if silq.environment is None:
                    return self[key.strip('environment:')]
                else:
                    return self[silq.environment][key.strip('environment:')]
            else:
                # Pass environment:path along to parent
                return self.parent[key]
        elif key == 'inherit':
            val = DotDict.__getitem__(self, key)
            if val.startswith('config:') or val.startswith('environment:'):
                return self[val]
            elif self.parent is not None:
                # Inherit is sibling of current dict item
                return self.parent[val]
            else:
                raise KeyError('Could not find inheriting config')
        elif DotDict.__contains__(self, key):
            val = DotDict.__getitem__(self, key)
            if isinstance(val, str) and (val.startswith('config:')
                                         or val.startswith('environment:')):
                return self[val]
            else:
                return val
        elif 'inherit' in self:
            return self['inherit'][key]
        else:
            raise KeyError

    def __setitem__(self, key, val):
        if not isinstance(key, str):
            raise TypeError(f'Config key {key} must have type str, not {type(key)}')

        # Update item in dict (modified version of DotDict)
        if '.' in key:
            myKey, restOfKey = key.split('.', 1)
            self.setdefault(myKey, DictConfig(name=myKey,
                                              config={restOfKey: val},
                                              parent=self))
        else:
            if isinstance(val, SubConfig):
                val.parent = self
                dict.__setitem__(self, key, val)
            elif isinstance(val, dict):
                # First set item, then update the dict. This avoids circular
                # referencing from mirrored attributes
                dict.__setitem__(self, key, DictConfig(name=key, parent=self))
                update_dict(self[key], val)
            elif isinstance(val, list):
                dict.__setitem__(self, key, ListConfig(name=key, parent=self))
                self[key] += val
            else:
                dict.__setitem__(self, key, val)
                if key == 'inherit':
                    # Register inheritance for signal sending
                    self[val]._inherited_configs.append(self.config_path)

        if isinstance(val, str) and (val.startswith('config:')
                                     or val.startswith('environment:')):
            # item should mirror another config item.
            if '.' in val:
                target_config_path, target_attr = val.rsplit('.', maxsplit=1)
            else:
                target_config_path = ['environment:', 'config:'][val.startswith('config:')]
            target_config = self[target_config_path]

            if not target_attr in target_config:
                raise KeyError(f'{target_config} does not have {target_attr}')

            if target_attr not in target_config._mirrored_config_attrs:
                target_config._mirrored_config_attrs[target_attr] = []

            target_config._mirrored_config_attrs[target_attr].append((self.config_path, key))

        if hasattr(self, key):
            # Add key to config path before sending
            delimiter = '' if self.config_path.endswith(':') else '.'
            attr_config_path = f'{self.config_path}{delimiter}{key}'

            # We make sure to get the value, in case the original value is mirrored
            self.signal.send(attr_config_path, value=self[key])
            if silq.environment is None:
                attr_environment_config_path = attr_config_path.replace(
                    'config:', 'environment:')
                self.signal.send(attr_environment_config_path, value=self[key])

            # If any other config attributes mirror the
            mirrored_config_attrs = self._mirrored_config_attrs.get(key, [])
            updated_mirrored_config_attrs = []
            for (mirrored_config_path, mirrored_attr) in mirrored_config_attrs:
                try:
                    # Check if mirrored attr value still referencing current
                    # attr. Getting the unreferenced value is a bit cumbersome
                    mirrored_config = self[mirrored_config_path]
                    mirrored_val = dict.__getitem__(mirrored_config, mirrored_attr)
                    if  mirrored_val == attr_config_path:

                        delimiter = '' if mirrored_config_path.endswith(':') else '.'
                        mirrored_attr_path = f'{mirrored_config_path}{delimiter}{mirrored_attr}'
                        self.signal.send(mirrored_attr_path, value=self[key])

                        if silq.environment is None:
                            mirrored_attr_environment_path = mirrored_attr_path.replace(
                                'config:', 'environment:')
                            self.signal.send(mirrored_attr_environment_path, value=self[key])

                        updated_mirrored_config_attrs.append((mirrored_config_path,
                                                              mirrored_attr))
                except KeyError:
                    pass
            if updated_mirrored_config_attrs:
                self._mirrored_config_attrs[key] = updated_mirrored_config_attrs

        else:
            print(f'Somehow after config.__setitem__ we dont have key {key}')

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self, dependent_value=True):
        if dependent_value:
            return {key: self[key] for key in self.keys()}.items()
        else:
            return {key: dict.__getitem__(self, key) for key in self.keys()}.items()

    def get(self, key: str,
            default: Any = None):
        """Override dictionary get, because it does not call __getitem__.

        Args:
            key: key to get
            default: default value if key not found. None by default

        Returns:
            value of key if in dictionary, else default value.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def load(self,
             folder: str = None,
             update: bool = True):
        """Load SubConfig from folder.

        Args:
            folder: Folder from which to load SubConfig.
        """
        if update:
            self.clear()
        config = super().load(folder=folder)
        if update:
            update_dict(self, config)
        return config

    def to_dict(self, dependent_value: bool = True):
        """Convert DictConfig including all its children to a dictionary."""
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
    """`SubConfig` for lists, extension of ``qcodes.config``.

    This is a SubConfig child class for lists.

    Args:
        name: Config name. SilQ config root is ``config``.
        folder: Absolute config folder path. Automatically set for child
            SubConfigs in the root SubConfig.
        parent: Parent SubConfig (None for root SubConfig).
        config: Pre-existing config to load into new ListConfig.
        save_as_dir: Save SubConfig as dir. If False, SubConfig and all elements
            in it are saved as a JSON file. If True, SubConfig is saved as a
            folder, each dict key being a separate JSON file.
    """

    def __init__(self, name, folder=None, parent=None, config=None, **kwargs):
        list().__init__(self)
        SubConfig.__init__(self, name=name, folder=folder, parent=parent)

        if config is not None:
            self += config
        elif folder is not None:
            self.load()

    def load(self,
             folder: str = None,
             update=True):
        """Load SubConfig from folder.

        Args:
            folder: Folder from which to load SubConfig.
            update: update current config
        """
        if update:
            self.clear()
        config = super().load(folder=folder)
        if update:
            self += config
        return config

    def to_list(self, dependent_value=True):
        """Convert Listconfig including all children into a list"""
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
    """ Update dictionary recursively.

    this ensures that subdicts are also converted
    This is a modified version of the update function in qcodes config
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping) and k in d:
            existing_val = d.setdefault(k, {})
            # Update existing dict in d with dict v
            v = update_dict(existing_val, v)
        if k == 'inherit':
            # Treat inherit specially, because it references another dict
            dict.__setitem__(d, k, v)
        else:
            d[k] = v
    return d
