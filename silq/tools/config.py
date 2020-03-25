from typing import Any, List, Union, Tuple
import warnings
import os
import collections
from blinker import signal, Signal
import json
from functools import partial
import copy
import logging

import qcodes as qc
from qcodes.config.config import DotDict
from qcodes.utils.helpers import SignalEmitter

import silq

__all__ = ['SubConfig', 'DictConfig', 'ListConfig', 'update_dict']

logger = logging.getLogger(__name__)


class SubConfig:
    """Config with added functionality, used within ``qcodes.config.user``.

    The SubConfig is a modified version of the qcodes config, the root being
    ``qcodes.config.user.silq_config``. It is used as the SilQ config
    (silq.config), attached during importing silq, and initialized during
    `silq.initialize` with the respective experiment config.

    SubConfigs can be nested, and there is a SubConfig child class for subtypes
    (`DictConfig` for dicts, `ListConfig` for lists). These are automatically
    instantiated when adding a dict/list to a SubConfig.

    The SubConfig contains the following main extensions over the qcodes config:

    1. Support for saving/loading the config as a JSON folder structure
       This simplifies editing part of the config in an editor.
       Each subconfig can be set to either save as a folder or as a file via the
       ``save_as_dir`` attribute.
    3. Handling of an environment (set by silq.environment). The environment
       (string) the key of a dict in silq.config (top-level). If the environment
       (string) is set, any call to `environment:{path}` will access
       `silq.config.{silq.environment}.{path}
       This allows easy switching between config settings.
    2. Emit a signal when a value changes. The signal  uses ``blinker.signal``,
       the signal name being ``config:{config_path}``, where ``config_path`` is
       a dot-separated path to of the config. For example, setting:

       >>> silq.config.properties.key1 = val

       The path is equal to ``qcodes.config.user.silq_config.properties.key1``
       and emits a signal with sender `config:properties.key1` and keyword
       argument `value=val`.
       This signal can then by picked up by other objects, in particular by
       parameters via its initialization kwarg `config_link`. This means that
       whenever that specific config value changes, the parameter is updated
       accordingly.

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
        the folder will be elements of the dict. All '.ipynb_checkpoints'
        folders will be ignored.

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
            try:
                with open(filepath, "r") as fp:
                    config = json.load(fp)
            except Exception as e:
                e.args = (e.args[0] + f'\nError reading json file {filepath}',
                          *e.args[1:])
                raise e

        elif os.path.isdir(folderpath):
            # Config is a folder, and so each item in the folder is added to
            # a dict.
            config = {}

            # Update self.save_as_dir to False unless explicitly set to True
            if self.save_as_dir is None:
                self.save_as_dir = False

            for file in os.listdir(folderpath):
                filepath = os.path.join(folderpath, file)
                if file.endswith(".json"):
                    with open(filepath, "r") as fp:
                        # Determine type of config
                        try:
                            subconfig = json.load(fp)
                        except Exception as e:
                            e.args = (
                                e.args[0] + f'\nError reading json file {filepath}',
                                *e.args[1:]
                            )
                            raise e

                        if isinstance(subconfig, list):
                            config_class = ListConfig
                        elif isinstance(subconfig, dict):
                            config_class = DictConfig
                        else:
                            raise RuntimeError(f'Could not load config file '
                                               f'{filepath}')

                        subconfig_name = file.split('.')[0]
                        subconfig = config_class(
                            name=subconfig_name,
                            folder=folderpath,
                            save_as_dir=False,
                            parent=self
                        )
                elif os.path.isdir(filepath):
                    if ".ipynb_checkpoints" in filepath:
                        continue
                    subconfig_name = file
                    subconfig = DictConfig(name=file,
                                           folder=folderpath,
                                           save_as_dir=True,
                                           parent=self)
                else:
                    logger.warning(f"Could not load {filepath} to config")
                config[subconfig_name] = subconfig

        else:
            raise FileNotFoundError(
                f"No file nor folder found to load for {self.name}")

        return config

    def refresh(self, config=None):
        if config is None:
            # Temporarily remove signal so it doesn't send many signals
            signal, DictConfig.signal = DictConfig.signal, Signal()

            config = DictConfig(name=self.name,
                                folder=self.folder,
                                parent=None,
                                save_as_dir=self.save_as_dir)

            # Restore signal
            DictConfig.signal = signal

        if isinstance(config, dict) and isinstance(self, dict):
            for key, val in config.items():
                if key in self:
                    if isinstance(self[key], SubConfig):
                        self[key].refresh(config=config[key])
                    elif self[key] != val:
                        logger.info(f'{self.config_path}.{key} changed from '
                                    f'{self[key]} to {val}')
                        self[key] = val

                else:
                    logger.info(f'New key {self.config_path}.{key} = val')
                    self[key] = val

            # Also remove any keys that are not in the new config
            for key in list(self):
                if key not in config:
                    logger.info(f'{self.config_path}.{key} not in new config')
                    self.pop(key)

        elif isinstance(config, list) and isinstance(self, list):
            if config != self:
                logger.info(f'{self.config_path} list differs to {config}')
                self.clear()
                self += config
        else:
            raise TypeError(
                f'{self.config_path} has different type as refreshed '
                f'config {config}'
            )

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
    exclude_from_dict = ['name', 'folder', 'parent', 'initializing',
                         'signal', '_signal_chain', '_signal_modifiers',
                         '_mirrored_config_attrs', '_inherited_configs',
                         'save_as_dir', 'config_path',
                         'sender', 'multiple_senders']

    signal = Signal()

    def __init__(self,
                 name: str,
                 folder: str = None,
                 parent: SubConfig = None,
                 config: dict = None,
                 save_as_dir: bool = None):
        self.initializing = True
        self._mirrored_config_attrs = {}
        self._inherited_configs = []

        SubConfig.__init__(self, name=name, folder=folder, parent=parent,
                           save_as_dir=save_as_dir)
        DotDict.__init__(self)
        SignalEmitter.__init__(self, initialize_signal=False)

        if config is not None:
            update_dict(self, config)
        elif folder is not None:
            self.load()

        if self.parent is None:
            self._attach_mirrored_items()

    def __contains__(self, key):
        if DotDict.__contains__(self, key):
            return True
        elif DotDict.__contains__(self, 'inherit'):
            try:
                if self['inherit'].startswith('config:') or \
                        self['inherit'].startswith('environment:'):
                    return key in self[self['inherit']]
                else:
                    return key in self.parent[self['inherit']]
            except KeyError:
                return False
        else:
            return False

    def __getitem__(self, key):
        if key.startswith('config:'):
            if self.parent is not None:
                # Let parent config deal with this
                return self.parent[key]
            elif key == 'config:':
                return self
            else:
                return self[key.replace('config:', '')]
        elif key.startswith('environment:'):
            if self.parent is None:
                if silq.environment is None:
                    environment_config = self
                else:
                    environment_config = self[silq.environment]

                if key == 'environment:':
                    return environment_config
                else:
                    return environment_config[key.replace('environment:', '')]
            else:
                # Pass environment:path along to parent
                return self.parent[key]
        elif DotDict.__contains__(self, key):
            val = DotDict.__getitem__(self, key)
            if key == 'inherit':
                return val
            elif isinstance(val, str) and \
                    (val.startswith('config:') or val.startswith('environment:')):
                try:
                    return self[val]
                except KeyError:
                    raise KeyError(f"Couldn't retrieve mirrored key {key} -> {val}")
            else:
                return val
        elif 'inherit' in self:
            if self['inherit'].startswith('config:') or \
                    self['inherit'].startswith('environment:'):
                return self[self['inherit']][key]
            else:
                return self.parent[self['inherit']][key]
        else:
            raise KeyError(f"Couldn't retrieve key {key}")

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
                sub_dict = DictConfig(name=key, parent=self)
                dict.__setitem__(self, key, sub_dict)
                update_dict(self[key], val)
                # If self.initializing, sub_dict._attach_mirrored_items will be
                # called at the end of initialization, otherwise call now
                if not self.initializing:
                    sub_dict._attach_mirrored_items()
            elif isinstance(val, list):
                dict.__setitem__(self, key, ListConfig(name=key, parent=self))
                self[key] += val
            else:
                dict.__setitem__(self, key, val)
                if (self.initializing
                        and (key == 'inherit'
                             or (isinstance(val, str)
                                 and (val.startswith('config:')
                                      or val.startswith('environment:'))))):
                    return

                if key == 'inherit':
                    if (val.startswith('config:') or val.startswith('environment:')):
                        config_path = val
                    else:
                        # inherit a neighbouring dict element
                        config_path = join_config_path(self.parent.config_path, val)

                    # Register inheritance for signal sending
                    self[config_path]._inherited_configs.append(self.config_path)

        if isinstance(val, str) and (val.startswith('config:')
                                     or val.startswith('environment:')):
            # item should mirror another config item.
            target_config_path, target_attr = split_config_path(val)
            target_config = self[target_config_path]

            if not target_attr in target_config:
                raise KeyError(f'{target_config} does not have {target_attr}')

            if target_attr not in target_config._mirrored_config_attrs:
                target_config._mirrored_config_attrs[target_attr] = []

            target_config._mirrored_config_attrs[target_attr].append(
                (self.config_path, key)
            )

        # Retrieve value from self, which also handles mirroring/inheriting
        value = self[key]

        # Add key to config path before sending
        attr_config_path = join_config_path(self.config_path, key)

        # We make sure to get the value, in case the original value is mirrored
        self.signal.send(attr_config_path, value=value)
        if silq.environment is None:
            attr_environment_config_path = attr_config_path.replace(
                'config:', 'environment:')
            self.signal.send(attr_environment_config_path, value=value)

        # If any other config attributes mirror the attribute being set,
        # also send signals with sender being the mirrored attributes
        if self._inherited_configs:
            self._inherited_configs = self._send_ancillary_signals(
                value=value, target_paths=self._inherited_configs,
                attr=key, attr_path=attr_config_path)

        # If any other config dicts inherit from this DictConfig via 'inherit',
        # Also emit signals with sender being the inherited dicts
        if self._mirrored_config_attrs.get(key, []):
            updated_mirrored_config = self._send_ancillary_signals(
                value=value, target_paths=self._mirrored_config_attrs[key],
                attr=None, attr_path=attr_config_path)
            if updated_mirrored_config:
                self._mirrored_config_attrs[key] = updated_mirrored_config
            else:
                self._mirrored_config_attrs.pop(key, None)

    def _send_ancillary_signals(self,
                                value: Any,
                                target_paths: List[Union[str, Tuple[str]]],
                                attr: str = None,
                                attr_path: str = None):
        # mirrored_config_attrs = self._mirrored_config_attrs.get(key, [])
        updated_target_paths = []
        for target_full_path in target_paths:
            try:
                if attr is None:  # Attr is the second argument of the full path
                    target_path, target_attr = target_full_path
                else:  # Use default attr
                    target_path, target_attr = target_full_path, attr

                # Check if mirrored attr value still referencing current
                # attr. Getting the unreferenced value is a bit cumbersome
                target_config = self[target_path]

                # Target either inherits all attrs of current dict, or one of
                # its attributes mirrors this attribute. Here we check if this
                # hasn't changed
                inheritance = dict.get(target_config, 'inherit', None)
                if inheritance == self.config_path \
                        or dict.get(target_config, target_attr) == attr_path \
                        or (inheritance == self.name and
                            target_config.parent ==self.parent):
                    target_attr_path = join_config_path(target_path, target_attr)

                    self.signal.send(target_attr_path, value=value)

                    if silq.environment is None:
                        target_attr_environment_path = target_attr_path.replace(
                            'config:', 'environment:')
                        self.signal.send(target_attr_environment_path, value=value)

                    updated_target_paths.append(target_full_path)
            except KeyError:
                pass
        return updated_target_paths

    def _attach_mirrored_items(self):
        """Attach mirrored items, to be done at the end of initialization.

        Mirrored items are those that inherit, or whose values start with
        ``config:`` or ``environment:``

        Note:
            Attribute ``initializing`` will be set to False
            """
        self.initializing = False
        for key, val in self.items(dependent_value=False):
            if isinstance(val, DictConfig):
                val._attach_mirrored_items()
            elif (key == 'inherit'
                  or (isinstance(val, str)
                      and (val.startswith('config:')
                           or val.startswith('environment:')))):
                self[key] = val

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
        d[k] = v
    return d


def split_config_path(config_path):
    """Splits a config path into the parent path and attr

    Args:
        config_path: Full config path

    Returns:
        parent_config_path: Everything except last element
        config_attr: final part of config path
    """
    if '.' in config_path:
        return config_path.rsplit('.', 1)
    else:
        # config path has form config:item, which should be ('config:', 'item')
        parent_config_path, config_attr = config_path.split(':')
        parent_config_path += ':'
        return parent_config_path, config_attr


def join_config_path(config_path, config_attr):
    delimiter = '' if config_path.endswith(':') else '.'
    return f'{config_path}{delimiter}{config_attr}'
