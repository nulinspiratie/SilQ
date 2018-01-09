from typing import Any
import os
import collections
from blinker import signal
import json
from functools import partial
import copy

import qcodes as qc
from qcodes.config.config import DotDict

__all__ = ['SubConfig', 'DictConfig', 'ListConfig', 'update']

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
        self._connected_attrs = {}
        self.parent = parent
        self.save_as_dir = save_as_dir

        qc.config.user.update({name: self})

    @property
    def config_path(self):
        """SubConfig path, e.g. ``config:dot.separated.path``"""
        if self.parent is None:
            return f'{self.name}:'
        else:
            parent_path = self.parent.config_path
            if parent_path[-1] != ':':
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
                self.save_as_dir = True

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

    def save(self,
             folder: str = None,
             save_as_dir: bool = None):
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
        if folder == None:
            folder = self.folder
        if save_as_dir == None:
            save_as_dir = self.save_as_dir

        if not save_as_dir:
            filepath = os.path.join(folder, f'{self.name}.json')
            with open(filepath, 'w') as fp:
                json.dump(self, fp, indent=4)
        else:
            folderpath = os.path.join(folder, self.name)
            if not os.path.isdir(folderpath):
                os.mkdir((folderpath))
            for subconfig in self.values():
                subconfig.save(folder=folderpath)


class DictConfig(SubConfig, DotDict):
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
    exclude_from_dict = ['name', 'folder', '_connected_attrs', 'parent',
                         'save_as_dir', 'config_path']
    def __init__(self,
                 name: str,
                 folder: str = None,
                 parent: SubConfig = None,
                 config: dict = None,
                 save_as_dir: bool = None):
        DotDict.__init__(self)
        SubConfig.__init__(self, name=name, folder=folder, parent=parent,
                           save_as_dir=save_as_dir)

        if config is not None:
            update(self, config)
        elif folder is not None:
            self.load()

    def __getitem__(self, key):
        val = DotDict.__getitem__(self, key)
        if isinstance(val, str) and 'config:' in val:
            val = qc.config['user'].__getitem__(val[7:])
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

    def items(self):
        return {key: self[key] for key in self.keys()}.items()

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
            return None

    def _handle_config_signal(self, dependent_attr,  listen_attr, _, **kwargs):
        """Sends signal when listened property of dependent property is updated.

        Args:
            dependent_attr: name of dependent attribute
            listen_attr: name of attribute that is listened.
            _: sender object (not important)
            **kwargs: {listened attr: val}
                The dependent attribute mirrors the value of the listened
                attribute
        """
        sender_key, sender_val = kwargs.popitem()
        if sender_key == listen_attr:
            signal(self.config_path).send(self, **{dependent_attr: sender_val})

    def load(self,
             folder: str = None):
        """Load SubConfig from folder.

        Args:
            folder: Folder from which to load SubConfig.
        """
        self.clear()
        config = super().load(folder=folder)
        update(self, config)

    def to_dict(self):
        """Convert DictConfig including all its children to a dictionary."""
        d = {}
        for key, val in self.items():
            if isinstance(val, DictConfig):
                d[key] = val.to_dict()
            elif isinstance(val, ListConfig):
                d[key] = val.to_list()
            else:
                d[key] = val
        return d

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
             folder: str = None):
        """Load SubConfig from folder.

        Args:
            folder: Folder from which to load SubConfig.
        """
        self.clear()
        config = super().load(folder=folder)
        self += config

    def to_list(self):
        """Convert Listconfig including all children into a list"""
        l = []
        for val in self:
            if isinstance(val, DictConfig):
                l.append(val.to_dict())
            elif isinstance(val, ListConfig):
                l.append(val.to_list())
            else:
                l.append(val)
        return l

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.to_list())

def update(d, u):
    """ Update dictionary recursively.

    this ensures that subdicts are also converted
    This is a modified version of the update function in qcodes config
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping) and k in d:
            # Update existing dict in d with dict v
            v = update(d.get(k, {}), v)
        d[k] = v
    return d
