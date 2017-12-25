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
    ``qcodes.config.user``. It contains two main extensions:
    
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
        name: Config name
        folder: 
        parent: 
        save_as_dir: 
    """
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

    def load(self, folder=None):
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

        filepath = os.path.join(folder, '{}.json'.format(self.name))
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

    def save(self, folder=None, save_as_dir=None):
        if folder == None:
            folder = self.folder
        if save_as_dir == None:
            save_as_dir = self.save_as_dir

        if not save_as_dir:
            filepath = os.path.join(folder, '{}.json'.format(self.name))
            with open(filepath, 'w') as fp:
                json.dump(self, fp, indent=4)
        else:
            folderpath = os.path.join(folder, self.name)
            if not os.path.isdir(folderpath):
                os.mkdir((folderpath))
            for subconfig in self.values():
                subconfig.save(folder=folderpath)


class DictConfig(SubConfig, DotDict):
    exclude_from_dict = ['name', 'folder', '_connected_attrs', 'parent',
                         'save_as_dir', 'config_path']
    def __init__(self, name, folder=None, parent=None, config=None,
                 save_as_dir=None):
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

    def load(self, folder=None):
        self.clear()
        config = super().load(folder=folder)
        update(self, config)

    def to_dict(self):
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
    def __init__(self, name, folder=None, parent=None, config=None, **kwargs):
        list().__init__(self)
        SubConfig.__init__(self, name=name, folder=folder, parent=parent)

        if config is not None:
            self += config
        elif folder is not None:
            self.load()

    def load(self, folder=None):
        self.clear()
        config = super().load(folder=folder)
        self += config

    def to_list(self):
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
    """ 
    Update dictionary recursively.
    this ensures that subdicts are also converted
    This is a modified version of the update function in qcodes config
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping) and k in d:
            # Update existing dict in d with dict v
            v = update(d.get(k, {}), v)
        d[k] = v
    return d
