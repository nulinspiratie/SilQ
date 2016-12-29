import sys
import operator
from functools import partial
from multiprocessing import active_children

from qcodes import config
from qcodes.config.config import DotDict
from qcodes.instrument.parameter import Parameter

properties_config = config['user'].get('properties', {})

def execfile(filename, globals=None, locals=None):
    if globals is None:
        globals = sys._getframe(1).f_globals
    if locals is None:
        locals = sys._getframe(1).f_locals
    with open(filename, "r") as fh:
        exec(fh.read()+"\n", globals, locals)


def get_truth(test_val, target_val, relation='=='):
    """
    Tests if the test_val satisfies the relation to target_val.
    Args:
        test_val: lhs of relation with target_val.
        target_val: rhs of relation with test_val
        relation: relation between test_val and target_val. Default is '=='
            Can be: '>', '<', '>=', '<=', '=='

    Returns:
        Bool depending on if relation holds
    """
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '==': operator.eq}
    return ops[relation](test_val, target_val)


def get_memory_usage():
    # return the memory usage in MB
    import os
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def partial_from_attr(func, obj, attr):
    """
    Returns a function that, when evaluated, evaluates func with arg obj.attr.
    This functions was created to delay the evaluation of obj.attr
    Args:
        func: func to evaluate
        obj: object from which to retrieve attr
        attr: attr of object whose value should be used as arg for func

    Returns:
        function that upon evaluation is equal to func(obj.attr)
    """
    # def newfunc():
    #     return func(getattr(obj, attr))
    return partial(func, getattr(obj, attr))


def print_attr(obj, attr):
    if not hasattr(obj, attr):
        print('Could not find attribute {} of {}'.format(attr, obj))
    print('{}.{} = {}'.format(obj.__class__.__name__, attr, getattr(obj, attr)))


class SettingsClass:
    def __init__(self, **kwargs):
        self._temporary_settings = {}
        self._single_settings = {}
        if not hasattr(self, 'mode'):
            self.mode = None

    def __getattribute__(self, item):
        """
        Called when requesting an attribute.
        The attribute is successively searched in the following places:
        1. single_settings
        2. temporary_settings
        3. self.{attr}
        4. If self.{attr} is explicitly None, it will check properties_config

        Some attributes (such as 'mode') are not allowed
        Args:
            item: Attribute to be retrieved

        Returns:

        """
        if item in ['_temporary_settings', '_single_settings',
                    '_attribute_from_config', 'mode', 'mode_str',
                    '__setstate__', '__dict__']:
            return object.__getattribute__(self, item)
        elif item in self._single_settings:
            return self._single_settings[item]
        elif item in self._temporary_settings:
            return self._temporary_settings[item]
        else:
            value = object.__getattribute__(self, item)
            if value is not None:
                return value

            value = self._attribute_from_config(item)
            return value

    @property
    def mode_str(self):
        return '' if self.mode is None else '_{}'.format(self.mode)

    def _attribute_from_config(self, item):
        """
        Check if attribute exists somewhere in the config
        It first ill check properties config if a key matches the item
        with self.mode appended. This is only checked if the param has a mode.
        Finally, it will check if properties_config contains the item
        """
        # check if {item}_{self.mode} is in properties_config
        # if mode is None, mode_str='', in which case it checks for {item}
        item_mode = '{}{}'.format(item, self.mode_str)
        if item_mode in properties_config:
            value = properties_config[item_mode]
        elif item in properties_config:
            # Check if item is in properties config
            value = properties_config[item]
        else:
            value = None

        if type(value) is DotDict:
            value = dict(value)

        return value

    def settings(self, **kwargs):
        """
        Sets up the meta properties of a measurement parameter
        """
        for item, value in kwargs.items():
            if hasattr(self, item):
                setattr(self, item, value)
            else:
                raise ValueError('Setting {} not found'.format(item))

    def temporary_settings(self, **kwargs):
        """
        Sets up the meta properties of a measurement parameter
        """
        if not kwargs:
            return self._temporary_settings

        self._temporary_settings.clear()
        for item, value in kwargs.items():
            if hasattr(self, item):
                self._temporary_settings[item] = value
            else:
                raise ValueError('Setting {} not found'.format(item))

    def single_settings(self, **kwargs):
        """
        Sets up the meta properties of a measurement parameter
        """
        if not kwargs:
            return self._single_settings

        self._single_settings.clear()
        for item, value in kwargs.items():
            if hasattr(self, item):
                self._single_settings[item] = value
            else:
                raise ValueError('Setting {} not found'.format(item))

    def clear_settings(self):
        """
        Clears temporary and single settings
        """
        self._temporary_settings.clear()
        self._single_settings.clear()


def clear_single_settings(f):
    def clear_single_settings_decorator(self):
        output = f(self)
        self._single_settings.clear()
        return output
    return clear_single_settings_decorator


def terminate_servers():
    for server in active_children():
        server.terminate()

def JSONListEncoder(l):
    return_l = []
    for element in l:
        if hasattr(element, '_JSONEncoder'):
            return_l.append(element._JSONEncoder())
        elif isinstance(element, (list, tuple)):
            return_l.append(JSONListEncoder(element))
        else:
            return_l.append(repr(element))
    return return_l

def JSONEncoder(obj, ignore_attrs=[], ignore_vals=[]):
    return_dict = {}
    for attr, val in vars(obj).items():
        if attr in ignore_attrs:
            continue
        if hasattr(val, '_JSONEncoder'):
            val = val._JSONEncoder()
        elif isinstance(val, (list, tuple)):
            val = JSONListEncoder(val)

        if isinstance(val, Parameter):
            return_dict[attr] = val.name
        elif val not in ignore_vals:
            return_dict[attr] = val

    return_dict['class'] = obj.__class__.__name__
    return return_dict