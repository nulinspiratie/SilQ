import sys
import operator
from functools import partial, wraps
import re
import numpy as np
import logging
import logging.handlers
import os
import time

from qcodes import config
from qcodes.config.config import DotDict
from qcodes.instrument.parameter import Parameter


code_labels = {}
properties_config = config['user'].get('properties', {})


def execfile(filename, globals=None, locals=None):
    if globals is None:
        globals = sys._getframe(1).f_globals
    if locals is None:
        locals = sys._getframe(1).f_locals
    with open(filename, "r") as fh:
        exec(fh.read()+"\n", globals, locals)


ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '==': operator.eq}


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
    _single_settings = {}
    _temporary_settings = {}
    def __init__(self, ignore_if_not_None=[], **kwargs):
        self._temporary_settings = {}
        self._single_settings = {}
        self._ignore_if_not_None = ignore_if_not_None

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
                    '_ignore_if_not_None', '__setstate__', '__dict__']:
            return object.__getattribute__(self, item)
        elif item in self._ignore_if_not_None and \
                        object.__getattribute__(self, item) is not None:
            return object.__getattribute__(self, item)
        elif item in self._single_settings:
            return self._single_settings[item]
        elif item in self._temporary_settings:
            return self._temporary_settings[item]
        else:
            value = object.__getattribute__(self, item)
            if value is not None:
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


class UpdateDotDict(DotDict):
    """
    DotDict that can evaluate function upon updating
    """
    exclude_from_dict = ['update_function', 'exclude_from_dict']
    def __init__(self, update_function=None, **kwargs):
        self.update_function = update_function
        super().__init__()

        for key, val in kwargs.items():
            DotDict.__setitem__(self, key, val)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.update_function is not None:
            self.update_function()


def attribute_from_config(item, config=properties_config):
    """
    Check if attribute exists somewhere in the config
    It first ill check properties config if a key matches the item
    with self.mode appended. This is only checked if the param has a mode.
    Finally, it will check if properties_config contains the item
    """
    # check if {item}_{self.mode} is in properties_config
    # if mode is None, mode_str='', in which case it checks for {item}
    if item in config:
        # Check if item is in properties config
        value = config[item]
    else:
        raise AttributeError

    if type(value) is DotDict:
        value = dict(value)

    return value


def clear_single_settings(f):
    @wraps(f)
    def clear_single_settings_decorator(self, *args, **kwargs):
        try:
            output = f(self, *args, **kwargs)
        finally:
            self._single_settings.clear()
        return output
    return clear_single_settings_decorator


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


def run_code(label, **kwargs):
    """
    Creates cell to run code from global variable code_labels
    Code labels is a dictionary in which each key has a corresponding value
    that is a string representation of executable code.
    Note that the module variable code_labels must be set to equal the global
    variable code_labels.
    Args:
        label: label referring to code in dict code_labels
        **kwargs: Optional kwargs that are replaced in code
            i.e. for a given kwarg {var}=5, a line matching:
            "{var} = {val}" will be replaced to {var} = 5" (Note whitespaces)

    Returns:
        Creates cell at bottom of notebook and executes it
    """
    from silq.tools.notebook_tools import create_cell
    code = code_labels[label]
    for var, val in kwargs.items():
        pattern = r'{} = .+'.format(var)
        repl = r'{} = {}'.format(var, val)
        code = re.sub(pattern, repl, code, count=1)
    create_cell(code, 'bottom', execute=True)


def get_exponent(val):
    if val <= 0:
        raise SyntaxError(f'Val {val} must be larger than zero')
    else:
        return int(np.floor(np.log10(val)))


def get_first_digit(val):
    first_digit = int(np.floor(val * 10 ** -get_exponent(val)))
    return first_digit


class ParallelTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    TimedRotatingFileHandler with some modifications.
    Files are stored with .log suffix, and dates are included in filename 
    from the beginning. Also allows multiple processes to access same log.
    
    From: https://stackoverflow.com/questions/24649789/how-to-force-a
    -rotating-name-with-pythons-timedrotatingfilehandler
    """
    def __init__(self, filename, when='h', interval=1, backupCount=0,
                 encoding=None, delay=False, utc=False, postfix = ".log"):
        self.origFileName = filename
        self.when = when.upper()
        self.interval = interval
        self.backupCount = backupCount
        self.utc = utc
        self.postfix = postfix

        # Seems to be needed for self.computeRollover
        self.atTime = None

        if self.when == 'S':
            self.interval = 1 # one second
            self.suffix = "%Y-%m-%d_%H-%M-%S"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        elif self.when == 'M':
            self.interval = 60 # one minute
            self.suffix = "%Y-%m-%d_%H-%M"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$"
        elif self.when == 'H':
            self.interval = 60 * 60 # one hour
            self.suffix = "%Y-%m-%d_%H"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}$"
        elif self.when == 'D' or self.when == 'MIDNIGHT':
            self.interval = 60 * 60 * 24 # one day
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}$"
        elif self.when.startswith('W'):
            self.interval = 60 * 60 * 24 * 7 # one week
            if len(self.when) != 2:
                raise ValueError("You must specify a day for weekly rollover from 0 to 6 (0 is Monday): %s" % self.when)
            if self.when[1] < '0' or self.when[1] > '6':
                 raise ValueError("Invalid day specified for weekly rollover: %s" % self.when)
            self.dayOfWeek = int(self.when[1])
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}$"
        else:
            raise ValueError("Invalid rollover interval specified: %s" % self.when)

        currenttime = int(time.time())
        logging.handlers.BaseRotatingHandler.__init__(self, self.calculateFileName(currenttime), 'a', encoding, delay)

        self.extMatch = re.compile(self.extMatch)
        self.interval = self.interval * interval # multiply by units requested

        self.rolloverAt = self.computeRollover(currenttime)

    def calculateFileName(self, currenttime):
        if self.utc:
             timeTuple = time.gmtime(currenttime)
        else:
             timeTuple = time.localtime(currenttime)

        return self.origFileName + "." + time.strftime(self.suffix, timeTuple) + self.postfix

    def getFilesToDelete(self, newFileName):
        dirName, fName = os.path.split(self.origFileName)
        dName, newFileName = os.path.split(newFileName)

        fileNames = os.listdir(dirName)
        result = []
        prefix = fName + "."
        postfix = self.postfix
        prelen = len(prefix)
        postlen = len(postfix)
        for fileName in fileNames:
            if fileName[:prelen] == prefix \
                    and fileName[-postlen:] == postfix \
                    and len(fileName)-postlen > prelen \
                    and fileName != newFileName:
                suffix = fileName[prelen:len(fileName)-postlen]
                if self.extMatch.match(suffix):
                     result.append(os.path.join(dirName, fileName))
        result.sort()
        if len(result) < self.backupCount:
            result = []
        else:
            result = result[:len(result) - self.backupCount]
        return result

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        currentTime = self.rolloverAt
        newFileName = self.calculateFileName(currentTime)
        newBaseFileName = os.path.abspath(newFileName)
        self.baseFilename = newBaseFileName
        self.mode = 'a'
        self.stream = self._open()
        if self.backupCount > 0:
            for s in self.getFilesToDelete(newFileName):
                try:
                    os.remove(s)
                except:
                    pass
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval

        #If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) \
                and not self.utc:
            dstNow = time.localtime(currentTime)[-1]
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:
                    # DST kicks in before next rollover, need to deduct an hour
                    newRolloverAt = newRolloverAt - 3600
                else:
                    # DST bows out before next rollover, need to add an hour
                    newRolloverAt = newRolloverAt + 3600
        self.rolloverAt = newRolloverAt


def convert_setpoints(*args):
    """
    Convert setpoints to tuples, supporting multidimensional setpoints.
    Temporary solution to make setpoints work (see issue #627).
    Args:
        *args: 1D setpoint arrays. each successive setpoints array gains an 
            extra dimension

    Returns:

    """
    if not args:
        return tuple()
    else:
        first_arg = tuple(args[0])
        remaining_args = convert_setpoints(*args[1:])
        if remaining_args:
            remaining_args = tuple((arg,) * len(first_arg)
                                   for arg in remaining_args)
        return (first_arg, ) + remaining_args


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def arreq_in_list(myarr, list_arrays):
    """
    Get index of array in list of arrays, testing equality
    Modified from https://stackoverflow.com/questions/23979146/
                  check-if-numpy-array-is-in-list-of-numpy-arrays
    Args:
        myarr: arr to be found in list
        list_arrays: List of numpy arrays

    Returns:
        idx of array in list equal to myarr, None if not found. 
    """
    return next((idx for idx, elem in enumerate(list_arrays)
                 if np.array_equal(elem, myarr)),
                None)


def arreqclose_in_list(myarr, list_arrays):
    """
    Get index of array in list of arrays, testing approximate equality
    Modified from https://stackoverflow.com/questions/23979146/
                  check-if-numpy-array-is-in-list-of-numpy-arrays
    Args:
        myarr: arr to be found in list
        list_arrays: List of numpy arrays

    Returns:
        idx of array in list approximately equal to myarr, None if not found. 
    """
    return next((idx for idx, elem in enumerate(list_arrays)
                 if elem.size == myarr.size
                 and np.allclose(elem, myarr)),
                None)


class property_ignore_setter(object):
    """
    Decorator similar to @property that ignores setter
    """
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __get__(self, obj, value):
        return self.func(obj)

    def __set__(self, obj, value):
        pass