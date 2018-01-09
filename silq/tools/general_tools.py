from typing import Any, List
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

__all__ = ['execfile', 'get_truth', 'get_memory_usage', 'SettingsClass',
           'UpdateDotDict',
           'attribute_from_config', 'clear_single_settings', 'JSONEncoder',
           'JSONListEncoder', 'run_code', 'get_exponent', 'get_first_digit',
           'ParallelTimedRotatingFileHandler', 'convert_setpoints',
           'Singleton', 'arreq_in_list', 'arreqclose_in_list',
           'property_ignore_setter', 'freq_to_str']

code_labels = {}
properties_config = config['user'].get('properties', {})


def execfile(filename: str,
             globals: dict = None,
             locals: dict = None):
    """Execute code in .py file, adding new variables to globals/locals.

    Args:
        globals: Global variables dictionary. If not specified, uses globals
            in first frame.
        locals: Local variables dictionary. If not specified, uses locals in
            first frame.
        """
    if globals is None:
        globals = sys._getframe(1).f_globals
    if locals is None:
        locals = sys._getframe(1).f_locals
    with open(filename, "r") as fh:
        exec(fh.read()+"\n", globals, locals)

"""Comparison operators, used for `get_truth`"""
ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '==': operator.eq}


def get_truth(test_val: Any,
              target_val: Any,
              relation: str = '==') -> bool:
    """Tests if the ``test_val`` satisfies the ``relation`` to ``target_val``.

    Args:
        test_val: lhs of relation with target_val.
        target_val: rhs of relation with test_val
        relation: relation between test_val and target_val. Default is '=='
            Can be: '>', '<', '>=', '<=', '=='

    Returns:
        True if relation holds
    """
    return ops[relation](test_val, target_val)


def get_memory_usage():
    """return the memory usage in MB"""
    import os
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


class SettingsClass:
    """
    Class used to temporarily override attributes.
    This can be done through obj.single_settings() and obj.temporary_settings().
    Any settings specified here will override the actual values of the object
    until settings are cleared

    Settings can be cleared in two ways:
    Using the decorator @clear_single_settings on a method, which will delete
    the single_settings.
    Using obj.clear_settings(), which will clear both the single settings and
    temporary settings.

    Furthermore, attribute_names can be added to ignore_if_set.
    If the object's value of that attribute is not equal to None, [], or (),
    it cannot be overridden through single or temporary settings.

    Note that for all attributes, they must be set in the object before they
    can be overridden by single/temporary settings
    """
    _single_settings = {}
    _temporary_settings = {}
    _ignore_if_set = {}
    def __init__(self, ignore_if_set=[], **kwargs):
        self._temporary_settings = {}
        self._single_settings = {}
        self._ignore_if_set = ignore_if_set

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
                    '_ignore_if_set', '__setstate__', '__dict__']:
            return object.__getattribute__(self, item)
        elif item in self._ignore_if_set and \
                        object.__getattribute__(self, item) not in \
                        (None, [], ()):
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
        return self

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
        return self

    def clear_settings(self):
        """
        Clears temporary and single settings
        """
        self._temporary_settings.clear()
        self._single_settings.clear()


class UpdateDotDict(DotDict):
    """DotDict that can evaluate function upon being updated.

    Args:
        update_function: Function that is called every time a value changes.
        **kwargs: Unused kwargs.
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


def attribute_from_config(item: str, config: dict):
    """Check if attribute exists is an item in the config

    Args:
        item: key in config to check
        config: config to check

    Returns:
        Value in config. If dict-like, will be converted to dict.

    Raises:
        AttributeError: item not found.

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
    """`SettingsClass` wrapper to clear single_settings after running function.

    Args:
        f: function after which to clear ``single_settings`` attribute.
    """
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


def JSONEncoder(obj,
                ignore_attrs: List[str] = [],
                ignore_vals: List[str]=[]):
    """Encode object as dict for JSON encoding.

    Args:
        ignore_attrs: Attributes that should not be included.
        ignore_vals: Vals to be ignored.

    Returns:
        dict representation of object.

    Notes:
        # If one of its attributes is an object containing method ``_JSONEncoder``,
          the method is called to get its JSON representation.
        # Lists are encoded using `JSONListEncoder`, which has an additional
          check for the method ``_JSONEncoder``.

    Todo:
        Ensure parameters etc. are encoded using their snapshot.
    """
    return_dict = {}
    for attr, val in vars(obj).items():
        if attr in ignore_attrs:
            continue
        if hasattr(val, '_JSONEncoder'):
            val = val._JSONEncoder()
        elif isinstance(val, (list, tuple)):
            val = JSONListEncoder(val)

        if isinstance(val, Parameter):
            # TODO: is this right?
            return_dict[attr] = val.name
        elif val not in ignore_vals:
            return_dict[attr] = val

    return_dict['class'] = obj.__class__.__name__
    return return_dict


def run_code(label, **kwargs):
    """Creates cell to run code from global variable code_labels

    Code labels is a dictionary in which each key has a corresponding value
    that is a string representation of executable code.
    Note that the module variable code_labels must be set to equal the global
    variable code_labels.

    Args:
        label: label referring to code in dict code_labels
        **kwargs: Optional kwargs that are replaced in code
            i.e. for a given kwarg {var}=5, a line matching:
            "{var} = {val}" will be replaced to "{var} = 5" (Note whitespaces)

    Returns:
        Creates cell at bottom of notebook and executes it

    Note:
        This function is not used anymore, though it should still work
    """
    from silq.tools.notebook_tools import create_cell
    code = code_labels[label]
    for var, val in kwargs.items():
        pattern = r'{} = .+'.format(var)
        repl = r'{} = {}'.format(var, val)
        code = re.sub(pattern, repl, code, count=1)
    create_cell(code, 'bottom', execute=True)


def get_exponent(val: float):
    """Get decimal exponent

    Example:
        >>> get_exponent(0.032)
        -2

    Args:
        val: Val of which to get exponent

    Returns:
        Exponent
    """
    if val <= 0:
        raise SyntaxError(f'Val {val} must be larger than zero')
    else:
        return int(np.floor(np.log10(val)))


def get_first_digit(val: float):
    """Get first nonzero digit.

    Example:
        >>> get_first_digit(0.032)
        3

    Args:
        val: Val for which to get first nonzero digit

    Returns:
        First nonero digit.
    """
    first_digit = int(np.floor(val * 10 ** -get_exponent(val)))
    return first_digit


class ParallelTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Logging handler that creates a new log file every day.

    Files are stored with .log suffix, and dates are included in filename
    from the beginning. Also allows multiple processes to access same log.

    Note:
        - From: https://stackoverflow.com/questions/24649789/how-to-force-a
          -rotating-name-with-pythons-timedrotatingfilehandler
        - Essentially a TimedRotatingFileHandler with some modifications.
        - Haven't gone through the code, but it does the trick.
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
    """Convert setpoints to tuples, supporting multidimensional setpoints.

    Temporary solution to make setpoints work (see issue #627).
    Currently, setpoints need to be tuples (arrays etc. give issues). Further,
    the second setpoint array needs to be 2D etc.

    Args:
        *args: 1D setpoint arrays. each successive setpoints array gains an
            extra dimension

    Returns:
        Setpoint arrays converted to tuples.
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
    """Meta-class for classes that can only have a single instance.

    If a second instance is created, it will instead return the already-existing
    instance.
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def arreq_in_list(myarr: np.ndarray,
                  list_arrays: List[np.ndarray]):
    """Get index of array in list of arrays, testing equality

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


def arreqclose_in_list(myarr: np.ndarray,
                       list_arrays: List[np.ndarray]):
    """Get index of array in list of arrays, testing approximate equality

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
    """Decorator similar to @property that ignores setter

    The setter shouldn't be defined, and any setting of attribute is ignored.
    """
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __get__(self, obj, value):
        return self.func(obj)

    def __set__(self, obj, value):
        pass


def freq_to_str(frequency, fmt='{:.15g}'):
    """Formats frequency, using right magnitude.

    Note:
        This will be superseded later on when all attributes are converted to
        parameters.
    """
    if frequency > 1e9:
        frequency_string = fmt.format(frequency/1e9) + ' GHz'
    elif frequency > 1e6:
        frequency_string = fmt.format(frequency/1e6) + ' MHz'
    elif frequency > 1e3:
        frequency_string = fmt.format(frequency/1e3) + ' kHz'
    else:
        frequency_string = fmt.format(frequency) + ' Hz'

    return frequency_string