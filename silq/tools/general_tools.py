import sys
import operator
from functools import partial


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
    print('{}.{} = {}'.format(obj.__class__.__name__, attr, getattr(obj, attr)))
