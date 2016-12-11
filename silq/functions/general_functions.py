from functools import partial


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

def printattr(obj, attr):
    print('{}.{} = {}'.format(obj.__class__.__name__, attr, getattr(obj, attr)))