from copy import copy


def merge_two_dicts(y, x):
    """Combine to dictionaries: first dictionary overwrites keys in the second dictionary"""
    if not isinstance(x, (dict, dict)) and not isinstance(y, (dict, dict)):
        return dict()
    elif not isinstance(x, (dict, dict)):
        return y
    elif not isinstance(y, (dict, dict)):
        return x
    else:
        z = x.copy()  # start with x's keys and values
        z.update(y)  # modifies z with y's keys and values & returns None
        return z
