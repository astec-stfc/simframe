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


def merge_dicts(*dicts: dict) -> dict:
    """Combine multiple dictionaries: first dictionary overwrites keys in the second dictionary and so on"""
    final_dict = dicts[-1]
    for dic in list(reversed(dicts))[1:]:
        final_dict = merge_two_dicts(dic, final_dict)
    return final_dict
