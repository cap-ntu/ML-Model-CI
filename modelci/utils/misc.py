import collections


def json_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = json_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def remove_dict_null(d: dict):
    """Remove `None` value in dictionary."""
    return {k: v for k, v in d.items() if v is not None}
