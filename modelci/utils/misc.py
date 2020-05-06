import collections


def json_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = json_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
