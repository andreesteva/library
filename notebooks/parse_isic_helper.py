"""Helper functions for ThrunResearch/data/isic2/parse_isic.ipynb, which is a setup for dealing with the unexpectedly tedious
ISIC dataset. They have duplicates, they have data on single lesions over time, its annoying."""

import numpy as np
import json
import os
import os.path as osp
import glob


def finditem(obj, key):
    """Recurses a dictionary obj for all keys matching 'key', and returns the items in a list.
    If 'key' is a list, it does so recursively.
    """
    _found = []
    def _finditem(obj, key):
        if key in obj or unicode(key) in obj:
            _found.append(obj[key])
        for k, v in obj.items():
            if isinstance(v,dict):
                _finditem(v, key)


    def _recursive_get(obj, keys):
        if len(keys) == 1:
            return obj[keys[0]]
        return _recursive_get(obj[keys[0]], keys[1:])


    def _finditem_hierarchy(obj, key):
        # look for the first one
        if key[0] in obj or unicode(key[0]) in obj:
            item = _recursive_get(obj, key)
            _found.append(item)

        for k, v in obj.items():
            if isinstance(v, dict):
                _finditem_hierarchy(v, key)


    if isinstance(key, list):
        _finditem_hierarchy(obj, key)
    else:
        _finditem(obj, key)

    return _found


def finditem_by_expression(obj, subvalue):
    """Recurses a dictionary obj and collects key:value pairs for any values that contain subvalue."""
    _found = []
    def _finditem_by_expression(obj, subvalue):
        for k, v in obj.items():
            if isinstance(v, dict):
                _finditem_by_expression(v, subvalue)
            elif isinstance(v, unicode) or isinstance(v, str):
                if subvalue in v.lower():
                    _found.append(k + ' : ' + v)
    _finditem_by_expression(obj, subvalue)
    return _found


def get_diagnosis(meta):
    """Returns the benign/malignant diagnosis of a meta entry, or None.

    Args:
        meta(dict): the ISIC metadata

    Returns:
        A string corresponding to the diagnosis, or None, if none was found.

    """
    bm = finditem(meta, 'ben_mal')
    def _set(bm):
        if not bm:
            return None
        else:
            return bm[0]
    if not bm:
        bm = finditem(meta, 'benign_malignant')
    if not bm:
        if 'sonic' in meta['path'].lower():
            bm = ['benign']
    bm = _set(bm)
    return bm


def get_image(folder):
    """Returns ISIC id jpg image. All the folders have only 1 jpg"""
    ims = glob.glob(osp.join(folder, '*.jpg'))
    assert len(ims) == 1, 'error, multiple jps in folder %s' % folder
    return ims[0]


def get_image_p1a(folder):
    """Returns ISIC id p1a image, corresponding to the close crop of the image. Removes *tile*p1a* instances. """
    ims = glob.glob(osp.join(folder, '*p1a.png'))
    ims = [im for im in ims if 'tile' not in im]
    assert len(ims) == 1, 'error, multiple jps in folder %s' % folder
    return ims[0]


def get_meta(folder):
    """Returns ISIC id json as a dict.
        If there are multiple entries, we return the first one containing a diagnosis.
        (SONIC has multiple jsons in it)
        We also assert that if there are multiple meta entries, then they all agree
        On the diagnosis, unless their diagnosis is 'None'.
    """
    j = glob.glob(osp.join(folder, '*.json'))
    metas = [json.load(open(jj, 'r')) for jj in j]
    meta_to_return = metas[0]
    if len(metas) > 1:
        d = get_diagnosis(metas[0])
        for m in metas[-1::-1]:
            dm = get_diagnosis(m)
            if dm is None:
                continue
            if d is None:
                d = dm
                meta_to_return = m
            else:
                assert d == dm, '%s, %s' % (d, dm)
    return meta_to_return


def idname(path):
    """Returns ISIC_XXXXXX from the given path."""
    if isinstance(path, list):
        return [osp.splitext(osp.basename(p))[0] for p in path]
    return osp.splitext(osp.basename(path))[0]


def print_meta(meta):
    """Prints the metadata like you would a taxonomy, but just for the keys."""
    level = 0
    def _tabs(level):
        return "".join(level * ['\t'])
    def _print_meta(meta, level):
        for k, v in meta.iteritems():

            if k in ['_id', 'baseParentId', 'originalFilename', 'lowerName']:
                print _tabs(level), k, v
            else:
                print _tabs(level), k
            if isinstance(v, dict):
                _print_meta(v, level+1)
    _print_meta(meta, level)


def collect_by_key(isic_ids, KEY):
    """Finds groups of entries in isic_ids that have the same value for KEY"""

    meta = [get_meta(i) for i in isic_ids]
    values = [finditem(m, KEY) for m in meta]
    for v in values:
        assert len(v) == 1, v
    values = [v[0] for v in values]

    collections = []
    for i, v1 in enumerate(values):
        if i % 100 == 0:
            print '\r', i, '/', len(values),
        collect = [i]
        for j, v2 in enumerate(values):
            if i == j:
                continue
            if v1 == v2 and v1 is not None:
                collect.append(j)

        if len(collect) > 1:
            collections.append(collect)

    id_collections = [[isic_ids[i] for i in group] for group in collections]
    print ''
    return id_collections, collections, values
