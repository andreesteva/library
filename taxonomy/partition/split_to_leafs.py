"""Given a folder with skindata images we populate and create subfolders based on the taxonomy.

Example:
    Given a directory 'inflammatory', this script creates 'inflammatory-split', a directory
    with a set of subdirectories in the form:

    rosacea_rosacea-rhinophyma-and-variants_rhinophyma
    rosacea_rosacea-rhinophyma-and-variants_rosacea_rosacea
    rosacea_rosacea-rhinophyma-and-variants_rosacea_rosacea-nose
    rosacea_rosacea-rhinophyma-and-variants_rosacea_rosacea-steroid

    Each populated with the images of 'inflammatory' that correspond to it.
    This correspondence is determined using the provided meta.json file, and the
    images' basenames.

    The meta.json is assumed to contain a dictionary where entries contain
    a field 'tax_path' denoting the taxonomy path of images.
"""

import argparse
import numpy as np
import os
import os.path as osp
import json
import shutil

import lib
from lib.taxonomy import loading


parser = argparse.ArgumentParser(description='Arguments for production partition')
parser.add_argument('--dataset_directory', type=str)
parser.add_argument('--new_dir_location', type=str)
parser.add_argument('--meta_file', type=str, help='The meta.json to load from.')
args = parser.parse_args()


def full_tax_path(entry):
    """Returns i.e. acne_of-the-leg_but-not-the-ass"""
    tax_path = entry['tax_path'][1:]
    tax_path = '_'.join(['-'.join(l) for l in tax_path])
    return tax_path


def copy(src, dst):
    """Copy src to dst, making a new symlink if src is a symlink."""
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    else:
        shutil.copy(src,dst)


def main():

    dataset_directory = args.dataset_directory
    new_dir = osp.join(args.new_dir_location,
            osp.basename(args.dataset_directory) + '-taxsplit')
    meta_file = args.meta_file
    print 'Using data directory: %s' % dataset_directory
    print 'Making data directory: %s' % new_dir

    # Read dataset_directory.
    image_names = os.listdir(dataset_directory)
    print "The dataset_directory has %d images." % len(image_names)

    # Load metadata
    meta = json.load(open(meta_file))
    meta = [m for m in meta if 'tax_path' in m]
    filename2meta = loading.Field2meta(meta, field='filename')

    # The new directory structure is determined by taxonomy
    # structure.
    new_dir_structure = {}
    bad = 0
    for filename in image_names:
        meta_entries = filename2meta(filename)
        tax_paths = [full_tax_path(m) for m in meta_entries]
        if len(tax_paths) > 1:
            bad += 1
        tax_path = tax_paths[0]
        if tax_path in new_dir_structure:
            new_dir_structure[tax_path].append(filename)
        else:
            new_dir_structure[tax_path] = [filename]

    # Make new directory, deleting the previous if it exists.
    if osp.exists(new_dir):
        print 'Deleting previous'
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

    for new_class in new_dir_structure:
        print 'Making: (%d images) %s' %(len(new_dir_structure[new_class]),
                                         new_class)
        subdir = osp.join(new_dir, new_class)
        os.makedirs(subdir)
        for filename in new_dir_structure[new_class]:
            src = osp.join(dataset_directory, filename)
            dst = osp.join(subdir, filename)
            copy(src, dst)


if __name__ == '__main__':
    main()
