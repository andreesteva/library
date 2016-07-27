"""This scripts creates a train/test datasplit on the skindata4 dataset using edinburgh and ISIC as the test sets.

FUNCTIONALITY:
This script creates a train directory (train_dir, below) with subdirectories representing
training classes and symlinks pointing to the images in dataset_directory.

We remove from this training the images that have been preserved as test sets, listed in 'excluded_datasets', below.

EXPECTED RESULTS:
Using this partition, a net trained using recursive_dividing treelearning with N=1000 should achieve the following
AUC values (roughly):
melanocytic lesions: 94%
dermoscopic lesions: 91%
epidermal lesions: 96%
"""

import json
import os
import numpy as np
import argparse

import lib
from lib.taxonomy.utils import SynonymsList
from lib.notebooks.vis_utils import tic, toc
from lib.taxonomy.loading import getEntryValues, gatherSynset, gatherPathsAndLabels, rootNode, rootNodeClasses, setEntries, getEntries
from lib.taxonomy.loading import imageExists
from lib.taxonomy.loading import TRAINING_SET, TESTING_SET, NO_SET
from lib.taxonomy.edge_extraction import *
from lib.taxonomy.io import *

import scipy.sparse as sp

## USER FLAGS
#parser = argparse.ArgumentParser(description='Arguments for production partition')
#parser.add_argument('--dataset_directory', type=str)
#parser.add_argument('--meta_file', type=str, help='The meta.json to load from.')
#parser.add_argument('--train_dir', type=str, help='The training directory to create.')
#parser.add_argument('--test_dir', type=str, help='The testing directory to create.')

skin_prob = 0.4
tax_path_score = 0.8

## User settings - MJOLNIR ----------------------------------
#dataset_directory = '/ssd/esteva/skindata4/images/'
#meta_file = '/ssd/esteva/skindata4/meta.json'
#
#train_dir = '/ssd/esteva/skindata4/splits/nine-way/train'
#labels_file = '/ssd/esteva/skindata4/splits/nine-way/labels.txt'
#
## Files with entries of the form [path/to/image] [label]
## All basenames listed in excluded_datasets will be ommitted from train/val
excluded_datasets = [
        '/ssd/esteva/skindata4/test_sets/dermoscopy_test.txt',
        '/ssd/esteva/skindata4/test_sets/epidermal_test.txt',
        '/ssd/esteva/skindata4/test_sets/melanocytic_test.txt'
        ]


## User settings - LEOPARD ----------------------------------
D = '/media/esteva/ExtraDrive1/ThrunResearch/data/skindata4'
dataset_directory = os.path.join(D, 'images/')
meta_file = os.path.join(D, 'meta.json')

train_dir = os.path.join(D, 'splits/production/nine-way/train')
labels_file = os.path.join(D, 'splits/production/nine-way/labels.txt')

# Files with entries of the form [path/to/image] [label]
# All basenames listed in excluded_datasets will be ommitted from train/val
excluded_datasets = [
        os.path.join(D, 'test_sets/dermoscopy_test.txt'),
        os.path.join(D, 'test_sets/epidermal_test.txt'),
        os.path.join(D, 'test_sets/melanocytic_test.txt')
        ]


def main():

    if os.path.exists(train_dir):
        print 'Train dir %s exists, exiting' % train_dir
        return

    # We load in images that exist on our filesystem,
    meta = json.load(open(meta_file))
    meta = [m for m in meta if imageExists(m, dataset_directory)]

    # Keep isic
    isic = getEntries(meta, 'database', 'isic')
    isic = [i for i in isic if 'label' in i and i['label'] in ['benign', 'malignant']]

    # Keep meta with desired skin probs and tax path scores
    meta = [m for m in meta if 'tax_path_score' in m and m['tax_path_score'] >= tax_path_score]
    meta = [m for m in meta if m['tax_path']]
    meta = [m for m in meta if 'skin_prob' in m and m['skin_prob'] >= skin_prob]
    meta.extend(isic)

    # Fix the naming convention issues of the top 9 categories (to dermal-tumor-benign, etc.)
    syns = SynonymsList()
    for m in meta:
        rootname = '-'.join(m['tax_path'][0])
        rootrename = syns.synonymOf(rootname).split('-')
        m['tax_path'][0] = rootrename

    # Rename 'label' field to 'disease_name'. 'label' will be used for integer labels.
    for m in meta:
        if 'label' in m:
            m['disease_name'] = m['label']
            m['label'] = None

    print "Kept Meta Entries: %d" % len(meta)

    # Assign nine-way rootnode classes.
    classes, labels = rootNodeClasses(meta)
    setEntries(meta, 'label', labels)
    setEntries(meta, 'clinical_label', labels)
    setEntries(meta, 'set_identifier', TRAINING_SET)
    for k,v in classes.iteritems():
        print k,v

    # Exclude all specified datasets from any set
    filename2meta = Field2meta(meta, field='filename')
    for exclusion_file in excluded_datasets:
        exclusion_count = 0
        filenames = [os.path.basename(line.strip().split()[0]) for line in open(exclusion_file).readlines()]

        for fn in filenames:
            ms = filename2meta(fn)
            for m in ms:
                m['set_identifier'] = NO_SET
                exclusion_count += 1
        print 'Excluding %d images listed in dataset %s' % (exclusion_count, exclusion_file)

    meta_train = getEntries(meta, 'set_identifier', TRAINING_SET)
    print 'Size of meta_train %d' % len(meta_train)
    synset = gatherSynset(meta_train)

    print 'Gathering paths and labels from the metadata'
    trainset = np.unique(gatherPathsAndLabels(meta, dataset_directory, TRAINING_SET))
    testset = np.unique(gatherPathsAndLabels(meta, dataset_directory, TESTING_SET))
    no_set = np.unique(gatherPathsAndLabels(meta, dataset_directory, NO_SET))

    print_partition_statistics(meta, classes, dataset_directory)

    # Make training directory structure
    subclasses = [s.split()[1] for s in synset]
    make_directory_structure(train_dir, subclasses)

    syms_train = generate_symlinks(trainset, train_dir, subclasses)

    create_symlinks(syms_train)

    print 'Directory created: %s' % train_dir

    with open(labels_file, 'w') as f:
        prefix = ""
        for s in subclasses:
            f.write(prefix)
            f.write(s)
            prefix = "\n"
    print 'Labels file created: %s' % labels_file

    return


if __name__ == '__main__':
    main()

