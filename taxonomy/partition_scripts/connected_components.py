"""This scripts creates a train/test datasplit on the skindata4 dataset using a connected components graph partition.

This script creates a train and test directory (train_dir, test_dir, below) with subdirectories representing
training classes and symlinks pointing to the images in dataset_directory.

ALGORITHM:
We treat theimages as nodes in a graph, and duplicate images are represented as edges.
We collect all known edges in the graph, and separate train/test by shifting entire connected components into either set.
The edges are from image metadata, two AMT runs and dermquest ids.

Finally, we only keep the test set images that have passed manual inspection.
    This script was previously run (as a notebook), it generated a test set, and that test set was
    curated by Andre Esteva and Rob Novoa to get rid of bad testing images (far away, blurry, etc.)

The purpose behind first generating the split algorithmically and then pruning out test set images
(as opposed to simply grabbing the curated test set images and placing everything else into train)
is that once a train/test components parition is established, images cannot be transfered between them,
only connectd components can. Thus it makes sense to re-create the partition, and then prune out test images
based on curation.

EXPECTED RESULTS:
Using this partition, a net trained using recursive_dividing treelearning with N=1000 should achieve about 55%
nine-way accuracy on the test set.
"""

import json
import os
import numpy as np

import lib
from lib.taxonomy.utils import SynonymsList
from lib.notebooks.vis_utils import tic, toc
from lib.taxonomy.loading import getEntryValues, gatherSynset, gatherPathsAndLabels, rootNode, rootNodeClasses, setEntries, getEntries
from lib.taxonomy.loading import imageExists
from lib.taxonomy.loading import TRAINING_SET, TESTING_SET, NO_SET
from lib.taxonomy.edge_extraction import *
from lib.taxonomy.io import *

import scipy.sparse as sp

# User settings - MJOLNIR ----------------------------------
dataset_directory = '/ssd/esteva/skindata4/images/'
meta_file = '/ssd/esteva/skindata4/meta.json'

train_dir = '/ssd/esteva/skindata4/splits/nine-way/train'
test_dir = '/ssd/esteva/skindata4/splits/nine-way/val'
labels_file = '/ssd/esteva/skindata4/splits/nine-way/labels.txt'

curated_test_file = '/ssd/esteva/skindata4/test_sets/validation_set.txt'

# Files with entries of the form [path/to/image] [label]
# All basenames listed in excluded_datasets will be ommitted from train/val
excluded_datasets = [
        '/ssd/esteva/skindata4/test_sets/dermoscopy_test.txt',
        '/ssd/esteva/skindata4/test_sets/epidermal_test.txt',
        '/ssd/esteva/skindata4/test_sets/melanocytic_test.txt'
        ]


# User settings - LEOPARD ----------------------------------
D = '/media/esteva/ExtraDrive1/ThrunResearch/data/skindata4'
dataset_directory = os.path.join(D, 'images/')
meta_file = os.path.join(D, 'meta.json')

train_dir = os.path.join(D, 'splits/nine-way/train')
test_dir = os.path.join(D, 'splits/nine-way/eval')
labels_file = os.path.join(D, 'splits/nine-way/labels.txt')

train_dir = os.path.join(D, 'splits/tmp-9/train')
test_dir = os.path.join(D, 'splits/tmp-9/eval')
labels_file = os.path.join(D, 'splits/tmp-9/labels.txt')

curated_test_file = os.path.join(D, 'test_sets/validation_set.txt')

# Files with entries of the form [path/to/image] [label]
# All basenames listed in excluded_datasets will be ommitted from train/val
excluded_datasets = [
        os.path.join(D, 'test_sets/dermoscopy_test.txt'),
        os.path.join(D, 'test_sets/epidermal_test.txt'),
        os.path.join(D, 'test_sets/melanocytic_test.txt')
        ]


skin_prob = 0.4
tax_path_score = 0.8



def main():

    if os.path.exists(train_dir):
        print 'Train dir %s exists, exiting' % train_dir
        return
    if os.path.exists(test_dir):
        print
        'Test dir %s exists, exiting' % test_dir
        return

    # We load in images that exist on our filesystem,
    meta = json.load(open(meta_file))
    meta = [m for m in meta if imageExists(m, dataset_directory)]

    # Connected components partition assigns one of TRAINING_SET or TESTING_SET to field 'set_identifier'
    partition_connected_components(meta)

    # Keep isic
    isic = getEntries(meta, 'database', 'isic')
    isic = [i for i in isic if 'label' in i and i['label'] in ['benign', 'malignant']]

    # Keep meta with desired skin probs and tax path scores
    meta = [m for m in meta if 'tax_path_score' in m and m['tax_path_score'] >= tax_path_score]
    meta = [m for m in meta if m['tax_path']]
    meta = [m for m in meta if 'skin_prob' in m and m['skin_prob'] >= skin_prob]
    meta.extend(isic)
    meta = [m for m in meta if m['set_identifier'] in [TRAINING_SET, TESTING_SET]]

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
    for k,v in classes.iteritems():
        print k,v

    meta_train = getEntries(meta, 'set_identifier', TRAINING_SET)
    meta_test = getEntries(meta, 'set_identifier', TESTING_SET)
    synset = gatherSynset(meta_train)

    print 'Size of meta_train %d' % len(meta_train)
    print 'Size of meta_test %d' % len(meta_test)

    # Keep only the test set entries that have passed manual curation
    print 'Keeping test set images that have been manually curated.',
    print 'Using curated test file: %s' % curated_test_file
    curated_test = [line.strip() for line in
                    open(curated_test_file).readlines()]
    curated_test = np.array([os.path.basename(t.split()[0]) for t in curated_test])

    filename2meta = Field2meta(meta_test, field='filename')
    for fn in curated_test:
        ms = filename2meta(fn)
        for m in ms:
            m['cc_keep'] = True

    for m in meta_test:
        if 'cc_keep' not in m:
            m['set_identifier'] = NO_SET

    # Exclude all specified datasets
    for exclusion_file in excluded_datasets:
        exclusion_count = 0
        filenames = [os.path.basename(line.strip().split()[0]) for line in open(exclusion_file).readlines()]

        for fn in filenames:
            ms = filename2meta(fn)
            for m in ms:
                m['set_identifier'] = NO_SET
                exclusion_count += 1
        print 'Excluding %d images listed in dataset %s' % (exclusion_count, exclusion_file)

    meta_test = getEntries(meta, 'set_identifier', TESTING_SET)
    print len(meta_test)

    print 'Gathering paths and labels from the metadata'
    trainset = np.unique(gatherPathsAndLabels(meta, dataset_directory, TRAINING_SET))
    testset = np.unique(gatherPathsAndLabels(meta, dataset_directory, TESTING_SET))
    no_set = np.unique(gatherPathsAndLabels(meta, dataset_directory, NO_SET))

    print_partition_statistics(meta, classes, dataset_directory)

    # Make training directory structure
    subclasses = [s.split()[1] for s in synset]
    make_directory_structure(train_dir, subclasses)
    make_directory_structure(test_dir, subclasses)

    syms_train = generate_symlinks(trainset, train_dir, subclasses)
    syms_test = generate_symlinks(testset, test_dir, subclasses)

    create_symlinks(syms_train)
    create_symlinks(syms_test)

    print 'Directory created: %s' % train_dir
    print 'Directory created: %s' % test_dir

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

