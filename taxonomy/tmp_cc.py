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
from lib.taxonomy.loading import getEntryValues, gatherSynset, gatherPathsAndLabels, rootNode, rootNodeClasses, setEntries
from lib.taxonomy.loading import imageExists
from lib.taxonomy.loading import TRAINING_SET, TESTING_SET, NO_SET, VALIDATION_SET
from lib.taxonomy.edge_extraction import *
from lib.taxonomy.io import *

import scipy.sparse as sp

dataset_directory = '/archive/esteva/skindata4/images/'
meta_file = '/archive/esteva/skindata4/meta.json'

train_dir = '/archive/esteva/skindata4/splits/nine-way/train'
test_dir = '/archive/esteva/skindata4/splits/nine-way/test'
train_dir = '/archive/esteva/skindata4/splits/tmp/train'
test_dir = '/archive/esteva/skindata4/splits/tmp/test'
labels_file = '/archive/esteva/skindata4/splits/nine-way/labels.txt'

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
    # Fix the naming convention issues of the top 9 categories, and
    # Rename 'label' field to 'disease_name'
    meta = json.load(open(meta_file))
    meta = [m for m in meta if imageExists(m, dataset_directory)]
    meta_exists = meta
    meta = [m for m in meta if 'tax_path_score' in m.keys() and m['tax_path_score'] >= tax_path_score]
    meta = [m for m in meta if m['tax_path']]
    meta = [m for m in meta if 'skin_prob' in m.keys() and m['skin_prob'] >= skin_prob]

    syns = SynonymsList()
    for m in meta:
        rootname = '-'.join(m['tax_path'][0])
        rootrename = syns.synonymOf(rootname).split('-')
        m['tax_path'][0] = rootrename

    for m in meta:
        if 'label' in m:
            m['disease_name'] = m['label']
            m['label'] = None

    print "Kept Meta Entries: %d" % len(meta)


    # These are the classes we will use
    classes, labels = rootNodeClasses(meta)
    setEntries(meta, 'label', labels)
    setEntries(meta, 'clinical_label', labels)
    for k,v in classes.iteritems():
        print k,v

    partition_connected_components(meta_exists)

    # Assign NO_SET to meta entries that dont make the threshold, and reduce meta to just those that do.
    meta = meta_exists
    print 'Eliminating images with skin_prob < %0.2f and tax_path_score < %0.2f' % (skin_prob, tax_path_score)
    pruned = 0
    for m in meta:
        if 'skin_prob' not in m or 'tax_path_score' not in m or 'tax_path' not in m:
            m['set_identifier'] = NO_SET
            pruned += 1
            continue
        if m['skin_prob'] < skin_prob or m['tax_path_score'] < tax_path_score:
            m['set_identifier'] = NO_SET
            pruned += 1

    meta = [m for m in meta if m['set_identifier'] in [TRAINING_SET, TESTING_SET]]
    classes, labels = rootNodeClasses(meta)
    setEntries(meta, 'label', labels)
    setEntries(meta, 'clinical_label', labels)

    meta_train = getEntries(meta, 'set_identifier', TRAINING_SET)
    meta_test = getEntries(meta, 'set_identifier', TESTING_SET)
    synset = gatherSynset(meta_train)

    print 'Pruning out %d / %d meta entries to NO_SET and assigning labels and clinical labels' % (pruned, len(meta_exists))
    print 'Size of meta_train %d' % len(meta_train)
    print 'Size of meta_test %d' % len(meta_test)

    # Keep only the test set entries that have passed manual curation
    curated_test_file = '/archive/esteva/skindata4/splits/test_curated.txt'
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

    meta_test = getEntries(meta, 'set_identifier', TESTING_SET)
    print len(meta_test)

    # Print out statistics on the dataset
    print 'Generating Training, Validation, and Testing sets.'
    trainset = gatherPathsAndLabels(meta, dataset_directory, TRAINING_SET)
    valset = gatherPathsAndLabels(meta, dataset_directory, VALIDATION_SET)
    testset = gatherPathsAndLabels(meta, dataset_directory, TESTING_SET)
    no_set = gatherPathsAndLabels(meta, dataset_directory, NO_SET)

    # Since some images have multiple diseases, we keep only the unique 'path [label]' entries
    trainset = np.unique(trainset)
    valset = np.unique(valset)
    testset = np.unique(testset)
    noset = np.unique(no_set)

    # Lets check that there is no overlap between train and test paths
    trainpaths = np.unique([os.path.basename(t.split()[0]) for t in trainset])
    testpaths = np.unique([os.path.basename(t.split()[0]) for t in testset])
    intersection = np.intersect1d(trainpaths, testpaths)
    print 'Train and test share %d images, according to filenames' % len(intersection)

    getClassFromValidationSet = lambda meta, c: [m for m in meta if m['set_identifier'] == VALIDATION_SET and m['clinical_label'] == c]
    getClassFromTrainingSet = lambda meta, c: [m for m in meta if m['set_identifier'] == TRAINING_SET and m['clinical_label'] == c]
    getClassFromTestingSet = lambda meta, c: [m for m in meta if m['set_identifier'] == TESTING_SET and m['clinical_label'] == c]
    print 'Dataset sizes (Based on Metadata):'
    print 'Train,\tVal,\tTest,\tTotal'
    for c in classes:
        v = len(getClassFromValidationSet(meta, c))
        t = len(getClassFromTrainingSet(meta, c))
        te = len(getClassFromTestingSet(meta, c))
        print t, '\t', v, '\t', te, '\t', v + t + te

    print ''
    print len(getEntries(meta, 'set_identifier', TRAINING_SET)),
    print len(getEntries(meta, 'set_identifier', VALIDATION_SET)),
    print len(getEntries(meta, 'set_identifier', TESTING_SET))
    print ''

    print 'Dataset sizes (Based on unique images):'
    print 'Train,\tVal,\tTest,\tTotal'
    for c in classes:
        v = len(np.unique([m['filename'] for m in getClassFromValidationSet(meta, c)]))
        t = len(np.unique([m['filename'] for m in getClassFromTrainingSet(meta, c)]))
        te = len(np.unique([m['filename'] for m in getClassFromTestingSet(meta, c)]))
        print t, '\t', v, '\t', te, '\t', v + t + te

    print '# Unique Images in Training:', len(trainset)
    print '# Unique Images in Validation:', len(valset)
    print '# Unique Images in Testing:', len(testset)
    print '# Unique Images tossed out:', len(noset)
    print ''

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

    return


if __name__ == '__main__':
    main()

