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
from lib.taxonomy.loading import getEntryValues, gatherSynset, gatherPathsAndLabels, rootNode, rootNodeClasses
from lib.taxonomy.loading import setEntries, getEntries
from lib.taxonomy.loading import imageExists, Field2meta
from lib.taxonomy.loading import TRAINING_SET, TESTING_SET, NO_SET, VALIDATION_SET
from lib.taxonomy.graph_structure import Taxonomy, recursive_division
from lib.taxonomy.edge_extraction import partition_connected_components
from lib.taxonomy.io import generate_symlinks, make_directory_structure, create_symlinks
from lib.taxonomy.io import print_partition_statistics

import scipy.sparse as sp

dataset_directory = '/archive/esteva/skindata4/images/'
meta_file = '/archive/esteva/skindata4/meta.json'

train_dir = '/archive/esteva/skindata4/splits/recursive_dividing_N=1000/train'
test_dir = '/archive/esteva/skindata4/splits/recursive_dividing_N=1000/test'
labels_file = '/archive/esteva/skindata4/splits/recursive_dividing_N=1000/labels.txt'

skin_prob = 0.4
tax_path_score = 0.8
N=1000


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

    # Keep meta with desired skin probs and tax path scores
    meta = [m for m in meta if 'tax_path_score' in m and m['tax_path_score'] >= tax_path_score]
    meta = [m for m in meta if m['tax_path']]
    meta = [m for m in meta if 'skin_prob' in m and m['skin_prob'] >= skin_prob]
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

    meta_train = getEntries(meta, 'set_identifier', TRAINING_SET)
    meta_test = getEntries(meta, 'set_identifier', TESTING_SET)

    # Reassign to the training set new labels based on a recursive dividing treelearning partition.
    taxonomy = Taxonomy(meta_train)
    new_classes, new_names = recursive_division(taxonomy.top_node, N)
    sort_indices = np.argsort(new_names)
    new_classes = [new_classes[i] for i in sort_indices]
    new_names = [new_names[i] for i in sort_indices]
    for i, (new_class, new_name) in enumerate(zip(new_classes, new_names)):
        new_name = new_name.strip('/').replace('/', '_')
        for entry in new_class:
            entry['label'] = i
            entry['label_name'] = new_name
    print 'Applying TreeLearning: Recursive Dividing with N=%d' % N

    def collectSynset(meta_train):
        """Returns the synset and checks that it is sorted.

        Args:
            meta_train (list): list of dicts in skindata format. Must contain field 'label' and 'label_name'

        Returns
            Sorted list of class names in the format [label_name] [label].
        """
        synset = []
        for m in meta_train:
            synset.append([m['label_name'], m['label']])
        synset = {tuple(s) for s in synset}
        synset = [list(s) for s in synset]
        synset.sort(key=lambda x: x[0])
        synset = [[str(ss) for ss in s] for s in synset]
        synset = [" ".join(s) for s in synset]

        # run sort checks
        ss = np.sort(synset)
        for i,j in zip(ss, synset):
            assert i == j

        for i, j in zip([s.split()[1] for s in ss], [s.split()[1] for s in synset]):
            assert i == j

        return synset


    synset = collectSynset(meta_train)

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

    print 'Gathering paths and labels from the metadata'
    trainset = np.unique(gatherPathsAndLabels(meta, dataset_directory, TRAINING_SET))
    valset = np.unique(gatherPathsAndLabels(meta, dataset_directory, VALIDATION_SET))
    testset = np.unique(gatherPathsAndLabels(meta, dataset_directory, TESTING_SET))
    no_set = np.unique(gatherPathsAndLabels(meta, dataset_directory, NO_SET))

    print_partition_statistics(meta, classes, dataset_directory)

    # Make testing directory structure - rootnode classes
    subclasses = np.unique([s.split()[0].split('_')[0] for s in synset])
    make_directory_structure(test_dir, subclasses)
    syms_test = generate_symlinks(testset, test_dir, subclasses)
    create_symlinks(syms_test)


    # Make training directory structure - taxonomy training classes
    subclasses = np.unique([s.replace(' ', '_') for s in synset])
    make_directory_structure(train_dir, subclasses)
    syms_train = generate_symlinks(trainset, train_dir, subclasses)
    create_symlinks(syms_train)

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

