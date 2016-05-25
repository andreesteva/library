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

import scipy.sparse as sp

dataset_directory = '/archive/esteva/skindata4/images/'
meta_file = '/archive/esteva/skindata4/meta.json'

train_dir = '/archive/esteva/skindata4/splits/nine-way/train'
test_dir = '/archive/esteva/skindata4/splits/nine-way/test'
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

    def component_is_split(comp):
        """Returns true if this component is split between train and test"""
        set_ids = set([m['set_identifier'] for m in comp if m['set_identifier'] != NO_SET])
        return len(set_ids) == 2


    def random_partition(meta_test, return_splits=False):
        """Randomly partitions the components of meta_test to train/test by flipping a coin."""
        split_comps = []
        for m in meta_test:
            comp = cc2meta(m['connected_component'])
            if component_is_split(comp):
                if np.random.rand() < 0.5:
                    setEntries(comp, 'set_identifier', TRAINING_SET)
                else:
                    setEntries(comp, 'set_identifier', TESTING_SET)
                split_comps.append(comp)
        if return_splits:
            return split_comps

    def maxset_partition(meta_test):
        """Deterministically place component into whichever set they already have more images of."""
        for m in meta_test:
            comp = cc2meta(m['connected_component'])
            if component_is_split(comp):
                N_test = len(getEntries(comp, 'set_identifier', TESTING_SET))
                N_train = len(getEntries(comp, 'set_identifier', TRAINING_SET))
                if N_test >= N_train:
                    setEntries(comp, 'set_identifier', TESTING_SET)
                else:
                    setEntries(comp, 'set_identifier', TRAINING_SET)


    def make_directory_structure(dir_name, subfolders):
        """Creates directory with subdirectories.

        dirname/subfolder1
        dirname/subfolder2
        dirname/subfolder3
        ...

        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for s in subfolders:
            p = os.path.join(dir_name, s)
            os.makedirs(p)


    def generate_symlinks(dataset, dirname, subclasses):
        """Creates symbolic link src-dst pairs from dirname/subclasses into the entries of dataset.

        Args:
            dataset (list): list of entries of the form 'path/to/an/image 1', where 1 is the synset subclass label
            dirname (str): the directory to place links into
            subclasses (list): list of strings of subdirectories

        Returns:
            A list in the form:
             u'/archive/esteva/skindata4/images/--7StukLcauw-M.jpg /archive/nine-way/train/dermal-tumor-malignant/--7StukLcauw-M.jpg',
             u'/archive/esteva/skindata4/images/---5qm8LHUUxCM.jpg /archive/nine-way/train/dermal-tumor-benign/---5qm8LHUUxCM.jpg',
             u'/archive/esteva/skindata4/images/--9OaBvO9URaDM.jpg /archive/nine-way/train/inflammatory/--9OaBvO9URaDM.jpg',
             u'/archive/esteva/skindata4/images/-0GnmpIR2i8V5M.jpg /archive/nine-way/train/inflammatory/-0GnmpIR2i8V5M.jpg',
             u'/archive/esteva/skindata4/images/-0cnZs3CfFFssM.jpg /archive/nine-way/train/inflammatory/-0cnZs3CfFFssM.jpg',
        """
        syms = []
        for entry in dataset:
            p = entry.split()[0]
            l = int(entry.split()[1])
            s = " ".join([p, os.path.join(dirname, subclasses[l], os.path.basename(p))])
            syms.append(s)
        return syms


    def create_symlinks(symlinks):
        """Creates symbolic links on the filesystem given src-dst pairs.

        Creates a symbolic link symlinks[i].split()[1] pointing to symlinks[i].split()[0] for all i.

        Args:
            symlinks(list): list of strings in the form given by generate_symlinks, above
        """
        for entry in symlinks:
            src = entry.split()[0]
            dst = entry.split()[1]
            os.symlink(src, dst)


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


    # Calculate time-camera edge matrix as a sparse matrix (about 2 minutes)
    meta = meta_exists
    for i,m in enumerate(meta):
        m['index'] = i

    insert_datetime_field(meta)
    insert_abs_time_field(meta)

    meta_datetime = getEntries(meta, 'datetime', None)
    cameras, camera_models = extract_cameras(meta_datetime)
    abs_times = np.array([m['abs_time'] for m in meta_datetime]).reshape((-1,1))
    cam_indx = np.array([camera_models[c] for c in cameras]).reshape((-1,1))
    print '%d Entries have the datetime metadata' % len(meta_datetime)

    print 'Calculating time-camera edge matrix...'
    M = len(meta_datetime)
    N = len(meta)

    t = tic()
    edge, _, _= edge_matrix(abs_times, cam_indx)
    edge = squareform_sparse(edge, M)
    toc(t)

    # Initialize the N x N Edge matrix
    E = sp.lil_matrix((N,N), dtype=bool)

    # Insert the datetime edges
    c = 0
    for i,j,v in sparse_matrix_iterator(edge):
        if v:
            c += 1
            idx_i = meta_datetime[i]['index']
            idx_j = meta_datetime[j]['index']
            E[idx_i, idx_j] = v
    print 'Adding %d edges to the graph' % c

    # Add into the edge matrix the duplicates of turk2 and turk 1
    turk_results = [
        '/archive/esteva/skindata4/duplicate_urls_turk1.json',
        '/archive/esteva/skindata4/duplicate_urls_turk2.json',
        ]

    def dict2list(dict_):
        list_ = []
        for key, value in dict_.iteritems():
            d = [key]
            d.extend([v for v in value])
            list_.append(d)
        return list_

    for tr in turk_results:
        turk = json.load(open(tr, 'r'))
        duplicates = dict2list(turk)
        insert_edges_into_edge_matrix(E, duplicates, meta, field_name='filename')
        print 'Adding %d turk edges to the graph' % np.sum([len(v)-1 for v in duplicates])

    # Add dermquest ids into the graph.
    dermquest = getEntries(meta, 'database', 'dermquest')
    dermquest = getEntries(dermquest, 'case', None)
    case2meta = Field2meta(dermquest, field='case')
    cases = np.unique(getEntryValues(dermquest, 'case'))
    duplicates = [[m['index'] for m in case2meta(case)] for case in cases]
    insert_edges_into_edge_matrix(E, duplicates, meta, field_name='index')
    print 'Adding %d dermquest edges to the graph' % np.sum([len(v)-1 for v in duplicates])

    # Add meta entries that share the same filenames as edges
    filename2meta = Field2meta(meta, field='filename')
    filenames = np.unique([m['filename'] for m in meta])
    duplicates = []
    for fn in filenames:
        meta_filename = filename2meta(fn)
        if len(meta_filename) == 0:
            print 'wtf'
            break
        if len(meta_filename) > 1:
            duplicates.append([m['index'] for m in meta_filename])
    insert_edges_into_edge_matrix(E, duplicates, meta, field_name='index')
    print 'Adding %d edges to the graph based on identical filenames' % np.sum([len(v)-1 for v in duplicates])

    # Extract connected components and assign them to the meta
    n_components, connected_components = sp.csgraph.connected_components(E, directed=False)
    unique_component_numbers, component_sizes = np.unique(connected_components, return_counts=True)

    for m, c in zip(meta, connected_components):
        m['connected_component'] = c
    print 'We find %d connected components' % n_components

    # Propose a test set (from the turked set)
    test_set = '/archive/esteva/skindata4/duplicate_urls_turk2.json'
    print 'Proposing test set from %s' % test_set
    test_set = json.load(open(test_set, 'r'))
    test_set = [key for key in test_set.keys()]
    filename2meta = Field2meta(meta, field='filename')
    cc2meta = Field2meta(meta, field='connected_component')
    meta_test = [m for fn in test_set for m in filename2meta(fn)]
    setEntries(meta, 'set_identifier', TRAINING_SET)
    setEntries(meta_test, 'set_identifier', TESTING_SET)
    print 'Proposed Test Set has %d entries' % len(meta_test)

    # Iterate over elements of the test set and push connected components to train or test
    maxset_partition(meta_test)
    meta_test = getEntries(meta, 'set_identifier', TESTING_SET)
    print 'Partitioned Test Set has %d meta entries' % len(meta_test)

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

