"""Library for read-write functions with the filesystem."""
import os
import numpy as np
import unittest
import json

import lib
from lib.taxonomy.loading import gatherPathsAndLabels, rootNodeClasses, getEntries, imageExists
from lib.taxonomy.loading import TRAINING_SET, TESTING_SET, NO_SET, VALIDATION_SET


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
        print 'Linking: %s to %s' % (src, dst)
        os.symlink(src, dst)


def print_partition_statistics(meta, classes, dataset_directory):
    """Print out statistics on the dataset."""

    trainset = np.unique(gatherPathsAndLabels(meta, dataset_directory, TRAINING_SET))
    valset = np.unique(gatherPathsAndLabels(meta, dataset_directory, VALIDATION_SET))
    testset = np.unique(gatherPathsAndLabels(meta, dataset_directory, TESTING_SET))

    # Lets check that there is no overlap between train and test paths
    trainpaths = np.unique([os.path.basename(t.split()[0]) for t in trainset])
    valpaths = np.unique([os.path.basename(t.split()[0]) for t in valset])
    testpaths = np.unique([os.path.basename(t.split()[0]) for t in testset])

    intersection = np.intersect1d(trainpaths, testpaths)
    print 'Train and test share %d images, according to filenames' % len(intersection)
    intersection = np.intersect1d(trainpaths, valpaths)
    print 'Train and val share %d images, according to filenames' % len(intersection)
    intersection = np.intersect1d(testpaths, valpaths)
    print 'Test and val share %d images, according to filenames' % len(intersection)

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
    print ''

class TestMethods(unittest.TestCase):

    def test_print_partition_statistics(self):
        meta = json.load(open('/archive/esteva/skindata4/meta_lite.json'))
        classes = range(9)
        for m in meta:
            c = np.random.choice(classes)
            m['clinical_label'] = c
            m['label'] = c
        dataset_directory = '/archive/esteva/skindata4/images'
        meta = [m for m in meta if imageExists(m, dataset_directory)]
        l = len(meta)
        for i,m in enumerate(meta):
            if i < l / 3:
                m['set_identifier'] = TRAINING_SET
                continue
            if i < 2 * l / 3:
                m['set_identifier'] = VALIDATION_SET
                continue
            m['set_identifier'] = TESTING_SET
        print_partition_statistics(meta, classes, dataset_directory)

if __name__ == '__main__':
    # Run unit test
    unittest.main()



