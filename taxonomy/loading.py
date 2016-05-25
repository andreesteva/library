"""A Library of functions for parsing raw datasets of images into
    useful formats for Caffe-based CNN training.

    Created on 11/02/2015 by Andre Esteva
"""
import numpy as np
import os
import sys
#import pandas
import lib

""" 'skindata' format. This format is assumed in many functions below.

   {u'database': u'dermaamin',
    u'file_name': u'httpwwwdermaamincomsiteimagesclinicalpicaaxillarygranularparakeratosisaxillarygranularparakeratosis1jpg.jpg',
    u'label': u'Axillary granular parakeratosis',
    u'link': u'http://www.dermaamin.com/site/images/clinical-pic/a/axillary_granular_parakeratosis/axillary_granular_parakeratosis1.jpg',
    u'tax_path': [[u'inflammatory'],
     [u'psoriasis',
      u'pityriasis',
      u'rubra',
      u'pilaris',
      u'and',
      u'papulosquamous',
      u'disorders'],
     [u'axillary', u'granular', u'parakeratosis'],
     [u'axillary', u'granular', u'parakeratosis']],
    u'tax_path_score': 1.0}

"""

NO_SET = -1
TRAINING_SET = 0
VALIDATION_SET = 1
TESTING_SET = 2

def taxonomyPath(data_point):
    """Returns the taxonomy path of data point"""
    return node(data_point)


def imageExists(data_point, directory):
    """Returns true if the image pointed to by data_point exists on the filesystem.

        Args:
            data_point(dict): A dictionary in 'skindata' format.
            directory (str): A path to the directory where we check if the image exists.

        Returns:
            True if the image exists, false otherwise.
    """
    p = os.path.join(directory, data_point['filename'])
    return os.path.exists(p)


def rootNode(data_point):
    """Return the root node in the taxonomy for a data point.

        Args:
            data_point(dict): a dict in skindata format.
        Returns:
            The name of the root node.
    """
    return '-'.join(data_point['tax_path'][0])


def skin6class(data_point):
    """Returns the class label under one of the categories below.

         cutaneous-lymphoma
         inflammatory-and-genetic-conditions
         non-pigmented-tumor-benign
         non-pigmented-tumor-malignant
         pigmented-lesion-benign
         pigmented-lesion-malignant
    """
    rn = rootNode(data_point).lower()
    has = utils.containsAll
    if rn == 'inflammatory' or 'genodermatoses' in rn or 'genodermatosis' in rn:
        return 'inflammatory-and-genetic-conditions'
    elif has(rn, ['epidermal', 'benign']) or has(rn, ['dermal', 'benign']):
        return 'non-pigmented-tumor-benign'
    elif has(rn, ['epidermal', 'malignant']) or has(rn, ['dermal', 'malignant']):
        return 'non-pigmented-tumor-malignant'
    else:
        return rn


def node(data_point):
    """Return the node in the taxonomy for a data point."""
    j = lambda j: '-'.join(j)
    return '/'.join([j(d) for d in data_point['tax_path']])


def nodeAtDepth(entry, depth):
    """Returns the node name at the specified depth (or its first ancestor) for the given entry.

        Args:
            entry (dict): a skindata formatted dictionary entry
            depth (int): the depth down the taxonomy to descend. 0 is root

        Returns:
            the name of the node at the depth, if it exists, or its first ancestor (i.e. parent)
    """
    if depth < len(entry['tax_path']):
        return '-'.join(entry['tax_path'][depth])
    else:
        return '-'.join(entry['tax_path'][-1])


#def imagePath(data_point, images_dir='images'):
#    """Returns the relative file-system path of data_point.
#
#        Args:
#            data_point: a dictionary in skindata format.
#            images_dir: the directory in /home/esteva/ThrunResearchReserves/skindata3 that contains the data we want to load
#                usually 'images'. Can also be 'images_224x224' or the equivalent if we want faster load times.
#
#        Returns:
#            the file-system path
#    """
#    return os.path.join(images_dir, data_point['file_name'])


def nodes_counts(meta):
    """Returns the nodes in meta, together with the number of images each contains.

        Args:
            meta(list): a list of dictionaries in 'skindata' format.

        Returns:
            A list ofeach node that has at least 1 image in it, and a list of the number of images each has.
    """
    nodes = [node(m) for m in meta]
    nodes, counts = np.unique(nodes, return_counts=True)
    return nodes, counts


def rootNodeClasses(meta):
    """Creates labels for our dataset based on the root nodes (the 9).

        Args:
            meta(list): a list of dictionaries in 'skindata' format.

        Returns:
            A dictionary of [label : class_name] key-value pairs sorted alphabetically.
            A list of integer labels for each entry in meta.
    """
    root_nodes = {rootNode(m) for m in meta}
    root_nodes = np.sort([r for r in root_nodes])
    root_nodes = {r : i for i,r in enumerate(root_nodes)}
    labels = [root_nodes[rootNode(entry)] for entry in meta]
    classes = {v : k for k,v in root_nodes.iteritems()}
    return classes, np.array(labels)


def skin6Classes(meta):
    """Create labels for dataset based on lumping root nodes into 6 classes.

        The 6 resultant classes are:
            cutaneous-lymphoma
            inflammatory-and-genetic-conditions
            non-pigmented-tumor-benign
            non-pigmented-tumor-malignant
            pigmented-lesion-benign
            pigmented-lesion-malignant

        These have been formed by doing the following mergers from the original 9:
            inflammatory + genodermatosis => inflammatory-and-genetic-conditions
            dermal-tumor-benign + epidermal-tumor-benign => non-pigmented-tumor-benign
            dermal-tumor-malignant + epidermal-tumor-malignant => non-pigmented-tumor-malignant

        Args:
            meta (list): a list of dictionaries in 'skindata' format

        Returns:
            A dictionary of [label : class_name] key-value pairs sorted alphabetically.
            A list of integer labels for each entry in meta.
    """
    root_nodes = {skin6class(m) for m in meta}
    root_nodes = np.sort([r for r in root_nodes])
    root_nodes = {r : i for i,r in enumerate(root_nodes)}
    labels = [root_nodes[skin6class(entry)] for entry in meta]
    classes = {v : k for k,v in root_nodes.iteritems()}
    return classes, np.array(labels)


def setEntries(meta, key, values):
    """Set the key in each entry of meta to its corresponding entry in the list value

        Args:
            meta (list): A list of dictionaries in 'skindata' format.
            key (string): the key to add/alter in each dictionary of meta.
            values (iterable, or single value): the iterable of values to add. len(values) == len(meta).
                If values is not iterable (i.e. a single value) then we set all entries of meta[key] to this value.
    """
    # If Values is not iterable
    if not hasattr(values, '__iter__'):
        values = len(meta) * [values]

    assert(len(meta) == len(values))
    for m, v in zip(meta, values):
        m[key] = v


def getEntries(meta, key, values):
    """Get the entries of meta such that the key is either equal to or contained in values.

    Args:
        meta (list): list of dictionaries in skindata format
        key (string): the key to the dictionaries
        values (iterable, or basic type): If iterable, we expect a list, dictionary, or set. If
            single value, we expect a string, bool, int, or other basic type. If None, we return
            any meta entries that have the specified key.
    """
    md = []
    if values is None:
        for m in meta:
            if key in m.keys():
                md.append(m)
        return md

    if not isinstance(values, (list, dict, set)):
        values = [values]
    for m in meta:
        if key in m.keys() and m[key] in values:
            md.append(m)
    return md


def getEntryValues(meta, key):
    """Return all values from meta that correspond to the key, if the key exists in the entry.

    Args:
        meta (list): a list of dictioanries in 'skindata' format
        key (str): the dictionary key.
    """
    return [m[key] for m in meta if key in m.keys()]


def cluster_integers(integers, m):
    """Clusters a list of integers until each cluster's net sum is at least of size m.

        This iterates over the integers from smallest to largest, in order to maximize
        the number of clusters created.

        Args:
            integers (list): a list of integers
            m (int): the minimum cluster size

        Returns:
            a numpy.array of length len(integers) that specifies which cluster each integer belongs to.
    """
    indices = np.argsort(integers)

    clusters = np.zeros(len(integers), dtype=int)
    cluster_index = 0
    cluster_size = 0
    for i in indices:
        integer = integers[i]
        clusters[i] = cluster_index
        cluster_size += integer
        if cluster_size >= m:
            cluster_size = 0
            cluster_index += 1
    return clusters


def sizePartition(sizes, chunk_sizes):
    """ Randomly grabs elements from the list of of ints 'sizes' until their is >= chunk_size.

    Given a list of integers 'sizes', and a list of integers 'chunk_sizes', this function returns
    a list L of sublists [{SL}], one for each element of 'chunk_sizes', that each index into sizes.
    The total count over the elements of sizes that are indexed into by each SL are >= the
    corresponding element in 'chunk_sizes'. That is sum(sizes[L[i]]) >= chunk_sizes[i] for all i.
    Additionally, L[i] and L[j] share no indices: intersect(L[i], L[j]) = <null>

    Args:
        sizes (list): A list of non-negative integers.
        chunk_sizes: A list of non-negative integers.

    Returns:
        A list of sublists where each sublists denotes indices into sizes.

    """
    L = []
    sizes = sizes.copy()
    for chunk_size in chunk_sizes:

        # Roll the dice and select a random ordering of the sizes to loop over.
        permutation = np.random.permutation(len(sizes))
        current_size = 0
        keep_indices = []

        # Inspect the elements of 'sizes', in the rolled order.
        for i in permutation:
            # If our sublist is big enough, we're done.
            if current_size >= chunk_size:
                break

            # If that element is available, we add it to this sublist and make it unavailable for further sublists.
            if sizes[i] > -1:
                keep_indices.append(i)
                current_size += sizes[i]
                sizes[i] = -1

        # Store our sublist
        L.append(keep_indices)

    return L


def leafNodeTraining(meta, classes,
                     numImagesPerClassForValidation = 100,
                     numImagesPerClassForTesting = 0):
    """Partition 'meta' into train/val/test based on leaf nodes.

        Updates meta such that meta[i]['set_identifier'] identifies if the data point belongs to
        the training, validation, or testing set.

        Any entries that aren't in any one of the classes are labeled as NO_SET.

        This sets up leaf node training. Meaning that no images in any given node will ever be
        split between training, testing, or validation. We use the labels defined at the top
        of this file.
        Therefore, if meta[i]['set_identifier'] == TRAINING_SET, this is a data point to be used in the training set.

        Args:
            meta (list): A list of dictionaries in 'skindata' format.
            classes (dictionary): A dictionary of label:class_name pairs
            numImagesPerClassForValidation (int): Self-explanatory
            numImagesPerClassForTesting (int): Self-explanatory
    """
    setEntries(meta, 'set_identifier', NO_SET)

    for label in classes:
        class_meta = getEntries(meta, 'label', label)
        class_nodes, node_amounts = nodes_counts(class_meta)
        val_indices, test_indices = sizePartition(node_amounts, [numImagesPerClassForValidation,
                                                                   numImagesPerClassForTesting])
        val_nodes = class_nodes[val_indices]
        test_nodes = class_nodes[test_indices]
        for m in class_meta:
            if node(m) in val_nodes:
                m['set_identifier'] = VALIDATION_SET
            elif node(m) in test_nodes:
                m['set_identifier'] = TESTING_SET
            else:
                m['set_identifier'] = TRAINING_SET


def gatherSynset(meta_train):
    """Returns the synset of meta_train.

    Examples:
    '0 cutaneous-lymphoma 0',
    '1 dermal-tumor-benign 1',
    '2 dermal-tumor-benign 1',
    '3 dermal-tumor-benign 1',

    Args:
        meta_train (list): A list of dicts in skindata format.

    Returns:
        A list of strings in the format '[label] [class-name] [clinical-label]'
    """
    synset = []
    for m in meta_train:
        synset.append([ int(m["label"]), rootNode(m), m["clinical_label"]])
    synset = {tuple(s) for s in synset}
    synset = [list(s) for s in synset]
    synset.sort(key=lambda x: x[2])
    synset.sort(key=lambda x: x[0])
    synset = [[str(ss) for ss in s] for s in synset]
    synset = [" ".join(s) for s in synset]
    return synset


def gatherPathsAndLabels(meta, directory, set_identifier):
    """Return the list of file-system paths and their labels from meta that match set_identifier.

        Args:
            meta (list): A list of dictionaries in 'skindata' format.
            directory (string): A prefix to add to each path. This is where the data is stored.
            set_identifier (int): This specifies which entries to gather (typically train/test/val)

        Returns:
            A list of strings in 'path/to/image [label]' format.
    """
    set_ = getEntries(meta, 'set_identifier', set_identifier)
    p = []
    for s in set_:
        p.append(os.path.join(directory, s['filename']) + ' ' + str(s['label']))
    return p


def getRandomNode(meta):
    """Returns all entries of meta that share a randomly selected node."""
    nodes = np.unique([node(m) for m in meta])
    leafnode = nodes[np.random.randint(0, len(nodes))]
    return [m for m in meta if node(m) == leafnode]


def dermQuestIDTraining(meta, classes, numImagesPerClassForValidation=200, database_exclusions=[None]):
    """Creates a validation set based on DermQuest Patient IDs, supplementing with leaf nodes.

        This function sets the meta[i]['set_identifiers'] property to one of:
            VALIDATION_SET
            TRAINING_SET
        Anything that isn't labeled as VALIDATION_SET is labeled as TRAINING_SET.
        Any entries that aren't in any one of the classes are labeled as NO_SET.

        Args:
            meta (list): A list of dictionaries in 'skindata' format
            classes (dictionary): A dictionary of label:class_name pairs
            numImagesPerClassForValidation (int): Self-explanatory. If None, then we use half
                of all the dermquest images for that class.
            database_exclusions (list): A list of strings.
                The databases to exclude in leafnode supplementation.
    """
    setEntries(meta, 'set_identifier', NO_SET)

    # Add dermquest to validation set based on unique case ids.
    dermquest = getEntries(meta, 'database', 'dermquest')
    for c in classes:
        print '\rClass', c,
        class_ = getEntries(dermquest, 'label', c)
        if numImagesPerClassForValidation is not None:
            N = numImagesPerClassForValidation
        else:
            N = int(0.5 * len(class_))

        # Iterate (randomly) over the unique cases, adding all images from a case to the validation set,
        # until we have enough images for this class
        unique_cases, case_counts = np.unique([m['case'] for m in class_], return_counts=True)
        cases = np.array([(u, c) for u,c in zip(unique_cases, case_counts)])
        np.random.shuffle(cases)
        class_count = 0
        for id_, case_count in cases:
            setEntries(getEntries(class_, 'case', id_), 'set_identifier', VALIDATION_SET)
            class_count += case_count
            if class_count >= N:
                break
    print '\rSupplementing...',

    # Supplement the validation set with entire leaf nodes until each has enough images.
    getClassFromValidationSet = lambda meta, c: [m for m in meta if m['set_identifier'] == VALIDATION_SET and m['label'] == c]
    for c in classes:
        class_size = len(getClassFromValidationSet(meta, c))
        if numImagesPerClassForValidation is not None:
            N = numImagesPerClassForValidation
        else:
            N = 100
        while class_size < N:
            entries = getEntries(meta, 'label', c)

            # Eliminate any entries in the database_exclusions list.
            keep = []
            for e in entries:
                if e['database'] not in database_exclusions:
                    keep.append(e)
            entries = keep

            # Set validation label
            setEntries(getRandomNode(entries), 'set_identifier', VALIDATION_SET)
            class_size = len(getClassFromValidationSet(meta, c))
            print 'Suppllementing Class: %d' % c

    # Set the rest of the entries in meta to be training
    entries = getEntries(meta, 'set_identifier', NO_SET)
    setEntries(entries, 'set_identifier', TRAINING_SET)
    print '\r'


def taxonomyLearning_depth1Mergers(meta, min_class_size):
    """Assign training labels to meta according to a taxonomy-based merging strategy.

        Motivation:
        We try to maximize the entropy of the data distribution by splitting root node classes
        up into subclasses such that each subclass is at least of size min_class_size.
        This seems to improve overall learning as the classifiers are forced to become more
        discriminative.

        Args:
            meta (list): metadata in skindata format. The 'label' fields will be set in this function.
            min_class_size (int): the minimum number of entries assigned to each class.
    """
    clinical_classes, labels = rootNodeClasses(meta)
    num_assigned_clusters = 0

    for c, cname in clinical_classes.iteritems():

        # Grab the root node entries.
        entries = getEntries(meta, 'clinical_label', c)

        # Gather all the depth1 nodes for these entries.
        depth1_nodes = []
        for e in entries:
            depth1_nodes.append(nodeAtDepth(e, 1))

        # Cluster the depth1 nodes based on number of images
        child_nodes, num_images_per_child_node = np.unique(depth1_nodes, return_counts=True)
        cluster_indices = cluster_integers(num_images_per_child_node, min_class_size)

        # Assign to each entry its proper label
        for entry in entries:
            node1 = nodeAtDepth(entry, 1)
            cluster_index = cluster_indices[child_nodes == node1]
            entry['label'] = int(cluster_index + num_assigned_clusters)
        num_assigned_clusters += len(np.unique(cluster_indices))


def textFiletoMeta(meta, text_file):
    """Returns the entries in meta that correspond to images listed in text_file.

    Args:
        meta (list): a list of dictionaries in skindata format
        text_file (string): the filesystem path to a text file containing entries
            in the format:
            [filesystem/path/to/image] [any other information]
    """
    paths = [line.split()[0].strip() for line in open(text_file).readlines()]
    meta_keep = []
    for i, path in enumerate(paths):
        print '\r', i, '/', len(paths),
        filename = os.path.basename(path)
        for entry in meta:
            if filename == entry['file_name']:
                meta_keep.append(entry)
    return meta_keep


#def metaToDataFrame(meta):
#    """Convert 'skindata' formatted metadata into a pandas dataframe."""
#
#    databases = sorted({m['database'] for m in meta})
#    labels = sorted({rootNode(m) for m in meta})
#    unique_filenames = {m['file_name'] for m in meta if 'file_name' in m.keys()}
#
#    A = [[sum(rootNode(m) == label and m['database'] == db
#            for m in meta) for db in databases] for label in labels]
#
#    df = pandas.DataFrame(A, columns=databases, index=labels)
#    df['total meta entries'] = df.sum(axis=1)
#    df.loc['total meta entries'] = df.sum()
#    counts = [len({m['file_name'] for m in meta if m['database']==db}) for db in databases]
#    df.loc['total unique images'] = counts + [len(unique_filenames)]
#    df = df.sort('total meta entries', ascending=True)
#    cols = np.array(databases)[np.argsort(df.loc['total meta entries'].values[:-1])]
#    cols = list(cols) + ['total meta entries']
#    df = df[cols]
#    return df


def collect_synset_tree_learning(meta_train):
    """Returns the synset for meta_train in tree learning format.

    The format is: [label] [clinical class name] [clinical_label]
    label: any int >= 0
    clinical_class_name: something from 'pigmented-lesion-benign', 'pigmented-lesion-malignant', etc.
    clinical_label: an int from 0 to 8

    Args:
        meta_train (list): A list of dictionary entries in skindata format

    Returns:
        The synset as a list of strings in the above format.
    """
    synset = []
    for m in meta_train:
        synset.append([ int(m["label"]), rootNode(m), m["clinical_label"]])
    synset = {tuple(s) for s in synset}
    synset = [list(s) for s in synset]
    synset.sort(key=lambda x: x[2])
    synset.sort(key=lambda x: x[0])
    synset = [[str(ss) for ss in s] for s in synset]
    synset = [" ".join(s) for s in synset]
    return synset


def unique_disease_categories(meta):
    """Returns all unique disease categories in file-system path format.

    Args:
        meta (list): A list of dictionary entries in skindata format.

    Returns:
        A sorted list of file-system paths (strings), each of which is a unique disease.
    """
    len(set([taxonomyPath(m) for m in meta]))
    paths = []
    for m in meta:
        tp = m['tax_path']
        for i in range(len(tp)):
            paths.append('/'.join(['-'.join(tp[j]) for j in range(i+1)]))
    return list(np.unique(paths))


class Field2meta:
    """Class for converting a skindata field into its corresponding metadata entries.

    Example usage:
    f2m = Field2meta(meta, field='file_name')
    fn = 'dermquestImageInMyStuff.jpg'
    m = f2m(fn)
    # m now holds the meta data entries that have 'file_name' of fn.

    """

    def __init__(self, meta, field='file_name'):
        """Initialize a dictionary for lookups.

        Args:
            meta (list): dictionary entries in skindata format
            field (string): the dictionary field in meta to organize by.

        """
        self._lookup_filename_meta = {}
        self.meta = meta
        for m in self.meta:
            self._lookup_filename_meta[m[field]] = []
        for m in meta:
            filename = m[field]
            self._lookup_filename_meta[filename].append(m)


    def __call__(self, field_value):
        """Return's a field_value's corresponding metadata entries."""
        try:
            return self._lookup_filename_meta[field_value]
        except:
            return []


class Filename2meta:
    """Class for converting a link to its corresponding filename.
    """

    def __init__(self, meta):
        self._lookup_filename_meta = {}
        self.meta = meta
        for m in self.meta:
            self._lookup_filename_meta[m['file_name']] = []
        for m in meta:
            filename = m['file_name']
            self._lookup_filename_meta[filename].append(m)


    def __call__(self, filename):
        """Return's a url's corresponding metadata entries."""
        try:
            return self._lookup_filename_meta[filename]
        except:
            return []


def assign_test_set_based_on_duplicates(meta, duplicates, max_cost):
    """Assigns every meta entry to one of train/test based on duplicates and desired max_cost.

    For each entry of duplicates, we add it to the test set if it has less than max_cost duplicates,
    and eliminate its duplicates from the training set.

    Args:
        meta (list): A list of dictionary entries in skindata format
        duplicates (dict): Entries of the form
            test_im_filename : [duplicate_im_filename1, duplicate_im_filename 2, ...]
        max_cost (int): We keep only test set images that have at most max_cost duplicates.

    """
    filename2meta = Filename2meta(meta)

    # Everything starts out in training
    setEntries(meta, 'set_identifier', TRAINING_SET)

    # Assign the test set
    test_set = [key for key in duplicates.keys() if len(duplicates[key]) <= max_cost]
    test_set = [filename2meta(fn) for fn in test_set]
    test_set = [m for mm in test_set for m in mm]
    setEntries(test_set, 'set_identifier', TESTING_SET)

    # Purge the duplicates from the training set
    no_set = [value for key, value in duplicates.iteritems() if len(value) <= max_cost]
    no_set = [filename2meta(u) for uu in no_set for u in uu]
    no_set = [m for list_ in no_set for m in list_]
    setEntries(no_set, 'set_identifier', NO_SET)

