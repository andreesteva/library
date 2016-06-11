"""Script that takes folder train/ and creates train-even/, with evenly distributed classes at the superclass level.

"""

import tensorflow as tf
import os
import numpy as np
import time
import shutil

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir',
                           '/archive/esteva/skindata4/splits/nine-way/train',
                           """Path to the data training directory."""
                           """Must contain subfolders of symlinks to images representing classes and data points."""
                           )
tf.app.flags.DEFINE_string('labels_file',
                           '/archive/esteva/skindata4/splits/nine-way/labels.txt',
                           """Path to the labels file containing the classes."""
                           )



def bufferFillPerGroup(entries, groups=None):
    """Bufferfills a set of path-labels, possibly at the rootNode class level.

    Args:
        entries (list): A list of entries in the format 'path/to/stuff [label]'
        groups (List of Lists, or None): Each sublist must contain a set of integers
            denoting the labels of 'entries' to cluster together into a supergroup.

    Returns:
        A list of these, duplicated, such that each supergroup occurs the same number
        of times. if m = max(instances(supergroup_i), over i), then new_set contains
        m copies of the instances of each supergroup.
    """
    def label(entry):
        return int(entry.strip().split()[1])

    # 'Enough' of a group means as many entries of that group as of the biggest group.
    labels = [label(e) for e in entries]
    if groups is None:
        groups = [[g] for g in set(labels)]
    group_sizes = [len([e for e in entries if label(e) in group]) for group in groups]
    m = max(group_sizes)

    # The strategy here is to iterate over each group until we have enough of that group.
    def group_iterator(entries, group):
        """Iterates forever over the entries that are in group"""
        labels = [int(e.split()[1]) for e in entries]
        relevant = [e for e in entries if label(e) in group]
        i = 0
        while(True):
            yield(relevant[i])
            i += 1
            if i == len(relevant):
                i = 0

    new_entries = []
    for group in groups:
        print 'Generating group', group
        time.sleep(0.5)
        count = 0
        for entry in group_iterator(entries, group):
            new_entries.append(entry)
            count += 1
            if count >= m:
                break
    return new_entries


# TODO: This function is hella slow. Renaming of files should probably be embedded into the group iterator.
def rename_file(filename):
    """Appends a '_' to a filename until the new filename no longer exists."""
    if not os.path.exists(filename):
        return filename
    else:
        filename = filename + '_'
        return rename_file(filename)


def copy(src, dst):
    """Copies symlinks. If dst exists we append a _ to it."""
    dst = rename_file(dst)
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    else:
        shutil.copy(src,dst)


def make_dir_structure(new_dir, classes):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        for c in classes:
            os.makedirs(os.path.join(new_dir, c))
    else:
        print 'Directory exists: %s' % new_dir


def main():
	data_dir = FLAGS.data_dir
	labels_file = FLAGS.labels_file
	new_dir = data_dir + '-even'

	print 'Processing directory %s' % data_dir
	classes = np.array([line.strip() for line in open(labels_file).readlines()])
	unique_classes = np.unique([c.split('_')[0] for c in classes])
	groups = [np.where(c == classes)[0].tolist() for c in unique_classes]

	dataset = []
	for i, c in enumerate(classes):
	    data = os.listdir(os.path.join(data_dir, c))
	    for d in data:
	        dataset.append(' '.join([os.path.join(data_dir, c, d), str(i)]))

	new_dataset = bufferFillPerGroup(dataset, groups)

	print 'Creating directory %s' % new_dir
	make_dir_structure(new_dir, classes)

	for i, entry in enumerate(new_dataset):
	    if i % 10000 == 0:
	        print '\rLinking new entry %d/%d' % (i, len(new_dataset)),
	    src = entry.split()[0]
	    dst = os.path.join(new_dir, src[len(data_dir)+1:])
	    copy(src, dst)


if __name__ == '__main__':
    main()
