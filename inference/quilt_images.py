"""
Creates a retrieval quilt.
Given training and validation features/filenames, this script finds the
nearest neighbors from validation to train, sorts them by first-nearest-neighbor
distance, and creates a quilt of their thumbnails.

The purpose of this is to quickly search for duplicates, visually.
"""
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

import lib
from lib.inference import retrieval

#IMAGES_DIR = '/ssd/esteva/skindata4/images'
#SAVE_NAME='/tmp/retrieval/quilt.jpg'

#TRAIN_FILENAMES = '/archive/esteva/experiments/skindata4/inflammatory2/retrieve/train-filenames.txt'
#VAL_FILENAMES = '/archive/esteva/experiments/skindata4/inflammatory2/retrieve/validation-filenames.txt'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--images_dir')
parser.add_argument('--save_name', default='/tmp/retrieval/quilt.jpg')
parser.add_argument('--train_filenames')
parser.add_argument('--train_features')
parser.add_argument('--val_filenames')
parser.add_argument('--val_features')

args = parser.parse_args()

def readtxt(fn):
    return np.array([line.strip() for line in open(fn).readlines()])


def readmat(fn):
    return np.load(fn)


def addFullImagePath(basename):
    return os.path.join(args.images_dir, basename)


def keep_unique_filenames(filenames, mat):
    """Reduces filenames and mat based on filename uniqueness.

    Filenames come in the form:
        '-0GnmpIR2i8V5M.jpg'
        '-0GnmpIR2i8V5M.jpg_1'
        '-0GnmpIR2i8V5M.jpg_2'
        ...
    This function strips the _1, _2 from them, then takes the unique set.
    Assumes jpgs only.
    """
    tmp = []
    for f in filenames:
        assert '.jpg' in f, 'Only .jpg is supported: %s' % f
        fn = f.split('.jpg')[0] + '.jpg'
        tmp.append(fn)
    filenames = tmp

    unique_filenames, indices = np.unique(filenames, return_index=True)
    unique_mat = mat[indices]
    return unique_filenames, unique_mat


def NN_sort(dist_mat, filenames):
    """Sorts dist_mat and filenames based on first-NN distance."""
    assert dist_mat.shape[0] == len(filenames)
    d_NN = np.min(dist_mat, axis=1)
    ind_sort = np.argsort(d_NN)
    dist_mat = dist_mat[ind_sort]
    filenames = filenames[ind_sort]
    return dist_mat, filenames


def save_quilt(quilt_paths, M=50):
    """Saves quilted image, M vertical thumbnails at a time."""
    N = len(quilt_paths)

    for i in range(0, N, M):
        print '\rQuilting %d/%d' % (i+M, N),
        quilt = retrieval.quiltTheImages(quilt_paths[i:i+M])
        ext = args.save_name.split('.')[-1]
        fn = ".".join(args.save_name.split('.')[:-1]) +'-' + str(i+M) + '.' + ext
        print fn
        quilt.save(fn)


def main():

    train_filenames = args.train_filenames
    train_features = args.train_features
    val_filenames = args.val_filenames
    val_features = args.val_features

    train_filenames = readtxt(train_filenames)
    train_features = readmat(train_features)
    train_filenames, train_features = keep_unique_filenames(
        train_filenames, train_features)

    val_filenames = readtxt(val_filenames)
    val_features = readmat(val_features)
    val_filenames, val_features = keep_unique_filenames(
        val_filenames, val_features)

    print '# Unique validation images: %d' % len(val_filenames)
    print '# Unique training images: %d' % len(train_filenames)

    # Find nearest neighbors and sort distance matrix and val_filenames by
    # first nearest neighbor distance
    print 'Calculating distance matrix',
    t = time.time()
    dist_mat = euclidean_distances(val_features, train_features)
    dist_mat, val_filenames = NN_sort(dist_mat, val_filenames)
    print 'Elapsed Time: %0.3f' % (time.time() - t)

    t = time.time()
    print 'Calculating Index matrix',
    index_mat = np.argsort(dist_mat, axis=1)
    print 'Elapsed Time: %0.3f' % (time.time() - t)

    # Put filenames from val and train into quilt format
    quilt_paths = []
    N = 10
    print 'Quilting the first %d nearest neighbors' % N
    for fn, indices in zip(val_filenames, index_mat):
        q = [fn]
        q.extend(train_filenames[indices[:N]].tolist())
        q = [addFullImagePath(qq) for qq in q]
        quilt_paths.append(q)

    save_quilt(quilt_paths)


if __name__ == '__main__':
    main()
