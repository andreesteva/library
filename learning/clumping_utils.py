"""Functions to handle the merging of classes based on taxonomy structure. Intended for use with tree-learning"""

import numpy as np
import unittest


def mergeProbabilities(probabilities, mapping, ignore_classes=[]):
    """Merges N x K probabilities matrix into N x L, with L <= K, using mapping.

    Args:
        probabilities (numpy.array): N x K matrix of probabilities. N data points.
        mapping (list): string entries of the form:
            [merge-class-name-0] [softmax-output-class-0]
            [merge-class-name-0] [softmax-output-class-1]
            [merge-class-name-1] [softmax-output-class-2]
        ignore_classes: list of classnames to ignore, by zeroing out their probabilities
            prior to renormalization.

    Returns
        N x L numpy.array of merged probabilities.
    """
    N = probabilities.shape[0]
    merge_classes = np.unique([m.split()[0] for m in mapping])
    L = len(merge_classes)
    merge_probs = np.zeros((N, L))

    for i, mc in enumerate(merge_classes):
        theClasses = np.array([m.split()[0] == mc for m in mapping])
        if mc in ignore_classes:
            merge_probs[:, i] = 0.0
        else:
            merge_probs[:, i] = np.sum(probabilities[:, theClasses], axis=1)
    merge_probs /= np.sum(merge_probs, axis=1).reshape((-1,1))
    return merge_probs



class TestFileFunctions(unittest.TestCase):

    def test_mergeProbabilities(self):
        mapping = [
                'merge0 class0',
                'merge0 class1',
                'merge0 class2',
                'merge1 class3',
                'merge1 class4',
                'merge1 class5',
                'merge1 class6',
                'merge1 class6',
                ]
        N = 10
        probs = np.ones((N, len(mapping)))
        p = mergeProbabilities(probs, mapping)
        assert np.sum(p[:,0]) == N * (3.0 / 8)
        assert np.sum(p[:,1]) == N * (5.0 / 8)


if __name__ == '__main__':
    unittest.main()
