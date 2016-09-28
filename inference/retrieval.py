"""
Library of functions for implementing retrieval of images using CNN features.
Works with the Tensorflow Deep Learning Library.
"""

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont



def retrieve(comparison_features, query_features, N):
    """ Return index into comparison_features of closest N euclidean-
        distance matches to query_features.

        Args:
            comparison_features (numpy.array): An array where each row is a
                feature vector to be compared to query_features.
            query_features (numpy.array): a 1D array.
            N (int): The number of nearest neighbors to retreive.

        Returns:
            The indices into comparison_features of the N nearest neighbors, and
            their distances from the query_features.
    """
    dists = np.linalg.norm(comparison_features - query_features,axis=1)
    topN = np.argsort(dists)[:N]
    return topN, dists[topN]



def quiltTheImages(image_paths, patch_size=200, inscribedText=None):
    """Turn a list of lists into a quilted PIL.Image image.

        Args:
            image_paths (list): A list of lists. Each element is an image path.
            patch_size (int): the width/height to use in displaying each patch.
            inscribedText (list): A list of lists. Text to place at the top-left of the images.

        Returns:
            A PIL.Image
    """
    m = 0
    for list_ in image_paths:
        if len(list_) > m: m = len(list_)

    if inscribedText is None:
        inscribedText = [["" for _ in list_] for list_ in image_paths]

    quilt = Image.new('RGB', (patch_size * m, patch_size * len(image_paths)))
    for i, (list_, tlist) in enumerate(zip(image_paths, inscribedText)):
        for j, (path, text) in enumerate(zip(list_, tlist)):
            im = Image.open(path).resize((patch_size, patch_size))
            draw = ImageDraw.Draw(im)
#           font = ImageFont.truetype('arial.ttf', 35)
            if im.mode == 'RGB':
                textcolor = (255,0,0)
            else:
                textcolor = 255
#           draw.text((10,10), text, textcolor, font=font)
            draw.text((10,10), text, textcolor)
            quilt.paste(im, (j * patch_size, i * patch_size))
    return quilt

