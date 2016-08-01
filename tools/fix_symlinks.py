"""Relinks all symlinks to the following:

/media/esteva/ExtraDrive1/ThrunResearch/data/skindata4/images/

"""

import os
import argparse

#TODO: Update this script to work with arguments

NEW_LINK_BASE="/media/esteva/ExtraDrive1/ThrunResearch/data/skindata4/images/"

def relink(f):
    """Relinks file f if its a symlink"""
    if os.path.islink(f):
        linkto = os.path.join(NEW_LINK_BASE, os.path.basename(os.readlink(f)))
        print 'Relinking %s-> %s from \n %s' % (f, linkto, os.readlink(f))
        print 'removing %s' % f
        os.remove(f)
        os.symlink(linkto, f)


for root, dirs, files in os.walk("."):
    for f in files:
        filename = os.path.join(root, f)
        relink(filename)

