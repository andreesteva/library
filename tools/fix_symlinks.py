"""Relinks all symlinks to the following:

/media/esteva/ExtraDrive1/ThrunResearch/data/skindata4/images/

"""

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dir', default = '', type=str)

args = parser.parse_args()

NEW_LINK_BASE="/media/esteva/ExtraDrive1/ThrunResearch/data/skindata4/images/"

def relink(f):
    """Relinks file f if its a symlink"""
    if os.path.islink(f):
        linkto = os.path.join(NEW_LINK_BASE, os.path.basename(os.readlink(f)))
       #print 'Relinking %s-> %s from \n %s' % (f, linkto, os.readlink(f))
       #print 'removing %s' % f
        os.remove(f)
        os.symlink(linkto, f)

def main():
    if not args.dir:
        print 'Usage: python fix_symlinks.py --dir=path/to/dir/to/fix'
        return
    else:
        fix_dir = os.path.abspath(args.dir)
    for root, dirs, files in os.walk(fix_dir):
        print 'Fixing %s' % root
        for f in files:
            filename = os.path.join(root, f)
            relink(filename)
    print "Links in %s have been redirected to %s " % (fix_dir, NEW_LINK_BASE)


if __name__ == '__main__':
    main()
