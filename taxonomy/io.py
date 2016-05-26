"""Library for read-write functions with the filesystem."""
import os


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
