"""Simple script to convert labels file in the form:

        [training-class-0]
        [training-class-1]
        [training-class-2]
        [training-class-3]

    nto a mapping file of the form:

       [validation-class-0] [training-class-0]
       [validation-class-0] [training-class-1]
       [validation-class-0] [training-class-2]
       [validation-class-1] [training-class-3]

Training class entries must have this format:
    cutaneous-lymphoma_cutaneous-t-cell-lymphoma_0
    cutaneous-lymphoma_erythema-arciforme-et-palpabile-migrans_1
They will be split using the first '_'.

"""

import argparse
import sys

parser = argparse.ArgumentParser(description='Labels file conversion.')
parser.add_argument('--labels_file', default = '', type=str)

args = parser.parse_args()
labels_file = args.labels_file
if not labels_file:
    print 'Need labels file'
    sys.exit()

val_labels = [line.strip() for line in open(labels_file).readlines()]
mapping = []
for entry in val_labels:
    m = entry.split('_')[0] + " " + entry
    mapping.append(m)

mapping_file = labels_file + '.mapping'
print 'Creating %s' % mapping_file
with open(mapping_file, 'w') as f:
    prefix = ''
    for m in mapping:
        f.write(prefix)
        f.write(m)
        prefix = '\n'


