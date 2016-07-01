"""Libary of functions for extracting the edges from metadata based on time and turn information.
"""
import datetime
import time
import numpy as np
import scipy.sparse as sp
import json
from scipy.spatial.distance import pdist, squareform
from scipy.misc import comb

import lib
from lib.notebooks.vis_utils import progress, tic, toc
from lib.taxonomy.loading import Field2meta, getEntries, setEntries, getEntryValues
from lib.taxonomy.loading import TRAINING_SET, TESTING_SET, NO_SET, VALIDATION_SET

FIELDS = dict(
    time_field = 'Image DateTime',
    camera_field = 'Image Model'
)


def get_time(m, time_field):
    """Return the 'Image DateTime' IMD of m, a meta entry, if it has it."""
    if 'exif' in m.keys() and time_field in m['exif'].keys():
        return m['exif'][time_field].strip()
    else:
        return None


def to_colons(tt):
    """Replace periods in tt with colons"""
    new_tt = ''
    for i, char in enumerate(tt):
        if char == unicode('.'):
            new_tt += ':'
        else:
            new_tt += char
    return new_tt


def get_ij(T, i,j):
    """Get element T[i,j] from T, when T is not squareform."""
    return T[i * j]


def absolute_time(dt):
    beginning_of_time = datetime.datetime(1,1,1,0,0,0)
    return (dt - beginning_of_time).total_seconds()


def insert_datetime_field(meta):
    """Inserts into meta[i] a datetime field with datetime object whenever meta[i] has meta['exif'][time_field].

    Here, time_field is defined in the dictionary FIELDS, above. It corresponds to the IMD entry that denotes
    when the image was taken.

    Args:
        meta (list): a list of dictionaries in skindata format.
    """
    bad = 0
    time_field = FIELDS['time_field']
    for m in meta:
        t = get_time(m, time_field)
        if t is None:
            continue
        dt = []
        for tt in t.split():
            tt = to_colons(tt)
            dt.extend(tt.split(':'))
        try:
            dt = [int(d) for d in dt]
            dt = datetime.datetime(*dt)
            m['datetime'] = dt
        except:
            bad += 1
    print '[FUNC: insert_datetime_field] Skipping %d entries that could not load datetime' % bad


def insert_abs_time_field(meta):
    """Inserts abs_time field into any meta entry that has a 'datetime' field.

    let m be in meta
    then m['abs_time'] = m['datetime'] - beginning of time, in seconds

    Args:
        meta (list): a list of dictionaries in skindata format.
    """
    meta_datetime = getEntries(meta, 'datetime', None)
    min_time = 0
    for m in meta_datetime:
        dt = (m['datetime'] - meta_datetime[0]['datetime']).total_seconds()
        if dt < min_time:
            min_time = dt
    for m in meta_datetime:
        dt = (m['datetime'] - meta_datetime[0]['datetime']).total_seconds() + min_time
        m['abs_time'] = dt


def extract_cameras(meta):
    """Returns the camera type for each entry in meta.

    If camera type isn't specificed, it returns a special key phrase
    for that meta entry.

    Args:
        meta (list): a list of dictionaries in skindata format.

    Returns:
        cameras (list): a list of strings - the camera types
        camera_models (dict): the unique cameras available.
    """
    cameras = []
    camera_field = FIELDS['camera_field']
    for m in meta:
        if camera_field in m['exif'].keys():
            cameras.append(m['exif'][camera_field].strip())
        else:
            cameras.append(unicode('0_aint_no_camera_model'))
    camera_models = np.unique(cameras)
    camera_models = {c : i for i, c in enumerate(camera_models)}
    return cameras, camera_models


def edge_matrix(abs_times, camera_types):
    """Returns the edge matrix E in non-squareform, calculated using pdist.

    We consider an edge between two metadata entries to exist if:
        a) those entries were taken within 120 seconds of each other, AND
        b) those entries came from different cameras.
    Note: it is recommended that camera_types contains an index for metadata entries
        that did not have a camera type listed.

    Args:
        abs_times (numpy.array): N-dimensional array of floats of absolute times,
            in seconds, that images were taken.
        camera_types (numpy.array): N-dimensional array of ints corresponding to the
            camera types that were used.

    Returns:
        The edge matrix, E.
    """
    assert len(abs_times) == len(camera_types)
    T = pdist(abs_times)
    T = np.asarray(T < 120, dtype=bool)
    C = pdist(camera_types)
    C = np.asarray(C, dtype=bool)
    E = T & C
    return E, T, C


def indx_square2vec(i,j,n):
    """Returns the vector index of matrix indicies i,j, into an n x n matrix.

    Here, the vector index corresponds to the ordering of the 1-dimensional distance
    matrix returned by scipy.spatial.pdist. Note, this requires i < j < n and asserts it.
    Args:
        i (int): row index
        j (int): column index
        n (int): size of matrix
    Returns:
        vector index.
    """
    return int(comb(n,2) - comb(n-i, 2) + (j-i-1))


def squareform_sparse(distance_vector, N):
    """Construct sparse squareform matrix from the distance vector and n, the matrix length/width.

    Args:
        distance_vector (numpy.array): the results of calling scipy.spatial.pdist
        N (int): the dimension of the square matrix to form.

    Returns:
        A sparse NxN matrix whose elements correspond to the elements of the distance vector.
        S[i,j] = distance_vector[indx_square2vec(i,j,n)]
    """
    S = sp.lil_matrix((N,N))
    i = 0
    j = 1
    t = time.time()
    V = np.where(distance_vector)[0] #returns indices to non-zero entries of distance_vector
    row_indexes = []
    count = 0
    for i in range(N):
        row_indexes.append(count)
        count += N-i-1
    row_indexes = np.array(row_indexes)
    t = time.time()
    for idx, v in enumerate(V):
        if idx % 5000 == 0:
            progress(idx, len(V), t)
        elem = distance_vector[v]
        # TODO: calculate i,j
        row = np.where(v >= row_indexes)[0]
        i = len(row) - 1
        j = i + (v - row_indexes[i] ) + 1
        assert i < j < N
        assert i >= 0
        S[i,j] = elem

    return S


def squareform_sparse_slow(distance_vector, N):
    """Construct sparse squareform matrix from the distance vector and n, the matrix length/width.

    Args:
        distance_vector (numpy.array): the results of calling scipy.spatial.pdist
        N (int): the dimension of the square matrix to form.

    Returns:
        A sparse NxN matrix whose elements correspond to the elements of the distance vector.
        S[i,j] = distance_vector[indx_square2vec(i,j,n)]
    """
    S = sp.lil_matrix((N,N))
    i = 0
    j = 1
    t = time.time()
    for v, elem in enumerate(distance_vector):
        if v % 1000000 == 0:
            progress(v, len(distance_vector), t)
        if not elem:
            continue
        S[i, j] = elem
        j += 1
        if j >= N:
            i += 1
            j += i + 1
            j = j % N
        assert i < N
    print '\r',
    return S


def sparse_matrix_iterator(mat):
    """Iterates quickly over a sparse matrix"""
    mat = sp.coo_matrix(mat)
    for i,j,v in zip(mat.row, mat.col, mat.data):
        yield i,j,v


def insert_edges_into_edge_matrix(E, duplicates, meta, field_name='index'):
    """Insert edges into E from d[0] to d[i] for i = 1,..len(d) for d in duplicates.

    Args:
        E (scipy.sparse.lil_matrix): The sparse matrix
        duplicates (list of lists): entries in the form ['duplicate1.jpg', 'duplicate2.jpg', ...]
        meta (list): dictionary entries in skindata format.
        field (string): the skindata dictionary field used to map duplicates into meta entries.
            I.e. 'file_name', or 'index'
    """
    field2meta = Field2meta(meta, field=field_name)
    for group in duplicates:
        meta_0 = field2meta(group[0])
        for g in group[1:]:
            meta_rest = field2meta(g)
            for m in meta_0:
                for n in meta_rest:
                    i = m['index']
                    j = n['index']
                    E[i, j] = True


def connected_component(E, index):
    """Returns the connected component of index in edge matrix E using bread-first-search."""
    return sp.csgraph.breadth_first_order(E, index, return_predecessors=False, directed=False)


def partition_connected_components(meta):
    """Partitions meta into train/test based on connected components information.

    Args:
        meta (list): list of dictionaries in skindata format

    The edge information is taken from two turk runs, dermquest ids, and identical filenames.
    It is hardcoded into this function.

    This function updates 'set_identifier' fields of meta to be either TRAINING_SET or TESTING_SET
    depending on where they get placed.
    """
    test_set = 'duplicate_urls_turk2.json'
    turk_results = [
        'duplicate_urls_turk1.json',
        'duplicate_urls_turk2.json',
        ]

    def component_is_split(comp):
        """Returns true if this component is split between train and test"""
        set_ids = set([m['set_identifier'] for m in comp if m['set_identifier'] != NO_SET])
        return len(set_ids) == 2


    def random_partition(meta_test, return_splits=False):
        """Randomly partitions the components of meta_test to train/test by flipping a coin."""
        split_comps = []
        for m in meta_test:
            comp = cc2meta(m['connected_component'])
            if component_is_split(comp):
                if np.random.rand() < 0.5:
                    setEntries(comp, 'set_identifier', TRAINING_SET)
                else:
                    setEntries(comp, 'set_identifier', TESTING_SET)
                split_comps.append(comp)
        if return_splits:
            return split_comps

    def maxset_partition(meta_test):
        """Deterministically place component into whichever set they already have more images of."""
        for m in meta_test:
            comp = cc2meta(m['connected_component'])
            if component_is_split(comp):
                N_test = len(getEntries(comp, 'set_identifier', TESTING_SET))
                N_train = len(getEntries(comp, 'set_identifier', TRAINING_SET))
                if N_test >= N_train:
                    setEntries(comp, 'set_identifier', TESTING_SET)
                else:
                    setEntries(comp, 'set_identifier', TRAINING_SET)

    # Calculate time-camera edge matrix as a sparse matrix (about 2 minutes)
    for i,m in enumerate(meta):
        m['index'] = i

    insert_datetime_field(meta)
    insert_abs_time_field(meta)

    meta_datetime = getEntries(meta, 'datetime', None)
    cameras, camera_models = extract_cameras(meta_datetime)
    abs_times = np.array([m['abs_time'] for m in meta_datetime]).reshape((-1,1))
    cam_indx = np.array([camera_models[c] for c in cameras]).reshape((-1,1))
    print '%d Entries have the datetime metadata' % len(meta_datetime)

    print 'Calculating time-camera edge matrix...'
    M = len(meta_datetime)
    N = len(meta)

    t = tic()
    edge, _, _= edge_matrix(abs_times, cam_indx)
    edge = squareform_sparse(edge, M)
    toc(t)

    # Initialize the N x N Edge matrix
    E = sp.lil_matrix((N,N), dtype=bool)

    # Insert the datetime edges
    c = 0
    for i,j,v in sparse_matrix_iterator(edge):
        if v:
            c += 1
            idx_i = meta_datetime[i]['index']
            idx_j = meta_datetime[j]['index']
            E[idx_i, idx_j] = v
    print 'Adding %d edges to the graph' % c

    # Add into the edge matrix the duplicates of turk2 and turk 1
    def dict2list(dict_):
        list_ = []
        for key, value in dict_.iteritems():
            d = [key]
            d.extend([v for v in value])
            list_.append(d)
        return list_

    for tr in turk_results:
        turk = json.load(open(tr, 'r'))
        duplicates = dict2list(turk)
        insert_edges_into_edge_matrix(E, duplicates, meta, field_name='filename')
        print 'Adding %d turk edges to the graph' % np.sum([len(v)-1 for v in duplicates])

    # Add dermquest ids into the graph.
    dermquest = getEntries(meta, 'database', 'dermquest')
    dermquest = getEntries(dermquest, 'case', None)
    case2meta = Field2meta(dermquest, field='case')
    cases = np.unique(getEntryValues(dermquest, 'case'))
    duplicates = [[m['index'] for m in case2meta(case)] for case in cases]
    insert_edges_into_edge_matrix(E, duplicates, meta, field_name='index')
    print 'Adding %d dermquest edges to the graph' % np.sum([len(v)-1 for v in duplicates])

    # Add meta entries that share the same filenames as edges
    filename2meta = Field2meta(meta, field='filename')
    filenames = np.unique([m['filename'] for m in meta])
    duplicates = []
    for fn in filenames:
        meta_filename = filename2meta(fn)
        if len(meta_filename) == 0:
            print 'wtf'
            break
        if len(meta_filename) > 1:
            duplicates.append([m['index'] for m in meta_filename])
    insert_edges_into_edge_matrix(E, duplicates, meta, field_name='index')
    print 'Adding %d edges to the graph based on identical filenames' % np.sum([len(v)-1 for v in duplicates])

    # Extract connected components and assign them to the meta
    n_components, connected_components = sp.csgraph.connected_components(E, directed=False)
    unique_component_numbers, component_sizes = np.unique(connected_components, return_counts=True)

    for m, c in zip(meta, connected_components):
        m['connected_component'] = c
    print 'We find %d connected components' % n_components

    # Propose a test set (from the turked set)
    print 'Proposing test set from %s' % test_set
    test_set = json.load(open(test_set, 'r'))
    test_set = [key for key in test_set.keys()]
    filename2meta = Field2meta(meta, field='filename')
    cc2meta = Field2meta(meta, field='connected_component')
    meta_test = [m for fn in test_set for m in filename2meta(fn)]
    setEntries(meta, 'set_identifier', TRAINING_SET)
    setEntries(meta_test, 'set_identifier', TESTING_SET)
    print 'Proposed Test Set has %d entries' % len(meta_test)

    # Iterate over elements of the test set and push connected components to train or test
    maxset_partition(meta_test)
    meta_test = getEntries(meta, 'set_identifier', TESTING_SET)
    print 'Partitioned Test Set has %d meta entries' % len(meta_test)

