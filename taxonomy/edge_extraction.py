"""Libary of functions for extracting the edges from metadata based on time and turn information.
"""
import datetime
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.misc import comb
import scipy.sparse as sp
import lib
from lib.notebooks.vis_utils import progress

from lib.taxonomy.loading import Field2meta, getEntries

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

#   for m in meta:
#       if 'datetime' in m.keys():
#           m ['abs_time'] = absolute_time(m['datetime'])


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
#   assert i < j < n, 'We require i < j < n. i = %d, j = %d, n = %d' % (i, j, n)
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


