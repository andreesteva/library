import numpy as np
import sys

import lib
from lib.taxonomy.loading import *


def taxonomyDelimiter():
    """The delimiter to use for node-name format. Must match that in lib.taxonomy.loading.node"""
    return '/'


def node_based_classes(taxonomy):
    """Splits up the taxonomy into classes such that each node becomes a class.

    Args:
        vertex (Vertex instance): the top node at which to begin parsing.

    Returns:
        A list of lists of metadata entries. Each list is a newly defined class for training.
    """
    new_classes = []
    for vertex in taxonomy.vertices.values():
        if len(vertex.meta) > 0:
            new_classes.append(vertex.meta)
    return new_classes


def recursive_division(vertex, N):
    """Recursively divide up the taxonomy based on data availability by descending it from the top.

    Beginning with vertex, this function does the following:
        if ancestral_images(vertex) > N:
            split up vertex into children
            vertex's personal images become a class
            apply recursive division on the children
        else:
           vertex and its ancestors become a class.

    Args:
        vertex (Vertex instance): the top node at which to begin parsing.
        N (int): If a node and its descendants have more than N images, we split it.
        return_names: bool, if True, returns the node-names of each node that becomes a class.

    Returns:
        1) A list of lists of metadata entries. Each list is a newly defined class for training.
        2) A list of each node-name that was taken as a class.
    """
    if vertex.N <= N:
        return [vertex.collectMetaFromSelfAndDescendants()], [vertex.name]
    else:
        new_classes = []
        new_names = []
        if vertex.meta:
            new_classes.append(vertex.meta)
            new_names.append(vertex.name)

        for child in vertex.children:
            division, division_names = recursive_division(child, N)
            new_classes.extend(division)
            new_names.extend(division_names)
        assert isinstance(new_classes, list)

        for new_class in new_classes:
            for element in new_class:
                assert(isinstance(element, dict))

#       for new_name in new_names:
#           assert(isinstance(new_name, str))

        return new_classes, new_names


class Vertex:
    """This class represents a single node in the taxonomy."""

    def __init__(self, name):
        """Initializes the vertex with its parents and children.

        Args:
            name (string): the vertex's name a taxonmomy path, in node-name format
        """
        self.name = name
        self.parents = []
        self.children = []
        self.meta = []
        self._N = None
        self._N_meta = None


    def addParent(self, parent_vertex):
        self.parents.append(parent_vertex)


    def addChild(self, child_vertex):
        self.children.append(child_vertex)


    def addMetaEntry(self, m):
        self.meta.append(m)


    def collectMetaFromSelfAndDescendants(self):
        meta = []
        meta.extend(self.meta)
        for child in self.children:
            meta.extend(child.collectMetaFromSelfAndDescendants())
        return meta


    @property
    def parentNodeName(self):
        d = taxonomyDelimiter()
        return d.join(self.name.split(d)[:-1])


    @property
    def N(self):
        """Returns the number of unique images represented in this node and its descendents"""
        if self._N is not None:
            return self._N
        self._N = len(self.meta)
        for child in self.children:
            self._N += child.N
        return self._N


    @property
    def N_meta(self):
        """Returns the number of metadata entries represented in this node and its descendents."""
        if self._N_meta is not None:
            return self._N_meta
        self._N_meta = len(set([m['filename'] for m in self.meta]))
        for child in self.children:
            self._N_meta += child.N_meta
        return self._N_meta



class Taxonomy:
    """Contains the entire taxonomy with its structure. Nodes and metadata."""


    def __init__(self, meta):
        """Create taxonomy vertices, and populate them with metadata.

        Args:
            meta(list): a list of dictionaries in skindata format.
        """
        print 'Initializing Taxonomy'
        print 'Creating vertices...'
        self.vertices = {}
        unique_nodes = self.uniqueNodeNames(meta)
        for n in unique_nodes:
            self.addVertex(Vertex(n))

        print 'Distributing metadata entries...'
        for m in meta:
            self.addMetaEntry(m)

        print 'Initializing vertex variables...'
        for _, vertex in self.vertices.iteritems():
            vertex.N_meta
            vertex.N

        print 'Identifying root nodes...'
        root_node_names = np.unique([rootNode(m) for m in meta])
        self.root_nodes = []
        for name in root_node_names:
            self.root_nodes.append(self.vertices[name])

        print 'Adding top node...'
        v = Vertex('/')
        self.vertices['/'] = v
        for root in self.root_nodes:
            v.addChild(root)
            root.addParent(v)


    @property
    def top_node(self):
        return self.vertices['/']


    def addMetaEntry(self, m):
        """Add the metadata entry m into the taxonomy.

        Args:
            m (dict): An entry in skindata format.
        """
        self.vertices[taxonomyPath(m)].addMetaEntry(m)


    def uniqueNodeNames(self, meta):
        """Returns a list of strings of the names of all unique nodes in meta, in node-name format."""
        all_tax_paths = []
        d = taxonomyDelimiter()
        for m in meta:
            path = taxonomyPath(m)
            for i in range(len(path.split(d))):
                all_tax_paths.append(d.join(path.split(d)[:i+1]))
        unique_nodes = np.unique(all_tax_paths)
        return unique_nodes


    def addVertex(self, vertex):
        """Adds a vertex to the taxonomy, linking it to its parent, and its parent to it.

        Args:
            vertex (Vertex): the vertex object. If it exists, the function errors.
        """
        if self.vertex_exists(vertex):
            raise ValueError('Vertex %s exists', vertex.name)
        self.vertices[vertex.name] = vertex
        if vertex.parentNodeName:
            vertex.addParent(self.vertices[vertex.parentNodeName])
            self.vertices[vertex.parentNodeName].addChild(vertex)


    def vertex_exists(self, vertex):
        """Returns true if the taxonomy already has a copy of this vertex."""
        return vertex.name in self.vertices.keys()



