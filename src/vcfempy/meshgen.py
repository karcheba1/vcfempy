"""A module containing attributes, functions, classes and methods for
mesh generation in the Voronoi Cell Finite Element Method (VCFEM).

See Also
--------
`numpy.array <https://numpy.org/doc/stable/reference/generated/\
numpy.array.html>`_
    The type `array_like` refers to objects accepted by the `numpy.array`
    routine.

.. role:: c(class)
.. role:: m(meth)
.. role:: a(attr)
.. role:: f(func)

"""

import distutils.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplclr
from scipy.spatial import Voronoi as Voronoi
import shapely.geometry as shp

import vcfempy.materials as mtl


class PolyMesh2D():
    """A class for 2D polygonal mesh generation.

    Parameters
    ----------
    name : str, optional
        A name for the mesh. If not provided, will be assigned a default
        value.
    vertices : array_like, optional
        Initial vertices to be added to :a:`vertices`. Passed to
        :m:`add_vertices`.
    boundary_vertices : int | list[int], optional
        Initial boundary vertex or list of boundary vertices to be added.
        Passed to :m:`insert_boundary_vertices`.

    Other Parameters
    ----------------
    verbose_printing : bool, optional, default=False
        Flag for verbose printing. Will set :a:`verbose_printing`.
    high_order_quadrature : bool, optional, default=False
        Flag for high order element quadrature generation. Will set
        :a:`high_order_quadrature`.

    Examples
    --------
    >>> # initialize a mesh, no initial input provided
    >>> # (mainly for testing: reset _meshes_created counter)
    >>> import vcfempy.meshgen
    >>> vcfempy.meshgen.PolyMesh2D._num_created = 0
    >>> msh = vcfempy.meshgen.PolyMesh2D()
    >>> print(msh.name)
    Unnamed Mesh 0
    >>> print(msh.num_vertices)
    0

    >>> # initialize another mesh, no initial input
    >>> msh = vcfempy.meshgen.PolyMesh2D()
    >>> print(msh.name)
    Unnamed Mesh 1

    >>> # give the mesh a descriptive name and add some vertices to the mesh
    >>> msh.name = 'test mesh'
    >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
    >>> msh.add_vertices(new_verts)
    >>> print(msh.name)
    test mesh
    >>> print(msh.num_vertices)
    4
    >>> print(msh.vertices)
    [[0. 0.]
     [0. 1.]
     [1. 1.]
     [1. 0.]]

    >>> # define the analysis boundary for the mesh
    >>> # this will also generate the boundary edges
    >>> bnd_verts = [k for k, _ in enumerate(msh.vertices)]
    >>> msh.insert_boundary_vertices(0, bnd_verts)
    >>> print(msh.num_boundary_vertices)
    4
    >>> print(msh.boundary_vertices)
    [0, 1, 2, 3]
    >>> print(msh.boundary_edges)
    [[0, 1], [1, 2], [2, 3], [3, 0]]

    >>> # create a material and material region
    >>> import vcfempy.materials
    >>> m = vcfempy.materials.Material('rock')
    >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, bnd_verts, m, 'rock region')
    >>> print(msh.num_material_regions)
    1
    >>> print(msh.material_regions[0].vertices)
    [0, 1, 2, 3]
    >>> print(msh.material_regions[0].name)
    rock region
    >>> print(msh.material_regions[0].material.name)
    rock

    >>> # generate a simple mesh and check some mesh statistics
    >>> msh.mesh_scale = 0.4
    >>> msh.add_seed_points([0.5, 0.5])
    >>> msh.generate_mesh()
    >>> print(msh.num_nodes)
    16
    >>> print(msh.num_elements)
    9
    >>> print(msh.num_nodes_per_element)
    [4, 4, 4, 4, 4, 4, 4, 4, 4]

    >>> # print out detailed mesh information
    >>> msh.verbose_printing = True
    >>> print(msh)
    vcfempy.meshgen.PolyMesh2D 'test mesh'
    Number of Vertices = 4
    Number of Boundary Vertices = 4
    Number of Material Regions = 1
    Number of Mesh Edges = 0
    Verbose Printing = True
    High Order Quadrature = False
    Mesh Generated = True
    Number of Nodes = 16
    Number of Elements = 9
    Number of Interface Elements = 12
    Number of Boundary Elements = 12
    <BLANKLINE>
    Vertices
    [[0. 0.]
     [0. 1.]
     [1. 1.]
     [1. 0.]]
    <BLANKLINE>
    Boundary Vertices
    [0, 1, 2, 3]
    <BLANKLINE>
    Boundary Edges
    [[0, 1], [1, 2], [2, 3], [3, 0]]
    <BLANKLINE>
    Material Region: rock region, Material: rock
    [0, 1, 2, 3]
    <BLANKLINE>
    Nodes
    [[-2.77555756e-17  1.00000000e+00]
     [ 3.50000000e-01  1.00000000e+00]
     [ 0.00000000e+00  2.77555756e-17]
     [ 3.50000000e-01  2.77555756e-17]
     [ 1.00000000e+00  1.00000000e+00]
     [ 6.50000000e-01  1.00000000e+00]
     [ 6.50000000e-01  6.50000000e-01]
     [ 1.00000000e+00  6.50000000e-01]
     [ 1.00000000e+00 -2.77555756e-17]
     [ 6.50000000e-01 -2.77555756e-17]
     [ 6.50000000e-01  3.50000000e-01]
     [ 1.00000000e+00  3.50000000e-01]
     [ 2.77555756e-17  6.50000000e-01]
     [ 2.77555756e-17  3.50000000e-01]
     [ 3.50000000e-01  6.50000000e-01]
     [ 3.50000000e-01  3.50000000e-01]]
    <BLANKLINE>
    Element Nodes, Areas, Points, Centroids, Materials
    [15, 14, 6, 10], 0.09000000000000002, [0.5 0.5], [0.5 0.5], rock
    [15, 3, 2, 13], 0.12249999999999998, [0.2 0.2], [0.175 0.175], rock
    [12, 0, 1, 14], 0.1225, [0.2 0.8], [0.175 0.825], rock
    [7, 4, 5, 6], 0.1225, [0.8 0.8], [0.825 0.825], rock
    [11, 8, 9, 10], 0.12249999999999998, [0.8 0.2], [0.825 0.175], rock
    [15, 13, 12, 14], 0.10500000000000001, [0.2 0.5], [0.175 0.5  ], rock
    [5, 1, 14, 6], 0.10499999999999998, [0.5 0.8], [0.5   0.825], rock
    [10, 6, 7, 11], 0.10500000000000001, [0.8 0.5], [0.825 0.5  ], rock
    [3, 15, 10, 9], 0.10500000000000001, [0.5 0.2], [0.5   0.175], rock
    <BLANKLINE>
    Interface Element Nodes and Neighbors
    [5, 6], [3, 6]
    [6, 7], [3, 7]
    [9, 10], [4, 8]
    [10, 11], [4, 7]
    [6, 10], [7, 0]
    [12, 14], [5, 2]
    [13, 15], [5, 1]
    [14, 15], [5, 0]
    [1, 14], [2, 6]
    [3, 15], [1, 8]
    [6, 14], [0, 6]
    [10, 15], [0, 8]
    <BLANKLINE>
    Boundary Element Nodes and Neighbors
    [0, 1], 2
    [2, 3], 1
    [4, 5], 3
    [4, 7], 3
    [1, 5], 6
    [8, 9], 4
    [8, 11], 4
    [7, 11], 7
    [3, 9], 8
    [12, 13], 5
    [0, 12], 2
    [2, 13], 1

    >>> # plot the mesh and save an image
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = plt.gca()
    >>> ax = msh.plot_boundaries()
    >>> ax = msh.plot_mesh()
    >>> ax = msh.plot_vertices()
    >>> ax = msh.plot_nodes()
    >>> xmin, xmax, ymin, ymax = plt.axis('equal')
    >>> xlab_text = ax.set_xlabel('x', fontweight='bold')
    >>> ylab_text = ax.set_ylabel('y', fontweight='bold')
    >>> title_text = ax.set_title('Simple square mesh', fontweight='bold')
    >>> plt.savefig('PolyMesh2D_simple_mesh_example.png')
    """

    _num_created = 0

    def __init__(self, name=None, vertices=None, boundary_vertices=None,
                 verbose_printing=False, high_order_quadrature=False):
        # initialize name
        if name is None:
            name = f'Unnamed Mesh {PolyMesh2D._num_created}'
        self.name = name
        PolyMesh2D._num_created += 1

        # initialize vertices
        self._vertices = np.empty((0, 2))
        self.add_vertices(vertices)

        # initialize boundary vertices and edges
        self._boundary_vertices = []
        self.insert_boundary_vertices(0, boundary_vertices)

        # initialize boundary edges and mesh properties
        # Note: Although inserting boundary vertices sometimes does this
        #       this is still necessary in case boundary_vertices is None
        #       or an empty list
        self._generate_boundary_edges()
        self.mesh_valid = False

        # initialize material regions and mesh edges
        self._material_regions = []
        self._mesh_edges = []

        # initialize flags for
        #      verbose printing
        #      high order quadrature in all elements
        self.verbose_printing = verbose_printing
        self.high_order_quadrature = high_order_quadrature

        # intialize mesh properties
        self._mesh_scale = None
        self.mesh_rand = 0.0
        self._seed_points = np.empty((0, 2))

    @property
    def name(self):
        """A descriptive name for the :c:`PolyMesh2D`.

        Parameters
        ----------
        name : str
            The name of the :c:`PolyMesh2D`. Will be cast to `str` regardless
            of type.

        Returns
        -------
        `str`
            The :a:`name` of the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # create a blank mesh without a name (reset mesh counter)
        >>> import vcfempy.meshgen
        >>> vcfempy.meshgen.PolyMesh2D._num_created = 0
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.name)
        Unnamed Mesh 0

        >>> # setting the name
        >>> msh.name = 'Flow analysis mesh'
        >>> print(msh.name)
        Flow analysis mesh

        >>> # changing the name property to non-str
        >>> # will be cast to str
        >>> msh.name = 1
        >>> print(msh.name)
        1
        >>> print(type(msh.name).__name__)
        str

        >>> # initialize a mesh with a name
        >>> msh = vcfempy.meshgen.PolyMesh2D('new mesh')
        >>> print(msh.name)
        new mesh

        >>> # initialize another mesh without a name
        >>> # notice that the "Untitled" counter increases for every mesh
        >>> # created (including those that were assigned an initial name)
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.name)
        Unnamed Mesh 2
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def num_vertices(self):
        """Number of vertices defining the :c:`PolyMesh2D` geometry.

        Returns
        -------
        `int`
            The number of :a:`vertices` in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # creating a mesh, no initial vertices provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.num_vertices)
        0

        >>> # creating a mesh, providing initial vertices
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh = vcfempy.meshgen.PolyMesh2D(vertices=new_verts)
        >>> print(msh.num_vertices)
        4

        >>> # add a vertex and check num_vertices
        >>> msh.add_vertices([1.5, 0.5])
        >>> print(msh.num_vertices)
        5
        """
        return len(self.vertices)

    @property
    def vertices(self):
        """Array of vertex coordinates defining the :c:`PolyMesh2D` geometry.

        Returns
        -------
        `numpy.ndarray`, shape = (:a:`num_vertices`, 2)
            Array of vertex coordinates in the :c:`PolyMesh2D`.

        Note
        ----
        The :a:`vertices` property is not intended to be directly mutable.
        Instead, modify it using the :m:`add_vertices` and
        :m:`delete_vertices` methods.

        Examples
        --------
        >>> # initialize a mesh, no initial vertices provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.vertices)
        []

        >>> # add some vertices
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> print(msh.vertices)
        [[0. 0.]
         [0. 1.]
         [1. 1.]
         [1. 0.]]
        """
        return self._vertices

    def add_vertices(self, vertices):
        """Add vertices to the :c:`PolyMesh2D`.

        Parameters
        ----------
        vertices : `array_like`, shape = (2, ) | (n, 2)
            A pair of coordinates or array of coordinate pairs to add to the
            :c:`PolyMesh2D`.

        Raises
        ------
        TypeError
            If **vertices** has no len() property (e.g. an `int`)
        ValueError
            If contents of **vertices** cannot be cast to `float`.
            If **vertices** cannot be stacked with :a:`vertices` (e.g. due to
            incompatible shape).

        Examples
        --------
        >>> # create a mesh, passing initial vertex list
        >>> import vcfempy.meshgen
        >>> verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh = vcfempy.meshgen.PolyMesh2D(vertices=verts)
        >>> print(msh.vertices)
        [[0. 0.]
         [0. 1.]
         [1. 1.]
         [1. 0.]]

        >>> # add an individual vertex
        >>> msh.add_vertices([1.5, 0.5])
        >>> print(msh.vertices)
        [[0.  0. ]
         [0.  1. ]
         [1.  1. ]
         [1.  0. ]
         [1.5 0.5]]

        >>> # add multiple vertices
        >>> msh.add_vertices([[-0.5, 0.5], [0.5, 1.5]])
        >>> print(msh.vertices)
        [[ 0.   0. ]
         [ 0.   1. ]
         [ 1.   1. ]
         [ 1.   0. ]
         [ 1.5  0.5]
         [-0.5  0.5]
         [ 0.5  1.5]]

        >>> # add nothing, in two different ways
        >>> msh.add_vertices(None)
        >>> msh.add_vertices([])
        >>> print(msh.vertices)
        [[ 0.   0. ]
         [ 0.   1. ]
         [ 1.   1. ]
         [ 1.   0. ]
         [ 1.5  0.5]
         [-0.5  0.5]
         [ 0.5  1.5]]

        >>> # try to add some incompatible types/shapes
        >>> msh.add_vertices(['one', 'two'])
        Traceback (most recent call last):
            ...
        ValueError: could not convert string to float: 'one'
        >>> msh.add_vertices(1)
        Traceback (most recent call last):
            ...
        TypeError: object of type 'int' has no len()
        >>> msh.add_vertices([1]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        >>> msh.add_vertices([1, 2, 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        """
        if vertices is None or len(vertices) == 0:
            return
        vertices = np.array(vertices, dtype=float)
        self._vertices = np.vstack([self.vertices, vertices])

    def delete_vertices(self, del_verts):
        """Remove vertices from the :c:`PolyMesh2D`.

        Parameters
        ----------
        del_verts : list[int]
            List of indices of vertices to remove.

        Raises
        ------
        TypeError
            If elements in **del_verts** cannot be cast to `int`.
        IndexError
            If elements in **del_verts** are >= :a:`num_vertices` or are
            < -:a:`num_vertices`.

        Note
        ----
        Prior to removing vertices, an attempt is made to cast **del_verts**
        to a flattened `numpy.array` of `int`. Duplicate indices are deleted,
        including positive and negative indices that refer to the same
        vertex. Each vertex removed will result in removal of that vertex
        from :a:`boundary_vertices`, :a:`mesh_edges`, and
        :a:`material_regions` and decrementing other vertex indices
        accordingly.

        Examples
        --------
        >>> # initialize a mesh and add/delete some vertices
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> print(msh.num_vertices)
        4
        >>> print(msh.boundary_vertices)
        [0, 1, 2, 3]
        >>> msh.delete_vertices(2)
        >>> print(msh.num_vertices)
        3
        >>> print(msh.vertices)
        [[0. 0.]
         [0. 1.]
         [1. 0.]]
        >>> print(msh.boundary_vertices)
        [0, 1, 2]
        >>> print(msh.boundary_edges)
        [[0, 1], [1, 2], [2, 0]]

        >>> # add some vertices, material regions, and mesh edges
        >>> # then delete vertices
        >>> # note that deleting vertices does not delete material regions
        >>> # or mesh edges, even if it leaves them with < 2 vertices
        >>> msh.add_vertices([[1, 1], [1.5, 0.5]])
        >>> msh.insert_boundary_vertices(2, [3, 4])
        >>> print(msh.vertices)
        [[0.  0. ]
         [0.  1. ]
         [1.  0. ]
         [1.  1. ]
         [1.5 0.5]]
        >>> print(msh.boundary_vertices)
        [0, 1, 3, 4, 2]
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh,
        ...                                       msh.boundary_vertices)
        >>> msh.add_vertices([[0.1, 0.1], [0.8, 0.2], [0.1, 0.8], [0.8, 0.9]])
        >>> me = [vcfempy.meshgen.MeshEdge2D(msh, [k, k+1]) for k in [5, 7]]
        >>> msh.delete_vertices([4, 6])
        >>> print(msh.vertices)
        [[0.  0. ]
         [0.  1. ]
         [1.  0. ]
         [1.  1. ]
         [0.1 0.1]
         [0.1 0.8]
         [0.8 0.9]]
        >>> print(msh.boundary_vertices)
        [0, 1, 3, 2]
        >>> print(mr.vertices)
        [0, 1, 3, 2]
        >>> print(msh.boundary_edges)
        [[0, 1], [1, 3], [3, 2], [2, 0]]
        >>> print(msh.num_mesh_edges)
        2
        >>> for me in msh.mesh_edges:
        ...     print(me.vertices)
        [4]
        [5, 6]

        >>> # generate a mesh, then delete a vertex
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.num_elements)
        9
        >>> msh.delete_vertices(-1)
        >>> print(msh.vertices)
        [[0.  0. ]
         [0.  1. ]
         [1.  0. ]
         [1.  1. ]
         [0.1 0.1]
         [0.1 0.8]]
        >>> print(msh.boundary_vertices)
        [0, 1, 3, 2]
        >>> print(msh.boundary_edges)
        [[0, 1], [1, 3], [3, 2], [2, 0]]
        >>> print(msh.num_mesh_edges)
        2
        >>> for me in msh.mesh_edges:
        ...     print(me.vertices)
        [4]
        [5]
        >>> print(msh.mesh_valid)
        False
        >>> print(msh.num_elements)
        0

        >>> # delete no vertices, in multiple ways
        >>> msh.delete_vertices(None)
        >>> msh.delete_vertices([])
        >>> print(msh.num_vertices)
        6

        >>> # try to delete some invalid vertices
        >>> msh.delete_vertices('one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> msh.delete_vertices(6)
        Traceback (most recent call last):
            ...
        IndexError: at least one index out of range
        >>> msh.delete_vertices(-7)
        Traceback (most recent call last):
            ...
        IndexError: at least one index out of range
        >>> msh.delete_vertices(
        ...                 [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        """
        # pre-process del_verts: flatten, check for invalid indices, and
        # eliminate duplicates
        if del_verts is None:
            return
        del_verts = np.array(del_verts, dtype=int, ndmin=1)
        if len(del_verts) == 0:
            return
        del_verts = np.where(del_verts < 0,
                             del_verts + self.num_vertices,
                             del_verts)
        del_verts = np.unique(del_verts)
        if np.any(del_verts < 0) or np.any(del_verts >= self.num_vertices):
            raise IndexError('at least one index out of range')

        # delete the vertices
        self._vertices = np.delete(self._vertices, del_verts, 0)

        # remove vertices from boundary vertices, mesh edges, and
        # material regions
        for dv in del_verts:
            if dv in self.boundary_vertices:
                self.remove_boundary_vertices(dv)
            for me in self.mesh_edges:
                if dv in me.vertices:
                    me.remove_vertices(dv)
            for mr in self.material_regions:
                if dv in mr.vertices:
                    mr.remove_vertices(dv)

        # decrement remaining vertex indices
        # go in reverse order so that multiple deletions are properly counted
        for dv in del_verts[-1::-1]:
            for k, bv in enumerate(self.boundary_vertices):
                if bv > dv:
                    self.boundary_vertices[k] -= 1
            for me in self.mesh_edges:
                for k, mv in enumerate(me.vertices):
                    if mv > dv:
                        me.vertices[k] -= 1
            for mr in self.material_regions:
                for k, mv in enumerate(mr.vertices):
                    if mv > dv:
                        mr.vertices[k] -= 1

        # reset properties
        self._generate_boundary_edges()
        self.mesh_valid = False

    @property
    def num_boundary_vertices(self):
        """Number of vertices defining the :c:`PolyMesh2D` boundary geometry.

        Returns
        -------
        `int`
            The number of :a:`boundary_vertices` in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # create a mesh, add some vertices
        >>> # no boundary vertices added yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> print(msh.num_boundary_vertices)
        0

        >>> # add some boundary vertices
        >>> msh.insert_boundary_vertices(0, [k for k, _
        ...                                  in enumerate(msh.vertices)])
        >>> print(msh.num_boundary_vertices)
        4

        >>> # create a new mesh, providing initial vertices
        >>> # and boundary vertices
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1.5, 0.5], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh = vcfempy.meshgen.PolyMesh2D(vertices=new_verts,
        ...                                  boundary_vertices=bnd_verts)
        >>> print(msh.num_boundary_vertices)
        5
        """
        return len(self.boundary_vertices)

    @property
    def boundary_vertices(self):
        """List of vertex indices defining :c:`PolyMesh2D` boundary geometry.

        Returns
        -------
        `list[int]`
            The list of vertex indices defining the :c:`PolyMesh2D` boundary.

        Note
        ----
        The :a:`boundary_vertices` property is not intended to be directly
        mutable. Instead modify it using the :m:`insert_boundary_vertices`,
        :m:`remove_boundary_vertices`, and :m:`pop_boundary_vertex` methods.

        Examples
        --------
        >>> # create a new mesh, no initial vertices provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.boundary_vertices)
        []

        >>> # add some vertices and boundary vertices
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.boundary_vertices)
        [0, 1, 2, 3]
        """
        return self._boundary_vertices

    @property
    def boundary_edges(self):
        """List of lists of vertex indices defining boundary edges in the
        :c:`PolyMesh2D`.

        Returns
        -------
        `list[list[int]]`, shape = (:a:`num_boundary_vertices`, 2)
            The list of index pairs that define each edge on the boundary of
            the :c:`PolyMesh2D`.

        Note
        ----
        The :a:`boundary_edges` property is not intended to be directly
        mutable. It is updated each time :a:`boundary_vertices` is changed by
        a method such as :m:`insert_boundary_vertices`,
        :m:`remove_boundary_vertices`, or :m:`pop_boundary_vertex`, so the
        number of :a:`boundary_edges` should always be
        :a:`num_boundary_vertices` since the geometry is a closed polygon.

        Examples
        --------
        >>> # creating a new mesh, no initial vertices provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.boundary_edges)
        []

        >>> # add some vertices and boundary vertices
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1.5, 0.5], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.num_boundary_vertices)
        5
        >>> print(len(msh.boundary_edges))
        5
        >>> print(msh.boundary_edges)
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]

        >>> # remove a boundary vertex, boundary edges are updated
        >>> msh.remove_boundary_vertices([1, 3])
        >>> print(msh.num_boundary_vertices)
        3
        >>> print(msh.boundary_edges)
        [[0, 2], [2, 4], [4, 0]]
        """
        return self._boundary_edges

    def insert_boundary_vertices(self, index, boundary_vertices):
        """Insert one or more boundary vertex indices to the :c:`PolyMesh2D`.

        Parameters
        ----------
        index : int
            The index at which to insert the **boundary_vertices** into
            :a:`boundary_vertices`.
        boundary_vertices : int | list[int]
            The list of vertex indices to add to :a:`boundary_vertices`.

        Note
        -----
        Before inserting the values in **boundary_vertices**, an attempt is
        made to cast to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        TypeError
            If **index** cannot be interpreted as `int`.
        ValueError
            If **boundary_vertices** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **boundary_vertices** cannot be cast to `int`,
            are already in :a:`boundary_vertices`, are negative, or are
            >= :a:`num_vertices`.

        Examples
        --------
        >>> # create mesh, add some vertices, and add boundary vertices
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> print(msh.boundary_vertices)
        [0, 1, 2, 3]

        >>> # add a single vertex and add it as a boundary vertex
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(index=3, boundary_vertices=4)
        >>> print(msh.boundary_vertices)
        [0, 1, 2, 4, 3]

        >>> # add two more vertices and add multiply boundary vertices
        >>> msh.add_vertices([[0.25, 1.25], [0.75, 1.25]])
        >>> msh.insert_boundary_vertices(2, [5, 6])
        >>> print(msh.boundary_vertices)
        [0, 1, 5, 6, 2, 4, 3]

        >>> # the list of boundary vertices need not be 1d
        >>> # if not, it will be flattened
        >>> msh.add_vertices([[-0.5, 0.1], [-0.75, 0.25],
        ...                   [-0.75, 0.75], [-0.5, 0.9]])
        >>> msh.insert_boundary_vertices(1, [[7, 8], [9, 10]])
        >>> print(msh.boundary_vertices)
        [0, 7, 8, 9, 10, 1, 5, 6, 2, 4, 3]

        >>> # add no boundary vertices, in two different ways
        >>> msh.insert_boundary_vertices(0, None)
        >>> msh.insert_boundary_vertices(0, [])
        >>> print(msh.boundary_vertices)
        [0, 7, 8, 9, 10, 1, 5, 6, 2, 4, 3]

        >>> # try to insert some invalid boundary vertices
        >>> msh.insert_boundary_vertices(0, 'one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> msh.insert_boundary_vertices(0, 1)
        Traceback (most recent call last):
            ...
        ValueError: 1 is already a boundary vertex
        >>> msh.insert_boundary_vertices(0, 11)
        Traceback (most recent call last):
            ...
        ValueError: vertex index 11 out of range
        >>> msh.insert_boundary_vertices(0, -1)
        Traceback (most recent call last):
            ...
        ValueError: vertex index -1 out of range
        >>> msh.insert_boundary_vertices(
        ...             0, [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        >>> msh.add_vertices([0.5, -0.5])
        >>> msh.insert_boundary_vertices('one', 11)
        Traceback (most recent call last):
            ...
        TypeError: 'str' object cannot be interpreted as an integer
        """
        if boundary_vertices is None:
            return
        boundary_vertices = np.array(boundary_vertices, dtype=int, ndmin=1)
        if len(boundary_vertices) == 0:
            return
        boundary_vertices = np.flip(boundary_vertices.ravel())
        for bv in boundary_vertices:
            if bv in self.boundary_vertices:
                raise ValueError(f'{bv} is already a boundary vertex')
            if bv < 0 or bv >= self.num_vertices:
                raise ValueError(f'vertex index {bv} out of range')
            self.boundary_vertices.insert(index, int(bv))
        self._generate_boundary_edges()
        self.mesh_valid = False

    def remove_boundary_vertices(self, remove_vertices):
        """Remove one or more boundary vertex indices from the
        :c:`PolyMesh2D`.

        Parameters
        ----------
        remove_vertices : int | list[int]
            The vertex or list of vertices to remove from
            :a:`boundary_vertices`.

        Note
        -----
        Before removing the values in **remove_vertices**, an attempt will be
        made to cast it to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        ValueError
            If **remove_vertices** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **remove_vertices** cannot be cast to `int` or
            are not in :a:`boundary_vertices`.

        Examples
        --------
        >>> # create mesh, add some vertices, add/remove boundary vertices
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.remove_boundary_vertices(1)
        >>> print(msh.boundary_vertices)
        [0, 2, 3]

        >>> # remove multiple boundary vertices
        >>> msh.insert_boundary_vertices(0, 1)
        >>> msh.remove_boundary_vertices([1, 3])
        >>> print(msh.boundary_vertices)
        [0, 2]

        >>> # the list of boundary vertices to remove need not be 1d
        >>> # if not, it will be flattened
        >>> msh.insert_boundary_vertices(1, 1)
        >>> msh.insert_boundary_vertices(3, 3)
        >>> msh.remove_boundary_vertices([[0, 1], [2, 3]])
        >>> print(msh.boundary_vertices)
        []

        >>> # remove no boundary vertices, in two different ways
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.remove_boundary_vertices(None)
        >>> msh.remove_boundary_vertices([])
        >>> print(msh.boundary_vertices)
        [0, 1, 2, 3]

        >>> # try to remove some invalid boundary vertices
        >>> msh.remove_boundary_vertices('one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> msh.remove_boundary_vertices(4)
        Traceback (most recent call last):
            ...
        ValueError: list.remove(x): x not in list
        >>> msh.remove_boundary_vertices(
        ...                 [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        """
        if remove_vertices is None:
            return
        remove_vertices = np.array(remove_vertices, dtype=int, ndmin=1)
        if len(remove_vertices) == 0:
            return
        remove_vertices = remove_vertices.ravel()
        for rv in remove_vertices:
            self.boundary_vertices.remove(rv)
        self._generate_boundary_edges()
        self.mesh_valid = False

    def pop_boundary_vertex(self, pop_index=-1):
        """Pop a boundary vertex at a given index of :a:`boundary_vertices`
        from the :c:`PolyMesh2D`.

        Parameters
        ----------
        pop_index : int, optional, default=-1
            The index at which to remove the boundary vertex.

        Returns
        -------
        `int`
            The value of the boundary vertex that was removed.

        Raises
        ------
        TypeError
            If **pop_index** cannot be cast to `int`.
        IndexError
            If **pop_index** is not a valid index into
            :a:`boundary_vertices`.

        Examples
        --------
        >>> # create mesh, add some vertices, add/remove boundary vertices
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.pop_boundary_vertex()
        3
        >>> print(msh.boundary_vertices)
        [0, 1, 2]

        >>> # remove a boundary vertex from the middle
        >>> msh.insert_boundary_vertices(0, 3)
        >>> msh.pop_boundary_vertex(1)
        0
        >>> print(msh.boundary_vertices)
        [3, 1, 2]

        >>> # try to remove some invalid boundary vertices
        >>> msh.pop_boundary_vertex('one')
        Traceback (most recent call last):
            ...
        TypeError: 'str' object cannot be interpreted as an integer
        >>> msh.pop_boundary_vertex(4)
        Traceback (most recent call last):
            ...
        IndexError: pop index out of range
        >>> msh.remove_boundary_vertices([1, 2, 3])
        >>> msh.pop_boundary_vertex()
        Traceback (most recent call last):
            ...
        IndexError: pop from empty list
        """
        ind = self.boundary_vertices.pop(pop_index)
        self._generate_boundary_edges()
        self.mesh_valid = False
        return ind

    def _generate_boundary_edges(self):
        """Generate boundary edge pairs from :m:`boundary_vertices`.

        Note
        ----
        This is a (private) helper function, called by other methods such as
        :m:`insert_boundary_vertices`, :m:`remove_boundary_vertices`, and
        :m:`pop_boundary_vertex` whenever :m:`boundary_vertices` changes. It
        should not normally be necessary to call this explicitly.
        """
        self._boundary_edges = [[self.boundary_vertices[k],
                                 self.boundary_vertices[(k+1)
                                 % self.num_boundary_vertices]]
                                for k in range(self.num_boundary_vertices)]

    @property
    def num_material_regions(self):
        """Number of material regions used to assign
        :c:`vcfempy.materials.Material` types to the :a:`elements` in the
        :c:`PolyMesh2D`.

        Returns
        -------
        `int`
            The number of :a:`material_regions` in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # creating a mesh, no initial material_regions provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1.5, 0.5], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.num_material_regions)
        0

        >>> # create a new material region, it will be added to the mesh
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh)
        >>> print(msh.num_material_regions)
        1

        >>> # remove the material region from the mesh
        >>> msh.remove_material_region(mr)
        >>> print(msh.num_material_regions)
        0
        """
        return len(self.material_regions)

    @property
    def material_regions(self):
        """List of :c:`MaterialRegion2D` in the :c:`PolyMesh2D` defining mesh
        material geometry.

        Returns
        -------
        `list` of :c:`MaterialRegion2D`
            The list of material regions in the :c:`PolyMesh2D`.

        Note
        ----
        The list of :a:`material_regions` is not intended to be directly
        mutable. Instead modify it using the :m:`add_material_region` and
        :m:`remove_material_region` methods. New :c:`MaterialRegion2D`
        objects require a parent mesh to be set, and by default will be
        added to that parent mesh.

        Examples
        --------
        >>> # initiaize a mesh, no material regions provided yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> bnd_verts = [k for k, _ in enumerate(msh.vertices)]
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.vertices)
        [[0. 0.]
         [0. 1.]
         [1. 1.]
         [1. 0.]]
        >>> print(msh.material_regions)
        []

        >>> # add a material region to the mesh
        >>> # this material region fills the bottom half of the mesh
        >>> import vcfempy.materials
        >>> msh.add_vertices([[0, 0.5], [1, 0.5]])
        >>> print(msh.vertices)
        [[0.  0. ]
         [0.  1. ]
         [1.  1. ]
         [1.  0. ]
         [0.  0.5]
         [1.  0.5]]
        >>> rock = vcfempy.materials.Material('rock')
        >>> mr_rock_verts = [0, 4, 5, 3]
        >>> mr_rock = vcfempy.meshgen.MaterialRegion2D(msh,
        ...                                            mr_rock_verts,
        ...                                            rock)
        >>> print(msh.num_material_regions)
        1
        >>> for mr in msh.material_regions:
        ...     print(mr.vertices)
        [0, 4, 5, 3]

        >>> # add another material region filling the top half of the mesh
        >>> sand = vcfempy.materials.Material('sand')
        >>> mr_sand_verts = [4, 1, 2, 5]
        >>> mr_sand = vcfempy.meshgen.MaterialRegion2D(msh,
        ...                                            mr_sand_verts,
        ...                                            sand)
        >>> print(msh.num_material_regions)
        2
        >>> for mr in msh.material_regions:
        ...     print(mr.vertices)
        [0, 4, 5, 3]
        [4, 1, 2, 5]

        >>> # remove a material region from the mesh
        >>> msh.remove_material_region(mr_rock)
        >>> print(msh.num_material_regions)
        1
        >>> for mr in msh.material_regions:
        ...     print(mr.vertices)
        [4, 1, 2, 5]
        """
        return self._material_regions

    def add_material_region(self, material_region):
        """Add a :c:`MaterialRegion2D` to the :c:`PolyMesh2D`.

        Parameters
        ----------
        material_region : :c:`MaterialRegion2D`
            :c:`MaterialRegion2D` to add to the :c:`PolyMesh2D`.

        Raises
        ------
        TypeError
            If **material_region** is not a :c:`MaterialRegion2D`.
        ValueError
            If **material_region** is already in :a:`material_regions` or
            does not have this :c:`PolyMesh2D` as its parent.

        Note
        ----
        It is not normally necessary to call :m:`add_material_region` when
        creating a new :c:`MaterialRegion2D` since it will add itself to the
        parent :c:`PolyMesh2D` by default. This is only necessary if the
        :c:`MaterialRegion2D` was created with **add_to_mesh** = ``False`` or
        if the :c:`MaterialRegion2D` was previously removed from the
        :c:`PolyMesh2D` using :m:`remove_material_region`.

        Examples
        --------
        >>> # create a mesh and a material region, this adds the material
        >>> # region by default
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh)
        >>> print(mr.mesh.name)
        test mesh
        >>> print(mr in msh.material_regions)
        True

        >>> # create another material region, but do not add it to the mesh
        >>> mr_new = vcfempy.meshgen.MaterialRegion2D(msh, add_to_mesh=False)
        >>> print(mr_new.mesh.name)
        test mesh
        >>> print(mr_new in msh.material_regions)
        False

        >>> # add the new material region to its parent mesh
        >>> msh.add_material_region(mr_new)
        >>> print(mr_new in msh.material_regions)
        True

        >>> # try to add invalid material regions
        >>> msh.add_material_region(1)
        Traceback (most recent call last):
            ...
        TypeError: material region not vcfempy.meshgen.MaterialRegion2D
        >>> msh.add_material_region(mr)
        Traceback (most recent call last):
            ...
        ValueError: material region already in list
        >>> new_msh = vcfempy.meshgen.PolyMesh2D()
        >>> mr_new = vcfempy.meshgen.MaterialRegion2D(new_msh)
        >>> msh.add_material_region(mr_new)
        Traceback (most recent call last):
            ...
        ValueError: material region does not have self as mesh
        """
        if not isinstance(material_region, MaterialRegion2D):
            raise TypeError('material region not '
                            + 'vcfempy.meshgen.MaterialRegion2D')
        if material_region in self.material_regions:
            raise ValueError('material region already in list')
        if material_region.mesh is not self:
            raise ValueError('material region does not have self as mesh')
        self.material_regions.append(material_region)
        self.mesh_valid = False

    def remove_material_region(self, material_region):
        """Remove a :c:`MaterialRegion2D` from the :c:`PolyMesh2D`.

        Parameters
        ----------
        material_region : :c:`MaterialRegion2D`
            :c:`MaterialRegion2D` to remove from the :c:`PolyMesh2D`.

        Raises
        ------
        ValueError
            If **material_region** is not in :a:`material_regions`.

        Note
        ----
        When removing a material region from the :c:`PolyMesh2D`, the
        :a:`MaterialRegion2D.mesh` is not changed, and it can be added again
        using :m:`add_material_region` if desired.

        Examples
        --------
        >>> # create a mesh and a material region, then remove it
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh)
        >>> msh.remove_material_region(mr)
        >>> print(msh.material_regions)
        []

        >>> # try to remove a material region that is not in the mesh
        >>> msh.remove_material_region(mr)
        Traceback (most recent call last):
            ...
        ValueError: list.remove(x): x not in list
        """
        self.material_regions.remove(material_region)
        self.mesh_valid = False

    @property
    def num_mesh_edges(self):
        """Number of non-boundary edges to be preserved in mesh generation
        for the :c:`PolyMesh2D`.

        Returns
        -------
        `int`
            The number of :c:`MeshEdge2D` in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # create a new mesh, no initial information provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.num_mesh_edges)
        0

        >>> # add some vertices, create a new mesh edge, and add it to
        >>> # the mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1.5, 0.5], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> new_verts = [[0.1, 0.1], [0.2, 0.2]]
        >>> msh.add_vertices(new_verts)
        >>> me = vcfempy.meshgen.MeshEdge2D(msh, [5, 6])
        >>> print(msh.num_mesh_edges)
        1
        """
        return len(self.mesh_edges)

    @property
    def mesh_edges(self):
        """List of :c:`MeshEdge2D` defining non-boundary edges to be
        preserved in mesh generation for the :c:`PolyMesh2D`.

        Returns
        -------
        `list` of :c:`MeshEdge2D`
            The list of :c:`MeshEdge2D` in the :c:`PolyMesh2D`.

        Note
        ----
        The list of :a:`mesh_edges` is not intended to be directly mutable.
        Instead modify it using the :m:`add_mesh_edge` and
        :m:`remove_mesh_edge` methods. New :c:`MeshEdge2D` objects require a
        parent mesh to be set, and by default will be added to that parent
        mesh.

        Examples
        --------
        >>> # initialize a mesh, no initial properties provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.mesh_edges)
        []

        >>> # add some vertices to the mesh and add a mesh edge
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> bnd_verts = [k for k, _ in enumerate(msh.vertices)]
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> msh.add_vertices([[0.25, 0.25], [0.75, 0.75]])
        >>> print(msh.vertices)
        [[0.   0.  ]
         [0.   1.  ]
         [1.   1.  ]
         [1.   0.  ]
         [0.25 0.25]
         [0.75 0.75]]
        >>> me = vcfempy.meshgen.MeshEdge2D(msh, [4, 5])
        >>> print(msh.num_mesh_edges)
        1
        >>> for me in msh.mesh_edges:
        ...     print(me.vertices)
        [4, 5]
        >>> print(msh.vertices[msh.mesh_edges[0].vertices, :])
        [[0.25 0.25]
         [0.75 0.75]]

        >>> # add two more mesh edges
        >>> # mesh edges can overlap the boundaries
        >>> msh.add_vertices([[0.5, 0.75], [1, 1.25],
        ...                   [0.5, 0.25], [0, -0.25]])
        >>> print(msh.vertices)
        [[ 0.    0.  ]
         [ 0.    1.  ]
         [ 1.    1.  ]
         [ 1.    0.  ]
         [ 0.25  0.25]
         [ 0.75  0.75]
         [ 0.5   0.75]
         [ 1.    1.25]
         [ 0.5   0.25]
         [ 0.   -0.25]]
        >>> me_new = [vcfempy.meshgen.MeshEdge2D(msh, [k, k+1])
        ...                                      for k in [6, 8]]
        >>> print(msh.num_mesh_edges)
        3
        >>> for me in msh.mesh_edges:
        ...     print(me.vertices)
        [4, 5]
        [6, 7]
        [8, 9]
        >>> for k, me in enumerate(msh.mesh_edges):
        ...     print(f'Mesh edge {k}')
        ...     print(msh.vertices[me.vertices, :])
        ...     print()
        Mesh edge 0
        [[0.25 0.25]
         [0.75 0.75]]
        <BLANKLINE>
        Mesh edge 1
        [[0.5  0.75]
         [1.   1.25]]
        <BLANKLINE>
        Mesh edge 2
        [[ 0.5   0.25]
         [ 0.   -0.25]]
        <BLANKLINE>
        """
        return self._mesh_edges

    def add_mesh_edge(self, mesh_edge):
        """Add a :c:`MeshEdge2D` to the :c:`PolyMesh2D`.

        Parameters
        ----------
        mesh_edge : :c:`MeshEdge2D`
            :c:`MeshEdge2D` to add to the :c:`PolyMesh2D`.

        Raises
        ------
        TypeError
            If **mesh_edge** is not a :c:`MeshEdge2D`.
        ValueError
            If **mesh_edge** is already in :a:`mesh_edges` or does not have
            this :c:`PolyMesh2D` as its parent.

        Note
        ----
        It is not normally necessary to call :m:`add_mesh_edge` when
        creating a new :c:`MeshEdge2D` since it will add itself to the
        parent :c:`PolyMesh2D` by default. This is only necessary if the
        :c:`MeshEdge2D` was created with **add_to_mesh** = ``False`` or
        if the :c:`MeshEdge2D` was previously removed from the
        :c:`PolyMesh2D` using :m:`remove_mesh_edge`.

        Examples
        --------
        >>> # create a mesh and a mesh edge, this adds the mesh edge by default
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> me = vcfempy.meshgen.MeshEdge2D(msh)
        >>> print(me.mesh.name)
        test mesh
        >>> print(me in msh.mesh_edges)
        True

        >>> # create another mesh edge, but do not add it to the mesh
        >>> me_new = vcfempy.meshgen.MeshEdge2D(msh, add_to_mesh=False)
        >>> print(me_new.mesh.name)
        test mesh
        >>> print(me_new in msh.mesh_edges)
        False

        >>> # add the new mesh edge to its parent mesh
        >>> msh.add_mesh_edge(me_new)
        >>> print(me_new in msh.mesh_edges)
        True

        >>> # try to add invalid mesh edges
        >>> msh.add_mesh_edge(1)
        Traceback (most recent call last):
            ...
        TypeError: mesh edge not vcfempy.meshgen.MeshEdge2D
        >>> msh.add_mesh_edge(me)
        Traceback (most recent call last):
            ...
        ValueError: mesh edge already in list
        >>> new_msh = vcfempy.meshgen.PolyMesh2D()
        >>> me_new = vcfempy.meshgen.MeshEdge2D(new_msh)
        >>> msh.add_mesh_edge(me_new)
        Traceback (most recent call last):
            ...
        ValueError: mesh edge does not have self as mesh
        """
        if not isinstance(mesh_edge, MeshEdge2D):
            raise TypeError('mesh edge not vcfempy.meshgen.MeshEdge2D')
        if mesh_edge in self.mesh_edges:
            raise ValueError('mesh edge already in list')
        if mesh_edge.mesh is not self:
            raise ValueError('mesh edge does not have self as mesh')
        self.mesh_edges.append(mesh_edge)
        self.mesh_valid = False

    def remove_mesh_edge(self, mesh_edge):
        """Remove a :c:`MeshEdge2D` from the :c:`PolyMesh2D`.

        Parameters
        ----------
        mesh_edge : :c:`MeshEdge2D`
            :c:`MeshEdge2D` to remove from the :c:`PolyMesh2D`.

        Raises
        ------
        ValueError
            If **mesh_edge** is not in :a:`mesh_edges`.

        Note
        ----
        When removing a mesh edge from the :c:`PolyMesh2D`, the
        :a:`MeshEdge2D.mesh` is not changed, and it can be added again
        using :m:`add_mesh_edge` if desired.

        Examples
        --------
        >>> # create a mesh and a mesh edge, then remove it
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> me = vcfempy.meshgen.MeshEdge2D(msh)
        >>> msh.remove_mesh_edge(me)
        >>> print(msh.mesh_edges)
        []

        >>> # try to remove a mesh edge that is not in the mesh
        >>> msh.remove_mesh_edge(me)
        Traceback (most recent call last):
            ...
        ValueError: list.remove(x): x not in list
        """
        self.mesh_edges.remove(mesh_edge)
        self.mesh_valid = False

    @property
    def num_seed_points(self):
        return len(self.seed_points)

    @property
    def seed_points(self):
        return self._seed_points

    def add_seed_points(self, seed_points):
        if seed_points is None or len(seed_points) == 0:
            return
        seed_points = np.array(seed_points, dtype=float)
        self._seed_points = np.vstack([self.seed_points, seed_points])
        self.mesh_valid = False

    def delete_seed_points(self, del_points):
        # pre-process del_points: flatten, check for invalid indices, and
        # eliminate duplicates
        if del_points is None:
            return
        del_points = np.array(del_points, dtype=int, ndmin=1)
        if len(del_points) == 0:
            return
        del_points = np.where(del_points < 0,
                              del_points + self.num_seed_points,
                              del_points)
        del_points = np.unique(del_points)
        if (np.any(del_points < 0)
                or np.any(del_points >= self.num_seed_points)):
            raise IndexError('at least one index out of range')

        # delete the points and invalidate the mesh
        self._seed_points = np.delete(self._seed_points, del_points, 0)
        self.mesh_valid = False

    @property
    def points(self):
        """Array of seed point coordinates for mesh generation of the
        :c:`PolyMesh2D`.

        Returns
        -------
        `numpy.ndarray`, shape = (:a:`num_elements`, 2)
            The array of seed point coordinates in the :c:`PolyMesh2D`.

        Note
        ----
        The :a:`points` property is not intended to be directly mutable,
        rather :a:`points` are generated by :m:`generate_mesh`. If the mesh
        is reset (by setting :a:`mesh_valid` to ``False``, or by changing
        a property that affects the mesh validity), then the :a:`points` will
        be cleared. The number of :a:`points` will always be
        :a:`num_elements`.

        Examples
        --------
        >>> # create a mesh and add some vertices, but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.num_elements)
        0
        >>> print(msh.points)
        []

        >>> # generate a simple mesh
        >>> # note: num_elements == len(msh.points)
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.num_elements)
        9
        >>> print(msh.points)
        [[0.5 0.5]
         [0.2 0.2]
         [0.2 0.8]
         [0.8 0.8]
         [0.8 0.2]
         [0.2 0.5]
         [0.5 0.8]
         [0.8 0.5]
         [0.5 0.2]]

        >>> # explicitly resetting the mesh clears the seed points
        >>> msh.mesh_valid = False
        >>> print(msh.num_elements)
        0
        >>> print(msh.points)
        []

        >>> # regenerate the mesh
        >>> msh.generate_mesh()
        >>> print(msh.num_elements)
        9
        >>> print(msh.points)
        [[0.5 0.5]
         [0.2 0.2]
         [0.2 0.8]
         [0.8 0.8]
         [0.8 0.2]
         [0.2 0.5]
         [0.5 0.8]
         [0.8 0.5]
         [0.5 0.2]]

        >>> # adding a boundary vertex also resets the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.points)
        []
        """
        return self._points

    @property
    def num_nodes(self):
        """Number of :a:`nodes` in the generated mesh of the :c:`PolyMesh2D`.

        Returns
        -------
        `int`
            The number of :a:`nodes` in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # create a mesh and add some vertices, but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.num_nodes)
        0

        >>> # generate a simple mesh
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.num_nodes)
        16

        >>> # explicitly resetting the mesh clears the nodes
        >>> msh.mesh_valid = False
        >>> print(msh.num_nodes)
        0

        >>> # regenerate the mesh
        >>> msh.generate_mesh()
        >>> print(msh.num_nodes)
        16

        >>> # adding a boundary vertex also resets the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.num_nodes)
        0
        """
        return len(self.nodes)

    @property
    def nodes(self):
        """Array of node coordinates defining the generated mesh geometry of
        the :c:`PolyMesh2D`.

        Returns
        -------
        `numpy.ndarray`, shape = (:a:`num_nodes`, 2)
            Array of node coordinates defining the mesh of the
            :c:`PolyMesh2D`.

        Note
        ----
        The :a:`nodes` property is not intended to be directly mutable,
        rather :a:`nodes` are generated by :m:`generate_mesh`. If the mesh is
        reset (by setting :a:`mesh_valid` to ``False``, or by changing a
        property that affects the mesh validity), then the :a:`nodes` will be
        cleared.

        Examples
        --------
        >>> # create a mesh and add some vertices, but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.nodes)
        []

        >>> # generate a simple mesh
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.nodes.round(14))
        [[-0.    1.  ]
         [ 0.35  1.  ]
         [ 0.    0.  ]
         [ 0.35  0.  ]
         [ 1.    1.  ]
         [ 0.65  1.  ]
         [ 0.65  0.65]
         [ 1.    0.65]
         [ 1.   -0.  ]
         [ 0.65 -0.  ]
         [ 0.65  0.35]
         [ 1.    0.35]
         [ 0.    0.65]
         [ 0.    0.35]
         [ 0.35  0.65]
         [ 0.35  0.35]]

        >>> # explicitly resetting the mesh clears the nodes
        >>> msh.mesh_valid = False
        >>> print(msh.nodes)
        []

        >>> # regenerate the mesh
        >>> msh.generate_mesh()
        >>> print(msh.nodes.round(14))
        [[-0.    1.  ]
         [ 0.35  1.  ]
         [ 0.    0.  ]
         [ 0.35  0.  ]
         [ 1.    1.  ]
         [ 0.65  1.  ]
         [ 0.65  0.65]
         [ 1.    0.65]
         [ 1.   -0.  ]
         [ 0.65 -0.  ]
         [ 0.65  0.35]
         [ 1.    0.35]
         [ 0.    0.65]
         [ 0.    0.35]
         [ 0.35  0.65]
         [ 0.35  0.35]]

        >>> # adding a boundary vertex also resets the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.nodes)
        []
        """
        return self._nodes

    @property
    def num_elements(self):
        """Number of :a:`elements` in the generated mesh of the
        :c:`PolyMesh2D`.

        Returns
        -------
        `int`
            The number of :a:`elements` in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # create a mesh and add some vertices, but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.num_elements)
        0

        >>> # generate a simple mesh
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.num_elements)
        9

        >>> # explicitly resetting the mesh clears the elements
        >>> msh.mesh_valid = False
        >>> print(msh.num_elements)
        0

        >>> # regenerate the mesh
        >>> msh.generate_mesh()
        >>> print(msh.num_elements)
        9

        >>> # adding a boundary vertex also resets the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.num_elements)
        0
        """
        return len(self.elements)

    @property
    def elements(self):
        """List of :c:`PolyElement2D` elements in the generated mesh for
        the :c:`PolyMesh2D`.

        Returns
        -------
        `list` of :c:`PolyElement2D`
            The list of elements in the :c:`PolyMesh2D`.

        Note
        ----
        The :a:`elements` property is not intended to be directly mutable,
        rather :a:`elements` are generated by :m:`generate_mesh`. If the mesh
        is reset (by setting :a:`mesh_valid` to ``False``, or by changing a
        property that affects the mesh validity), then the :a:`elements` will
        be cleared.

        Examples
        --------
        >>> # create a mesh and add some vertices but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.elements)
        []

        >>> # generate a simple mesh
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.nodes.round(14))
        [[-0.    1.  ]
         [ 0.35  1.  ]
         [ 0.    0.  ]
         [ 0.35  0.  ]
         [ 1.    1.  ]
         [ 0.65  1.  ]
         [ 0.65  0.65]
         [ 1.    0.65]
         [ 1.   -0.  ]
         [ 0.65 -0.  ]
         [ 0.65  0.35]
         [ 1.    0.35]
         [ 0.    0.65]
         [ 0.    0.35]
         [ 0.35  0.65]
         [ 0.35  0.35]]
        >>> for e in msh.elements:
        ...     print(e.nodes)
        [15, 14, 6, 10]
        [15, 3, 2, 13]
        [12, 0, 1, 14]
        [7, 4, 5, 6]
        [11, 8, 9, 10]
        [15, 13, 12, 14]
        [5, 1, 14, 6]
        [10, 6, 7, 11]
        [3, 15, 10, 9]

        >>> # explicitly resetting the mesh clears the elements
        >>> msh.mesh_valid = False
        >>> print(msh.elements)
        []

        >>> # regenerate the mesh
        >>> msh.generate_mesh()
        >>> for e in msh.elements:
        ...     print(e.nodes)
        [15, 14, 6, 10]
        [15, 3, 2, 13]
        [12, 0, 1, 14]
        [7, 4, 5, 6]
        [11, 8, 9, 10]
        [15, 13, 12, 14]
        [5, 1, 14, 6]
        [10, 6, 7, 11]
        [3, 15, 10, 9]

        >>> # adding a boundary vertex also resets the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.elements)
        []
        """
        return self._elements

    @property
    def num_nodes_per_element(self):
        """Number of nodes per element in the generated mesh for the
        :c:`PolyMesh2D`.

        Returns
        -------
        `list[int]`
            The number of nodes in each element in the :c:`PolyMesh2D`.

        Note
        ----
        The len(:a:`num_nodes_per_element`) will always be the same as
        :a:`num_elements`.

        Examples
        --------
        >>> # initialize a mesh, no initial information provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.num_nodes_per_element)
        []

        >>> # add some vertices and boundary vertices to the mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, [k for k, _
        ...                                  in enumerate(msh.vertices)])

        >>> # generate a simple mesh
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> for e in msh.elements:
        ...     print(e.nodes)
        [15, 14, 6, 10]
        [15, 3, 2, 13]
        [12, 0, 1, 14]
        [7, 4, 5, 6]
        [11, 8, 9, 10]
        [15, 13, 12, 14]
        [5, 1, 14, 6]
        [10, 6, 7, 11]
        [3, 15, 10, 9]
        >>> print(msh.num_nodes_per_element)
        [4, 4, 4, 4, 4, 4, 4, 4, 4]

        >>> # changing the boundary geometry clears the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.num_nodes_per_element)
        []
        """
        return [e.num_nodes for e in self.elements]

    @property
    def element_materials(self):
        """List of :c:`vcfempy.materials.Material` types assigned to each
        element in the generated mesh for the :c:`PolyMesh2D`.

        Returns
        -------
        `list` of :c:`vcfempy.materials.Material`
            The list of material types assigned to each element in the
            :c:`PolyMesh2D`.

        Note
        ----
        The len(:a:`element_materials`) will always be :a:`num_elements`.

        Examples
        --------
        >>> # initialize a mesh, no mesh generated yet, so list of element
        >>> # materials is empty
        >>> # material regions can be overlapping and need not be entirely
        >>> # inside the boundaries; to ensure all elements get assigned
        >>> # materials, it is better to make the regions overlap, with
        >>> # later assigned materials overwriting earlier
        >>> import vcfempy.meshgen
        >>> import vcfempy.materials
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, [k for k, _
        ...                                  in enumerate(msh.vertices)])
        >>> msh.add_vertices([[0, 0.5], [1, 0.5]])
        >>> rock = vcfempy.materials.Material('rock')
        >>> sand = vcfempy.materials.Material('sand')
        >>> mr_rock = vcfempy.meshgen.MaterialRegion2D(msh,
        ...                                            [0, 1, 2, 3],
        ...                                            rock)
        >>> mr_sand = vcfempy.meshgen.MaterialRegion2D(msh,
        ...                                            [4, 1, 2, 5],
        ...                                            sand)
        >>> print(msh.element_materials)
        []

        >>> # generate a simple mesh
        >>> # now materials will be assigned to elements
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> for m in msh.element_materials:
        ...     print(m.name)
        rock
        rock
        sand
        sand
        rock
        rock
        sand
        rock
        rock

        >>> # changing the boundary geometry clears the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.element_materials)
        []
        """
        return [e.material for e in self.elements]

    @property
    def element_areas(self):
        """Array of areas of the :a:`elements` in the :c:`PolyMesh2D`.

        Returns
        -------
        `numpy.ndarray`, size = (:a:`num_elements`, )
            The array of areas of the :a:`elements` in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # initialize a mesh, no initial information provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.element_areas)
        []

        >>> # add some vertices and boundary vertices to the mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, [k for k, _
        ...                                  in enumerate(msh.vertices)])
        >>> # still no element areas
        >>> # because the mesh has not been generated
        >>> print(msh.element_areas)
        []

        >>> # generate a simple mesh
        >>> # now element areas can be calculated
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.element_areas.round(14))
        [0.09   0.1225 0.1225 0.1225 0.1225 0.105  0.105  0.105  0.105 ]

        >>> # notice that element areas sum to the area of the boundary
        >>> import numpy as np
        >>> import shapely.geometry as shp
        >>> print(np.sum(msh.element_areas))
        1.0
        >>> print(shp.Polygon(msh.vertices).area)
        1.0

        >>> # changing the boundary geometry clears the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.element_areas)
        []
        """
        return np.array([e.area for e in self.elements])

    @property
    def element_centroids(self):
        """Array of centroid coordinates for the :a:`elements` in the
        :c:`PolyMesh2D`.

        Returns
        -------
        `numpy.ndarray`, shape = (:a:`num_elements`, 2)

        Examples
        --------
        >>> # initialize a mesh, no initial information provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.element_centroids)
        []

        >>> # add some vertices and boundary vertices to the mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, [k for k, _
        ...                                  in enumerate(msh.vertices)])
        >>> # still no element centroids
        >>> # because the mesh has not been generated
        >>> print(msh.element_centroids)
        []

        >>> # generate a simple mesh
        >>> # now element centroids can be calculated
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.element_centroids.round(14))
        [[0.5   0.5  ]
         [0.175 0.175]
         [0.175 0.825]
         [0.825 0.825]
         [0.825 0.175]
         [0.175 0.5  ]
         [0.5   0.825]
         [0.825 0.5  ]
         [0.5   0.175]]

        >>> # changing the boundary geometry clears the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.element_centroids)
        []
        """
        return np.array([e.centroid for e in self.elements])

    @property
    def element_quad_points(self):
        """List of arrays of quadrature point coordinates for the
        :a:`elements` in the :c:`PolyMesh2D`.

        Returns
        -------
        `list[numpy.ndarray]`
            The list of quadrature point coordinate arrays for the
            :a:`elements` in the :c:`PolyMesh2D`.

        Note
        ----
        These are returned as a list of 2d arrays because the number of
        quadrature points is different for each element, in general, so it
        cannot be converted into a 3d array. This property is mostly provided
        for plotting convenience (e.g. if performing global integrations over
        the whole mesh or plotting the quadrature points for the whole mesh).
        If performing integrations at the element level, it is better to
        access the :a:`PolyElement2D.quad_points` property for each element.

        Examples
        --------
        >>> # initialize a mesh, no initial information provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.element_quad_points)
        []

        >>> # add some vertices and boundary vertices to the mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, [k for k, _
        ...                                  in enumerate(msh.vertices)])
        >>> # still no element quadrature points
        >>> # because the mesh has not been generated
        >>> print(msh.element_quad_points)
        []

        >>> # generate a simple mesh
        >>> # now element quadrature points can be generated
        >>> # notice that the quadrature points are listed in a local
        >>> # coordinate system relative to the element centroid
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> for k, qp in enumerate(msh.element_quad_points):
        ...     print(f'Element {k} quad points, nq{k} = {len(qp)}')
        ...     print(qp.round(14))
        Element 0 quad points, nq0 = 9
        [[-0.1125 -0.1125]
         [-0.1125  0.1125]
         [ 0.1125  0.1125]
         [ 0.1125 -0.1125]
         [-0.075   0.    ]
         [ 0.      0.075 ]
         [ 0.075   0.    ]
         [ 0.     -0.075 ]
         [ 0.      0.    ]]
        Element 1 quad points, nq1 = 9
        [[ 0.13125  0.13125]
         [ 0.13125 -0.13125]
         [-0.13125 -0.13125]
         [-0.13125  0.13125]
         [ 0.0875   0.     ]
         [ 0.      -0.0875 ]
         [-0.0875   0.     ]
         [ 0.       0.0875 ]
         [ 0.       0.     ]]
        Element 2 quad points, nq2 = 9
        [[-0.13125 -0.13125]
         [-0.13125  0.13125]
         [ 0.13125  0.13125]
         [ 0.13125 -0.13125]
         [-0.0875   0.     ]
         [-0.       0.0875 ]
         [ 0.0875   0.     ]
         [ 0.      -0.0875 ]
         [ 0.       0.     ]]
        Element 3 quad points, nq3 = 9
        [[ 0.13125 -0.13125]
         [ 0.13125  0.13125]
         [-0.13125  0.13125]
         [-0.13125 -0.13125]
         [ 0.0875  -0.     ]
         [-0.       0.0875 ]
         [-0.0875  -0.     ]
         [-0.      -0.0875 ]
         [ 0.       0.     ]]
        Element 4 quad points, nq4 = 9
        [[ 0.13125  0.13125]
         [ 0.13125 -0.13125]
         [-0.13125 -0.13125]
         [-0.13125  0.13125]
         [ 0.0875  -0.     ]
         [ 0.      -0.0875 ]
         [-0.0875  -0.     ]
         [ 0.       0.0875 ]
         [ 0.       0.     ]]
        Element 5 quad points, nq5 = 9
        [[ 0.13125 -0.1125 ]
         [-0.13125 -0.1125 ]
         [-0.13125  0.1125 ]
         [ 0.13125  0.1125 ]
         [ 0.      -0.075  ]
         [-0.0875  -0.     ]
         [ 0.       0.075  ]
         [ 0.0875  -0.     ]
         [ 0.       0.     ]]
        Element 6 quad points, nq6 = 9
        [[ 0.1125   0.13125]
         [-0.1125   0.13125]
         [-0.1125  -0.13125]
         [ 0.1125  -0.13125]
         [-0.       0.0875 ]
         [-0.075   -0.     ]
         [ 0.      -0.0875 ]
         [ 0.075   -0.     ]
         [ 0.       0.     ]]
        Element 7 quad points, nq7 = 9
        [[-0.13125 -0.1125 ]
         [-0.13125  0.1125 ]
         [ 0.13125  0.1125 ]
         [ 0.13125 -0.1125 ]
         [-0.0875  -0.     ]
         [ 0.       0.075  ]
         [ 0.0875  -0.     ]
         [ 0.      -0.075  ]
         [ 0.       0.     ]]
        Element 8 quad points, nq8 = 9
        [[-0.1125  -0.13125]
         [-0.1125   0.13125]
         [ 0.1125   0.13125]
         [ 0.1125  -0.13125]
         [-0.075    0.     ]
         [-0.       0.0875 ]
         [ 0.075    0.     ]
         [-0.      -0.0875 ]
         [ 0.       0.     ]]

        >>> # changing the boundary geometry clears the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.element_quad_points)
        []
        """
        return [e.quad_points for e in self.elements]

    @property
    def element_quad_weights(self):
        """List of arrays of quadrature point weights for the :a:`elements`
        in the :c:`PolyMesh2D`.

        Returns
        -------
        `list[numpy.ndarray]`
            The list of quadrature point weight arrays for the :a:`elements`
            in the :c:`PolyMesh2D`.

        Note
        ----
        These are returned as a list of 1d arrays because the number of
        quadrature points is different for each element, in general, so it
        cannot be converted into a 2d array. This property is mostly provided
        for plotting convenience (e.g. if performing global integrations over
        the whole mesh or plotting the quadrature points for the whole mesh).
        If performing integrations at the element level, it is better to
        access the :a:`PolyElement2D.quad_weights` property for each element.

        Examples
        --------
        >>> # initialize a mesh, no initial information provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.element_quad_weights)
        []

        >>> # add some vertices and boundary vertices to the mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, [k for k, _
        ...                                  in enumerate(msh.vertices)])
        >>> # still no element quadrature points
        >>> # because the mesh has not been generated
        >>> print(msh.element_quad_weights)
        []

        >>> # generate a simple mesh
        >>> # now element quadrature points can be generated
        >>> # notice that the quadrature points are listed in a local
        >>> # coordinate system relative to the element centroid
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> for k, qw in enumerate(msh.element_quad_weights):
        ...     print(f'Element {k} quad weights, nq{k} = {len(qw)}')
        ...     print(qw)
        Element 0 quad weights, nq0 = 9
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        Element 1 quad weights, nq1 = 9
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        Element 2 quad weights, nq2 = 9
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        Element 3 quad weights, nq3 = 9
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        Element 4 quad weights, nq4 = 9
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        Element 5 quad weights, nq5 = 9
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        Element 6 quad weights, nq6 = 9
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        Element 7 quad weights, nq7 = 9
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        Element 8 quad weights, nq8 = 9
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]

        >>> # changing the boundary geometry clears the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.element_quad_weights)
        []
        """
        return [e.quad_weights for e in self.elements]

    @property
    def num_interface_elements(self):
        """Number of :a:`interface_elements` in the generated mesh for a
        :c:`PolyMesh2D`.

        Returns
        -------
        `int`
            The number of :a:`interface_elements` in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # create a mesh and add some vertices but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.num_interface_elements)
        0

        >>> # generate a simple mesh
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.num_interface_elements)
        12

        >>> # explicitly resetting the mesh clears the interface elements
        >>> msh.mesh_valid = False
        >>> print(msh.num_interface_elements)
        0

        >>> # regenerate the mesh
        >>> msh.generate_mesh()
        >>> print(msh.num_interface_elements)
        12

        >>> # adding a boundary vertex also resets the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.num_interface_elements)
        0
        """
        return len(self.interface_elements)

    @property
    def interface_elements(self):
        """List of :c:`InterfaceElement2D` in the generated mesh for a
        :c:`PolyMesh2D`.

        Returns
        -------
        `list` of :c:`InterfaceElement2D`
            The list of interface elements in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # create a mesh and add some vertices but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.interface_elements)
        []

        >>> # generate a simple mesh and print the interface elements
        >>> # notice that interface node indices are all < msh.num_nodes
        >>> # and the neighbor element indices are all < msh.num_elements
        >>> # also note that interface elements all include at least one
        >>> # node that is not on the boundary
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.num_nodes)
        16
        >>> print(msh.nodes.round(14))
        [[-0.    1.  ]
         [ 0.35  1.  ]
         [ 0.    0.  ]
         [ 0.35  0.  ]
         [ 1.    1.  ]
         [ 0.65  1.  ]
         [ 0.65  0.65]
         [ 1.    0.65]
         [ 1.   -0.  ]
         [ 0.65 -0.  ]
         [ 0.65  0.35]
         [ 1.    0.35]
         [ 0.    0.65]
         [ 0.    0.35]
         [ 0.35  0.65]
         [ 0.35  0.35]]
        >>> for e in msh.interface_elements:
        ...     print(e.nodes)
        [5, 6]
        [6, 7]
        [9, 10]
        [10, 11]
        [6, 10]
        [12, 14]
        [13, 15]
        [14, 15]
        [1, 14]
        [3, 15]
        [6, 14]
        [10, 15]
        >>> print(msh.num_elements)
        9
        >>> for e in msh.interface_elements:
        ...     print([msh.elements.index(n) for n in e.neighbors])
        [3, 6]
        [3, 7]
        [4, 8]
        [4, 7]
        [7, 0]
        [5, 2]
        [5, 1]
        [5, 0]
        [2, 6]
        [1, 8]
        [0, 6]
        [0, 8]
        >>> # explicitly resetting the mesh clears the interface elements
        >>> msh.mesh_valid = False
        >>> print(msh.interface_elements)
        []

        >>> # regenerate the mesh
        >>> msh.generate_mesh()
        >>> for e in msh.interface_elements:
        ...     print(e.nodes)
        [5, 6]
        [6, 7]
        [9, 10]
        [10, 11]
        [6, 10]
        [12, 14]
        [13, 15]
        [14, 15]
        [1, 14]
        [3, 15]
        [6, 14]
        [10, 15]
        >>> for e in msh.interface_elements:
        ...     print([msh.elements.index(n) for n in e.neighbors])
        [3, 6]
        [3, 7]
        [4, 8]
        [4, 7]
        [7, 0]
        [5, 2]
        [5, 1]
        [5, 0]
        [2, 6]
        [1, 8]
        [0, 6]
        [0, 8]

        >>> # adding a boundary vertex also resets the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.interface_elements)
        []
        """
        return self._interface_elements

    @property
    def num_boundary_elements(self):
        """Number of :a:`boundary_elements` in the generated mesh for a
        :c:`PolyMesh2D`.

        Returns
        -------
        `int`
            The number of :a:`boundary_elements` in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # create a mesh and add some vertices but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.num_boundary_elements)
        0

        >>> # generate a simple mesh
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.num_boundary_elements)
        12

        >>> # explicitly resetting the mesh clears the interface elements
        >>> msh.mesh_valid = False
        >>> print(msh.num_boundary_elements)
        0

        >>> # regenerate the mesh
        >>> msh.generate_mesh()
        >>> print(msh.num_boundary_elements)
        12

        >>> # adding a boundary vertex also resets the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.num_boundary_elements)
        0
        """
        return len(self.boundary_elements)

    @property
    def boundary_elements(self):
        """List of :c:`BoundaryElement2D` in the generated mesh for a
        :c:`PolyMesh2D`.

        Returns
        -------
        `list` of :c:`BoundaryElement2D`
            The list of boundary elements in the :c:`PolyMesh2D`.

        Examples
        --------
        >>> # create a mesh and add some vertices but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.boundary_elements)
        []

        >>> # generate a simple mesh and print the boundary elements
        >>> # notice that boundary node indices are all < msh.num_nodes
        >>> # and the neighbor element indices are all < msh.num_elements
        >>> # also note that boundary elements all have both nodes on the
        >>> # analysis boundaries
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.num_nodes)
        16
        >>> print(msh.nodes.round(14))
        [[-0.    1.  ]
         [ 0.35  1.  ]
         [ 0.    0.  ]
         [ 0.35  0.  ]
         [ 1.    1.  ]
         [ 0.65  1.  ]
         [ 0.65  0.65]
         [ 1.    0.65]
         [ 1.   -0.  ]
         [ 0.65 -0.  ]
         [ 0.65  0.35]
         [ 1.    0.35]
         [ 0.    0.65]
         [ 0.    0.35]
         [ 0.35  0.65]
         [ 0.35  0.35]]
        >>> for e in msh.boundary_elements:
        ...     print(e.nodes)
        [0, 1]
        [2, 3]
        [4, 5]
        [4, 7]
        [1, 5]
        [8, 9]
        [8, 11]
        [7, 11]
        [3, 9]
        [12, 13]
        [0, 12]
        [2, 13]
        >>> print(msh.num_elements)
        9
        >>> print([msh.elements.index(e.neighbor)
        ...        for e in msh.boundary_elements])
        [2, 1, 3, 3, 6, 4, 4, 7, 8, 5, 2, 1]

        >>> # explicitly resetting the mesh clears the boundary elements
        >>> msh.mesh_valid = False
        >>> print(msh.boundary_elements)
        []

        >>> # regenerate the mesh
        >>> msh.generate_mesh()
        >>> for e in msh.boundary_elements:
        ...     print(e.nodes)
        [0, 1]
        [2, 3]
        [4, 5]
        [4, 7]
        [1, 5]
        [8, 9]
        [8, 11]
        [7, 11]
        [3, 9]
        [12, 13]
        [0, 12]
        [2, 13]
        >>> print([msh.elements.index(e.neighbor)
        ...        for e in msh.boundary_elements])
        [2, 1, 3, 3, 6, 4, 4, 7, 8, 5, 2, 1]

        >>> # adding a boundary vertex also resets the mesh
        >>> msh.add_vertices([1.5, 0.5])
        >>> msh.insert_boundary_vertices(3, 4)
        >>> print(msh.boundary_elements)
        []
        """
        return self._boundary_elements

    @property
    def mesh_valid(self):
        """Flag for whether there is a valid generated mesh for the
        :c:`PolyMesh2D`.

        Parameters
        ----------
        flag : bool_like
            The new value of the :a:`mesh_valid` flag.

        Returns
        -------
        `bool`
            The value of the :a:`mesh_valid` flag.

        Raises
        ------
        `ValueError`
            If the value of **flag** cannot be converted to a `bool`.
            If **flag** is ``True``-like, but mesh properties are not set or
            are invalid or inconsistent.

        Note
        ----
        If setting to ``False``, mesh properties are reset. If setting to
        ``True``, basic checks of mesh validity are performed before setting
        the value. `str` values that can be cast to `float` are considered
        ``True``-like if non-zero and ``False``-like if zero. If the `str`
        cannot be cast to `float`, then the values 'y', 'yes', 't', 'true',
        and 'on' (case insensitive) are converted to ``True`` and the values
        'n', 'no', 'f', 'false', and 'off' are converted to ``False``. Other
        `str` values raise a `ValueError`. In general, directly setting
        :a:`mesh_valid` to ``False`` is a way to explicitly clear the mesh,
        but setting :a:`mesh_valid` to ``True`` should only be done
        indirectly using the :m:`generate_mesh` method, otherwise it is
        likely that a `ValueError` will be raised due to invalid mesh
        properties.

        Examples
        --------
        >>> # create a mesh and add some vertices but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.mesh_valid)
        False

        >>> # generate a simple mesh, which sets mesh_valid as a side effect
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.mesh_valid)
        True
        >>> print(msh.num_elements)
        9

        >>> # explicitly resetting the mesh with a bool value
        >>> msh.mesh_valid = False
        >>> print(msh.mesh_valid)
        False
        >>> print(msh.num_elements)
        0

        >>> # regenerate the mesh
        >>> msh.generate_mesh()
        >>> print(msh.mesh_valid)
        True

        >>> # resetting the mesh with a False-like string
        >>> msh.mesh_valid = 'no'
        >>> print(msh.mesh_valid)
        False
        >>> print(msh.num_elements)
        0

        >>> # attempting to set mesh_valid to True-like value, but mesh has
        >>> # not been generated
        >>> msh.mesh_valid = 'yes'
        Traceback (most recent call last):
        ...
        ValueError: trying to set PolyMesh2D.mesh_valid = True, but \
self.nodes is empty
        >>> print(msh.mesh_valid)
        False

        >>> # attempting to set mesh_valid to non-truth-like str value
        >>> msh.mesh_valid = 'dslk'
        Traceback (most recent call last):
        ...
        ValueError: invalid truth value 'dslk'
        """
        return self._mesh_valid

    @mesh_valid.setter
    def mesh_valid(self, flag):
        # try to cast flag to bool
        # will raise a ValueError if this does not work
        if isinstance(flag, (str, np.str_)):
            try:
                flag = float(flag)
            except ValueError:
                flag = distutils.util.strtobool(flag)
        flag = bool(flag)

        # if invalidating mesh,
        # then reset mesh properties
        if not flag:
            self._mesh_valid = False
            self._nodes = np.empty((0, 2))
            self._points = np.empty((0, 2))
            self._elements = []
            self._interface_elements = []
            self._boundary_elements = []
        # otherwise, trying to validate mesh
        # check that mesh properties have been set
        else:
            if not self.num_nodes:
                raise ValueError('trying to set PolyMesh2D.mesh_valid '
                                 + '= True, but self.nodes is empty')
            if not self.num_elements:
                raise ValueError('trying to set PolyMesh2D.mesh_valid '
                                 + '= True, but self.elements is empty')
            if len(self.points) != self.num_elements:
                raise ValueError('trying to set PolyMesh2D.mesh_valid '
                                 + '= True, but len(self.points) '
                                 + '!= self.num_elements')
            if (not self.num_interface_elements
                    and not self.num_boundary_elements):
                raise ValueError('trying to set PolyMesh2D.mesh_valid '
                                 + '= True, but self.interface_elements and '
                                 + 'self.boundary_elements are both empty')
            # if here, then all checks for mesh validity succeeded
            # set the mesh valid flag
            self._mesh_valid = True

    @property
    def high_order_quadrature(self):
        """Flag for whether high order quadrature will be used by the
        :a:`elements` of the generated mesh for the :c:`PolyMesh2D`.

        Parameters
        ----------
        flag : bool_like
            The new value of the :a:`high_order_quadrature` flag.

        Returns
        -------
        `bool`
            The value of the :a:`high_order_quadrature` flag.

        Raises
        ------
        `ValueError`
            If the value of **flag** cannot be converted to a `bool`.

        Note
        ----
        Setting :a:`high_order_quadrature` for the :c:`PolyMesh2D` will clear
        previously generated :a:`quad_points`, :a:`quad_weights`, and
        :a:`quad_integrals` for :a:`elements` in the :c:`PolyMesh2D`,
        regardless of the previous value. `str` values that can be cast to
        `float` are considered ``True``-like if non-zero and ``False``-like
        if zero. If the `str` cannot be cast to `float`, then the values 'y',
        'yes', 't', 'true', and 'on' (case insensitive) are converted to
        ``True`` and the values 'n', 'no', 'f', 'false', and 'off' are
        converted to ``False``. Other `str` values raise a `ValueError`.

        Examples
        --------
        >>> # create a mesh and add some vertices but no mesh generated yet
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.high_order_quadrature)
        False

        >>> # generate a simple mesh
        >>> # elements will use minimum order quadrature
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> for k, qp in enumerate(msh.element_quad_points):
        ...     print(f'Element {k} quad points, nq{k} = {len(qp)}')
        ...     print(qp.round(14)) #doctest: +ELLIPSIS
        Element 0 quad points, nq0 = 9
        [[-0.1125 -0.1125]
         [-0.1125  0.1125]
         ...
         [ 0.     -0.075 ]
         [ 0.      0.    ]]
        Element 1 quad points, nq1 = 9
        [[ 0.13125  0.13125]
         [ 0.13125 -0.13125]
         ...
         [ 0.       0.0875 ]
         [ 0.       0.     ]]
        Element 2 quad points, nq2 = 9
        [[-0.13125 -0.13125]
         [-0.13125  0.13125]
         ...
         [ 0.      -0.0875 ]
         [ 0.       0.     ]]
        Element 3 quad points, nq3 = 9
        [[ 0.13125 -0.13125]
         [ 0.13125  0.13125]
         ...
         [-0.      -0.0875 ]
         [ 0.       0.     ]]
        Element 4 quad points, nq4 = 9
        [[ 0.13125  0.13125]
         [ 0.13125 -0.13125]
         ...
         [ 0.       0.0875 ]
         [ 0.       0.     ]]
        Element 5 quad points, nq5 = 9
        [[ 0.13125 -0.1125 ]
         [-0.13125 -0.1125 ]
         ...
         [ 0.0875  -0.     ]
         [ 0.       0.     ]]
        Element 6 quad points, nq6 = 9
        [[ 0.1125   0.13125]
         [-0.1125   0.13125]
         ...
         [ 0.075   -0.     ]
         [ 0.       0.     ]]
        Element 7 quad points, nq7 = 9
        [[-0.13125 -0.1125 ]
         [-0.13125  0.1125 ]
         ...
         [ 0.      -0.075  ]
         [ 0.       0.     ]]
        Element 8 quad points, nq8 = 9
        [[-0.1125  -0.13125]
         [-0.1125   0.13125]
         ...
         [-0.      -0.0875 ]
         [ 0.       0.     ]]

        >>> # switch to high order quadrature
        >>> # no need to regenerate mesh, element quadrature will be reset
        >>> msh.high_order_quadrature = True
        >>> print(msh.high_order_quadrature)
        True
        >>> for k, qp in enumerate(msh.element_quad_points):
        ...     print(f'Element {k} quad points, nq{k} = {len(qp)}')
        ...     print(qp.round(14)) #doctest: +ELLIPSIS
        Element 0 quad points, nq0 = 17
        [[-0.1275 -0.1275]
         [-0.1275  0.1275]
         ...
         [ 0.     -0.051 ]
         [ 0.      0.    ]]
        Element 1 quad points, nq1 = 17
        [[ 0.14875  0.14875]
         [ 0.14875 -0.14875]
         ...
         [ 0.       0.0595 ]
         [ 0.       0.     ]]
        Element 2 quad points, nq2 = 17
        [[-0.14875 -0.14875]
         [-0.14875  0.14875]
         ...
         [ 0.      -0.0595 ]
         [ 0.       0.     ]]
        Element 3 quad points, nq3 = 17
        [[ 0.14875 -0.14875]
         [ 0.14875  0.14875]
         ...
         [-0.      -0.0595 ]
         [ 0.       0.     ]]
        Element 4 quad points, nq4 = 17
        [[ 0.14875  0.14875]
         [ 0.14875 -0.14875]
         ...
         [ 0.       0.0595 ]
         [ 0.       0.     ]]
        Element 5 quad points, nq5 = 17
        [[ 0.14875 -0.1275 ]
         [-0.14875 -0.1275 ]
         ...
         [ 0.0595  -0.     ]
         [ 0.       0.     ]]
        Element 6 quad points, nq6 = 17
        [[ 0.1275   0.14875]
         [-0.1275   0.14875]
         ...
         [ 0.051   -0.     ]
         [ 0.       0.     ]]
        Element 7 quad points, nq7 = 17
        [[-0.14875 -0.1275 ]
         [-0.14875  0.1275 ]
         ...
         [ 0.      -0.051  ]
         [ 0.       0.     ]]
        Element 8 quad points, nq8 = 17
        [[-0.1275  -0.14875]
         [-0.1275   0.14875]
         ...
         [-0.      -0.0595 ]
         [ 0.       0.     ]]

        >>> # switch back to low order quadrature
        >>> # use a False-like string
        >>> msh.high_order_quadrature = 'off'
        >>> print(msh.high_order_quadrature)
        False
        >>> for k, qp in enumerate(msh.element_quad_points):
        ...     print(f'Element {k} quad points, nq{k} = {len(qp)}')
        ...     print(qp.round(14)) #doctest: +ELLIPSIS
        Element 0 quad points, nq0 = 9
        [[-0.1125 -0.1125]
         [-0.1125  0.1125]
         ...
         [ 0.     -0.075 ]
         [ 0.      0.    ]]
        Element 1 quad points, nq1 = 9
        [[ 0.13125  0.13125]
         [ 0.13125 -0.13125]
         ...
         [ 0.       0.0875 ]
         [ 0.       0.     ]]
        Element 2 quad points, nq2 = 9
        [[-0.13125 -0.13125]
         [-0.13125  0.13125]
         ...
         [ 0.      -0.0875 ]
         [ 0.       0.     ]]
        Element 3 quad points, nq3 = 9
        [[ 0.13125 -0.13125]
         [ 0.13125  0.13125]
         ...
         [-0.      -0.0875 ]
         [ 0.       0.     ]]
        Element 4 quad points, nq4 = 9
        [[ 0.13125  0.13125]
         [ 0.13125 -0.13125]
         ...
         [ 0.       0.0875 ]
         [ 0.       0.     ]]
        Element 5 quad points, nq5 = 9
        [[ 0.13125 -0.1125 ]
         [-0.13125 -0.1125 ]
         ...
         [ 0.0875  -0.     ]
         [ 0.       0.     ]]
        Element 6 quad points, nq6 = 9
        [[ 0.1125   0.13125]
         [-0.1125   0.13125]
         ...
         [ 0.075   -0.     ]
         [ 0.       0.     ]]
        Element 7 quad points, nq7 = 9
        [[-0.13125 -0.1125 ]
         [-0.13125  0.1125 ]
         ...
         [ 0.      -0.075  ]
         [ 0.       0.     ]]
        Element 8 quad points, nq8 = 9
        [[-0.1125  -0.13125]
         [-0.1125   0.13125]
         ...
         [-0.      -0.0875 ]
         [ 0.       0.     ]]

        >>> # attempting to set high_order_quadrature
        >>> # to a non-truth-like str value
        >>> msh.high_order_quadrature = 'dslk'
        Traceback (most recent call last):
        ...
        ValueError: invalid truth value 'dslk'
        """
        return self._high_order_quadrature

    @high_order_quadrature.setter
    def high_order_quadrature(self, flag):
        # try to cast flag to bool
        # will raise a ValueError if this does not work
        if isinstance(flag, (str, np.str_)):
            try:
                flag = float(flag)
            except ValueError:
                flag = distutils.util.strtobool(flag)
        self._high_order_quadrature = bool(flag)
        # assume value changed, reset element quadrature
        for e in self.elements:
            e.invalidate_properties()

    @property
    def mesh_scale(self):
        return self._mesh_scale

    @mesh_scale.setter
    def mesh_scale(self, val):
        self._mesh_scale = float(val)

    @property
    def mesh_rand(self):
        return self._mesh_rand

    @mesh_rand.setter
    def mesh_rand(self, val):
        val = float(val)
        if val < 0.0 or val >= 1.0:
            raise ValueError(f'mesh_rand {val} invalid, should be in [0., 1.)')
        self._mesh_rand = val

    def _check_edge_scale(self, edge_verts):
        d_scale = self.mesh_scale
        for k, v0 in enumerate(self.vertices[edge_verts[:-1]]):
            dd = self.vertices[edge_verts[k+1]] - v0
            dd = 0.4 * np.linalg.norm(dd)
            if dd < d_scale:
                d_scale = dd
        return d_scale

    def _delete_points_near_edge(self, edge_verts, d_scale):
        # no vertices, do not do anything
        if len(edge_verts) == 0:
            return
        # single vertex, just delete points within d_scale
        elif len(edge_verts) == 1:
            v = self.vertices[edge_verts[0]]
            keep_points = np.ones(len(self.points), dtype=bool)
            for j, p in enumerate(self.points):
                d = np.linalg.norm(v - p)
                if d < d_scale:
                    keep_points[j] = False
            self._points = self.points[keep_points]
            return
        # multiple vertices, delete points near each segment
        dv_min = -d_scale
        dd_max = +d_scale
        for k, v in enumerate(self.vertices[edge_verts[:-1]]):
            tt = self.vertices[edge_verts[k+1]] - v
            tt_len = np.linalg.norm(tt)
            tt /= tt_len
            dv_max = tt_len + d_scale
            keep_points = np.ones(len(self.points), dtype=bool)
            for j, p in enumerate(self.points):
                vp = p - v
                pp = v + np.dot(vp, tt) * tt
                dd = np.linalg.norm(p - pp)
                dv = np.sign(np.dot(pp - v, tt)) * np.linalg.norm(pp - v)
                if dd < dd_max and dv >= dv_min and dv <= dv_max:
                    keep_points[j] = False
            self._points = self.points[keep_points]

    def _create_edge_points(self, edge_verts, closed=False):
        # no vertices, do not do anything
        if len(edge_verts) == 0:
            return
        # single vertex, just insert points near the vertex
        elif len(edge_verts) == 1:
            v = self.vertices[edge_verts[0]]
            d_scale = self._check_edge_scale(edge_verts)
            self._delete_points_near_edge(edge_verts, d_scale)
            self._points = np.vstack([self.points,
                                      v + 0.5 * d_scale * np.array([-1, -1]),
                                      v + 0.5 * d_scale * np.array([-1, +1]),
                                      v + 0.5 * d_scale * np.array([+1, -1]),
                                      v + 0.5 * d_scale * np.array([+1, +1])])
            return

        # multiple vertices, create reflections along each segment
        top_points = np.empty((0, 2))
        bot_points = np.empty((0, 2))
        ref_points = np.empty((0, 2))
        # ensure local d_scale is small enough for edge vertex spacing
        # eliminate points near the edge
        check_verts = edge_verts + [edge_verts[0]] if closed else edge_verts
        d_scale = self._check_edge_scale(check_verts)
        self._delete_points_near_edge(check_verts, d_scale)
        # insert points around the edge
        num_verts = len(edge_verts)
        for k, v in enumerate(self.vertices[edge_verts]):
            # first vertex, not closed, insert points before and after
            if not closed and k == 0:
                vp1 = self.vertices[edge_verts[k+1]]
                tt1, nn1 = _get_unit_tangent_normal(v, vp1)
                top_points = np.vstack([top_points,
                                        v + 0.5 * d_scale * (-tt1 - nn1),
                                        v + 0.5 * d_scale * (+tt1 - nn1)])
                bot_points = np.vstack([bot_points,
                                        v + 0.5 * d_scale * (-tt1 + nn1),
                                        v + 0.5 * d_scale * (+tt1 + nn1)])
            # last vertex, not closed, insert points before and after
            # also reflect points before
            elif not closed and k == num_verts - 1:
                vm1 = self.vertices[edge_verts[k-1]]
                tt0, nn0 = tt1, nn1
                # insert points behind
                top_points = np.vstack([top_points,
                                        v + 0.5 * d_scale * (-tt0 - nn0)])
                bot_points = np.vstack([bot_points,
                                        v + 0.5 * d_scale * (-tt0 + nn0)])
                # reflect points behind
                new_ref_points = _get_edge_reflection_points(
                                    bot_points[-2],
                                    bot_points[-1],
                                    vm1, tt0, d_scale, self.mesh_rand)
                ref_points = np.vstack([ref_points, new_ref_points])
                # insert points ahead
                top_points = np.vstack([top_points,
                                        v + 0.5 * d_scale * (+tt0 - nn0)])
                bot_points = np.vstack([bot_points,
                                        v + 0.5 * d_scale * (+tt0 + nn0)])
            # middle vertex (or first/last vertex of a closed edge)
            else:
                vm1 = self.vertices[edge_verts[k-1]]
                vp1 = self.vertices[edge_verts[(k+1) % num_verts]]
                tt0, nn0 = _get_unit_tangent_normal(vm1, v)
                tt1, nn1 = _get_unit_tangent_normal(v, vp1)
                tt_cross = np.cross(-tt0, tt1)
                # straight vertex
                if np.abs(tt_cross) < 1.e-8:
                    # insert points behind
                    top_points = np.vstack([top_points,
                                            v + 0.5 * d_scale * (-tt1 - nn1)])
                    bot_points = np.vstack([bot_points,
                                            v + 0.5 * d_scale * (-tt1 + nn1)])
                    # reflect points behind
                    if k > 0:
                        new_ref_points = _get_edge_reflection_points(
                                            bot_points[-2],
                                            bot_points[-1],
                                            vm1, tt0, d_scale, self.mesh_rand)
                        ref_points = np.vstack([ref_points, new_ref_points])
                    # insert points ahead
                    top_points = np.vstack([top_points,
                                            v + 0.5 * d_scale * (+tt1 - nn1)])
                    bot_points = np.vstack([bot_points,
                                            v + 0.5 * d_scale * (+tt1 + nn1)])
                    # reflect points ahead
                    if k == num_verts - 1:
                        new_ref_points = _get_edge_reflection_points(
                                            top_points[-1],
                                            top_points[0],
                                            v, tt1, d_scale, self.mesh_rand)
                # non-straight vertex
                else:
                    # find intersection point on concave side
                    vv = 0.5 * (-tt0 + tt1)
                    vv_len = np.linalg.norm(vv)
                    vv /= vv_len
                    ss = 0.5 * d_scale * np.cross(nn0, tt0) / np.cross(vv, tt0)
                    # concave vertex, insert bot_point, reflect x2 top_points
                    if tt_cross > 0:
                        new_point = v + ss * vv
                        bot_points = np.vstack([bot_points, new_point])
                        new_top_points = [
                                _reflect_point_across_edge(new_point,
                                                           vm1, tt0),
                                _reflect_point_across_edge(new_point,
                                                           v, tt1)]
                        top_points = np.vstack([top_points, new_top_points])
                        # reflect points behind
                        if k > 0:
                            new_ref_points = _get_edge_reflection_points(
                                    bot_points[-2],
                                    bot_points[-1],
                                    vm1, tt0, d_scale, self.mesh_rand)
                            ref_points = np.vstack([ref_points,
                                                    new_ref_points])
                    # convex vertex, insert top_point, reflect x2 bot_points
                    else:
                        new_point = v - ss * vv
                        top_points = np.vstack([top_points, new_point])
                        new_bot_points = [
                                _reflect_point_across_edge(new_point,
                                                           vm1, tt0),
                                _reflect_point_across_edge(new_point,
                                                           v, tt1)]
                        bot_points = np.vstack([bot_points, new_bot_points])
                        # reflect points behind
                        if k > 0:
                            new_ref_points = _get_edge_reflection_points(
                                    top_points[-2],
                                    top_points[-1],
                                    vm1, tt0, d_scale, self.mesh_rand)
                            ref_points = np.vstack([ref_points,
                                                    new_ref_points])
                    # reflect points ahead
                    if k == num_verts - 1:
                        new_ref_points = _get_edge_reflection_points(
                                top_points[-1],
                                top_points[0],
                                v, tt1, d_scale, self.mesh_rand)
                        ref_points = np.vstack([ref_points, new_ref_points])
        # update global mesh points
        self._points = np.vstack([self.points,
                                  top_points, bot_points, ref_points])

    def generate_mesh(self):
        """ Generate polygonal mesh. """
        # if seed points provided, initialize points with those
        self._points = np.array(self.seed_points)
        # if no seed points, generate seed points on a regular grid
        if not len(self.points):
            xmin = (np.min(self.vertices[self.boundary_vertices, 0])
                    - 2 * self.mesh_scale)
            xmax = (np.max(self.vertices[self.boundary_vertices, 0])
                    + 2 * self.mesh_scale)
            ymin = (np.min(self.vertices[self.boundary_vertices, 1])
                    - 2 * self.mesh_scale)
            ymax = (np.max(self.vertices[self.boundary_vertices, 1])
                    + 2 * self.mesh_scale)
            # get dimensions and number of points for regular grid
            Lx = xmax - xmin
            Ly = ymax - ymin
            nx = int(np.round(Lx / self.mesh_scale)) + 1
            ny = int(np.round(Ly / self.mesh_scale)) + 1
            # generate regular grid and shift points for hexagonal grid
            xc = np.linspace(xmin, xmax, nx)
            yc = np.linspace(ymin, ymax, ny)
            xc, yc = np.meshgrid(xc, yc)
            for k, _ in enumerate(xc):
                xc[k, :] += (-1 if (k % 2) else +1) * 0.25 * self.mesh_scale
            self._points = np.vstack([self.points,
                                      np.vstack([xc.ravel(), yc.ravel()]).T])
            # randomly shift seed points
            if self.mesh_rand:
                rand_shift = (self.mesh_rand * self.mesh_scale
                              * (2. * np.random.random((xc.size, 2)) - 1.))
                self._points += rand_shift
        # generate points for mesh edges
        for edge in self.mesh_edges:
            self._create_edge_points(edge.vertices)
        # eliminate points that are outside the boundaries and mesh boundary
        bpoly = shp.Polygon(self.vertices[self.boundary_vertices])
        point_collection = shp.MultiPoint(self.points).geoms
        in_bnd = np.array([bpoly.contains(x) for x in point_collection])
        self._points = self.points[in_bnd]
        self._create_edge_points(self.boundary_vertices, True)
        # generate Voronoi diagram and eliminate points outside the boundary
        vor = Voronoi(self.points)
        points_to_keep = np.arange(len(self.points))
        point_collection = shp.MultiPoint(self.points).geoms
        in_bnd = np.array([bpoly.contains(x) for x in point_collection])
        self._points = self.points[in_bnd]
        points_to_keep = points_to_keep[in_bnd]
        point_dict = {n: k for k, n in enumerate(points_to_keep)}
        # get elements to keep
        point_elements = np.array(vor.point_region, dtype=int)
        point_elements = point_elements[in_bnd]
        elements = []
        for pe in point_elements:
            elements.append(vor.regions[pe])
        # get edges to keep
        element_edges = []
        element_neighbors = []
        for rp, rv in zip(vor.ridge_points, vor.ridge_vertices):
            if in_bnd[rp[0]] or in_bnd[rp[1]]:
                element_edges.append(rv)
                element_neighbors.append([point_dict[rp[0]]
                                          if in_bnd[rp[0]] else -1,
                                          point_dict[rp[1]]
                                          if in_bnd[rp[1]] else -1])
        # get nodes to keep and correct node indices in elements
        nodes_to_keep = []
        for e in elements:
            nodes_to_keep += e
        nodes_to_keep = np.unique(nodes_to_keep)
        self._nodes = vor.vertices[nodes_to_keep]
        node_dict = {n: k for k, n in enumerate(nodes_to_keep)}
        for k, el in enumerate(elements):
            for j, n in enumerate(el):
                elements[k][j] = node_dict[n]
        for k, ee in enumerate(element_edges):
            for j, n in enumerate(ee):
                element_edges[k][j] = node_dict[n]
        # determine material type of each element
        m0 = mtl.Material('NULL')
        element_materials = [m0 for k, _ in enumerate(elements)]
        element_materials = np.array(element_materials)
        point_collection = shp.MultiPoint(self.points).geoms
        for mr in self.material_regions:
            mpoly = shp.Polygon(self.vertices[mr.vertices])
            in_bnd = np.array([mpoly.contains(x) for x in point_collection])
            element_materials[in_bnd] = mr.material
        # create list of elements
        self._elements = []
        for e, m in zip(elements, element_materials):
            self.elements.append(PolyElement2D(self, e, m))
        # create lists of interface and boundary elements
        self._interface_elements = []
        self._boundary_elements = []
        for ee, en in zip(element_edges, element_neighbors):
            # boundary element
            if en[0] < 0 or en[1] < 0:
                nn = (self.elements[en[1]] if en[0] < 0
                      else self.elements[en[0]])
                self.boundary_elements.append(BoundaryElement2D(self, ee, nn))
            # interface element, assign material type from first neighbor
            else:
                nn = [self.elements[n] for n in en]
                m = nn[0].material
                self.interface_elements.append(
                        InterfaceElement2D(self, ee, m, nn))
        # set mesh valid, the setter will perform checks for mesh validity
        self.mesh_valid = True

    @property
    def verbose_printing(self):
        """Flag for whether :m:`__str__` will print verbose mesh information
        for the :c:`PolyMesh2D`.

        Parameters
        ----------
        flag : bool_like
            The new value of the :a:`verbose_printing` flag.

        Returns
        -------
        `bool`
            The value of the :a:`verbose_printing` flag.

        Raises
        ------
        `ValueError`
            If the value of **flag** cannot be converted to a `bool`.

        Note
        ----
        `str` values that can be cast to `float` are considered ``True``-like
        if non-zero and ``False``-like if zero. If the `str` cannot be cast
        to `float`, then the values 'y', 'yes', 't', 'true', and 'on' (case
        insensitive) are converted to ``True`` and the values 'n', 'no', 'f',
        'false', and 'off' are converted to ``False``. Other `str` values
        raise a `ValueError`.

        Examples
        --------
        >>> # initialize a mesh with no initial information provided
        >>> import vcfempy.meshgen
        >>> import vcfempy.materials
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> print(msh)
        vcfempy.meshgen.PolyMesh2D 'test mesh'
        Number of Vertices = 0
        Number of Boundary Vertices = 0
        Number of Material Regions = 0
        Number of Mesh Edges = 0
        Verbose Printing = False
        High Order Quadrature = False
        Mesh Generated = False

        >>> # add some vertices and boundary vertices to the mesh
        >>> # no mesh generated yet
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh)
        vcfempy.meshgen.PolyMesh2D 'test mesh'
        Number of Vertices = 4
        Number of Boundary Vertices = 4
        Number of Material Regions = 0
        Number of Mesh Edges = 0
        Verbose Printing = False
        High Order Quadrature = False
        Mesh Generated = False

        >>> # set verbose printing flag
        >>> msh.verbose_printing = True
        >>> print(msh)
        vcfempy.meshgen.PolyMesh2D 'test mesh'
        Number of Vertices = 4
        Number of Boundary Vertices = 4
        Number of Material Regions = 0
        Number of Mesh Edges = 0
        Verbose Printing = True
        High Order Quadrature = False
        Mesh Generated = False
        <BLANKLINE>
        Vertices
        [[0. 0.]
         [0. 1.]
         [1. 1.]
         [1. 0.]]
        <BLANKLINE>
        Boundary Vertices
        [0, 1, 2, 3]
        <BLANKLINE>
        Boundary Edges
        [[0, 1], [1, 2], [2, 3], [3, 0]]

        >>> # turn off verbose printing and add some vertices
        >>> # and two material regions
        >>> msh.verbose_printing = 'off'
        >>> msh.add_vertices([[0, 0.5], [1, 0.5]])
        >>> rock = vcfempy.materials.Material('rock')
        >>> sand = vcfempy.materials.Material('sand')
        >>> rock_region = vcfempy.meshgen.MaterialRegion2D(mesh=msh,
        ...     vertices=[0, 1, 2, 3], name='rock region', material=rock)
        >>> sand_region = vcfempy.meshgen.MaterialRegion2D(mesh=msh,
        ...     vertices=[4, 1, 2, 5], name='sand region', material=sand)
        >>> print(msh)
        vcfempy.meshgen.PolyMesh2D 'test mesh'
        Number of Vertices = 6
        Number of Boundary Vertices = 4
        Number of Material Regions = 2
        Number of Mesh Edges = 0
        Verbose Printing = False
        High Order Quadrature = False
        Mesh Generated = False

        >>> # turn verbose printing back on and generate the mesh
        >>> msh.verbose_printing = 'yes'
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh)
        vcfempy.meshgen.PolyMesh2D 'test mesh'
        Number of Vertices = 6
        Number of Boundary Vertices = 4
        Number of Material Regions = 2
        Number of Mesh Edges = 0
        Verbose Printing = True
        High Order Quadrature = False
        Mesh Generated = True
        Number of Nodes = 16
        Number of Elements = 9
        Number of Interface Elements = 12
        Number of Boundary Elements = 12
        <BLANKLINE>
        Vertices
        [[0.  0. ]
         [0.  1. ]
         [1.  1. ]
         [1.  0. ]
         [0.  0.5]
         [1.  0.5]]
        <BLANKLINE>
        Boundary Vertices
        [0, 1, 2, 3]
        <BLANKLINE>
        Boundary Edges
        [[0, 1], [1, 2], [2, 3], [3, 0]]
        <BLANKLINE>
        Material Region: rock region, Material: rock
        [0, 1, 2, 3]
        <BLANKLINE>
        Material Region: sand region, Material: sand
        [4, 1, 2, 5]
        <BLANKLINE>
        Nodes
        [[-2.77555756e-17  1.00000000e+00]
         [ 3.50000000e-01  1.00000000e+00]
         [ 0.00000000e+00  2.77555756e-17]
         [ 3.50000000e-01  2.77555756e-17]
         [ 1.00000000e+00  1.00000000e+00]
         [ 6.50000000e-01  1.00000000e+00]
         [ 6.50000000e-01  6.50000000e-01]
         [ 1.00000000e+00  6.50000000e-01]
         [ 1.00000000e+00 -2.77555756e-17]
         [ 6.50000000e-01 -2.77555756e-17]
         [ 6.50000000e-01  3.50000000e-01]
         [ 1.00000000e+00  3.50000000e-01]
         [ 2.77555756e-17  6.50000000e-01]
         [ 2.77555756e-17  3.50000000e-01]
         [ 3.50000000e-01  6.50000000e-01]
         [ 3.50000000e-01  3.50000000e-01]]
        <BLANKLINE>
        Element Nodes, Areas, Points, Centroids, Materials
        [15, 14, 6, 10], 0.09000000000000002, [0.5 0.5], [0.5 0.5], rock
        [15, 3, 2, 13], 0.12249999999999998, [0.2 0.2], [0.175 0.175], rock
        [12, 0, 1, 14], 0.1225, [0.2 0.8], [0.175 0.825], sand
        [7, 4, 5, 6], 0.1225, [0.8 0.8], [0.825 0.825], sand
        [11, 8, 9, 10], 0.12249999999999998, [0.8 0.2], [0.825 0.175], rock
        [15, 13, 12, 14], 0.10500000000000001, [0.2 0.5], [0.175 0.5  ], rock
        [5, 1, 14, 6], 0.10499999999999998, [0.5 0.8], [0.5   0.825], sand
        [10, 6, 7, 11], 0.10500000000000001, [0.8 0.5], [0.825 0.5  ], rock
        [3, 15, 10, 9], 0.10500000000000001, [0.5 0.2], [0.5   0.175], rock
        <BLANKLINE>
        Interface Element Nodes and Neighbors
        [5, 6], [3, 6]
        [6, 7], [3, 7]
        [9, 10], [4, 8]
        [10, 11], [4, 7]
        [6, 10], [7, 0]
        [12, 14], [5, 2]
        [13, 15], [5, 1]
        [14, 15], [5, 0]
        [1, 14], [2, 6]
        [3, 15], [1, 8]
        [6, 14], [0, 6]
        [10, 15], [0, 8]
        <BLANKLINE>
        Boundary Element Nodes and Neighbors
        [0, 1], 2
        [2, 3], 1
        [4, 5], 3
        [4, 7], 3
        [1, 5], 6
        [8, 9], 4
        [8, 11], 4
        [7, 11], 7
        [3, 9], 8
        [12, 13], 5
        [0, 12], 2
        [2, 13], 1

        >>> # attempting to set verbose_printing
        >>> # to a non-truth-like str value
        >>> msh.verbose_printing = 'dslk'
        Traceback (most recent call last):
        ...
        ValueError: invalid truth value 'dslk'
        """
        return self._verbose_printing

    @verbose_printing.setter
    def verbose_printing(self, flag):
        # try to cast flag to bool
        # will raise a ValueError if this does not work
        if isinstance(flag, (str, np.str_)):
            try:
                flag = float(flag)
            except ValueError:
                flag = distutils.util.strtobool(flag)
        self._verbose_printing = bool(flag)

    def __str__(self):
        # print header indicating basic information
        mesh_string = (f"{__name__}.{type(self).__name__} '{self.name}'\n"
                       + 'Number of Vertices = '
                       + f'{self.num_vertices}\n'
                       + 'Number of Boundary Vertices = '
                       + f'{self.num_boundary_vertices}\n'
                       + 'Number of Material Regions = '
                       + f'{self.num_material_regions}\n'
                       + 'Number of Mesh Edges = '
                       + f'{self.num_mesh_edges}\n'
                       + f'Verbose Printing = {self.verbose_printing}\n'
                       + 'High Order Quadrature = '
                       + f'{self.high_order_quadrature}\n'
                       + f'Mesh Generated = {self.mesh_valid}')

        # if mesh has been generated, print basic mesh information
        if self.mesh_valid:
            mesh_string += ('\nNumber of Nodes = '
                            + f'{self.num_nodes}\n'
                            + 'Number of Elements = '
                            + f'{self.num_elements}\n'
                            + 'Number of Interface Elements = '
                            + f'{self.num_interface_elements}\n'
                            + 'Number of Boundary Elements = '
                            + f'{self.num_boundary_elements}')

        # check for verbose printing flag, return now if False
        if not self.verbose_printing:
            return mesh_string

        # otherwise, if verbose printing is True, continue printing
        # detailed mesh information
        if self.num_vertices:
            mesh_string += f'\n\nVertices\n{self.vertices}'
        if self.num_boundary_vertices:
            mesh_string += f'\n\nBoundary Vertices\n{self.boundary_vertices}'
            mesh_string += f'\n\nBoundary Edges\n{self.boundary_edges}'
        if self.num_material_regions:
            for mr in self.material_regions:
                mesh_string += f'\n\nMaterial Region: {mr.name}, '
                mesh_string += f'Material: {mr.material.name}'
                mesh_string += f'\n{mr.vertices}'
        if self.num_mesh_edges:
            mesh_string += '\n\nMesh Edges'
            for e in self.mesh_edges:
                mesh_string += '\n{e}'
        if self.num_nodes:
            mesh_string += f'\n\nNodes\n{self.nodes}'
        if self.num_elements:
            mesh_string += ('\n\nElement Nodes, Areas, Points, Centroids, '
                            + 'Materials')
            for e, p in zip(self.elements, self.points):
                mesh_string += (f'\n{e.nodes}, {e.area}, {p}, {e.centroid}, '
                                + f'{e.material.name}')
        if self.num_interface_elements:
            mesh_string += '\n\nInterface Element Nodes and Neighbors'
            for e in self.interface_elements:
                en = [self.elements.index(n) for n in e.neighbors]
                mesh_string += f'\n{e.nodes}, {en}'
        if self.num_boundary_elements:
            mesh_string += '\n\nBoundary Element Nodes and Neighbors'
            for e in self.boundary_elements:
                en = self.elements.index(e.neighbor)
                mesh_string += f'\n{e.nodes}, {en}'
        return mesh_string

    def __repr__(self):
        return (f"<{__name__}.{type(self).__name__} object '{self.name}' at "
                + f"{hex(id(self))}>")

    def plot_boundaries(self, ax=None, **kwargs):
        """Plot the :c:`PolyMesh2D` :a:`boundary_edges` using
        :m:`matplotlib.pyplot.fill`.

        Parameters
        ----------
        ax : None | :c:`matplotlib.axes.Axes`
            The axes to plot on. If not provided, will try to get one using
            :m:`matplotlib.pyplot.gca`.

        Other Parameters
        ----------------
        **kwargs : :c:`matplotlib.patches.Polygon` properties, optional
            Default values:
            `edgecolor` = 'black'
            `linewidth` = 1.0,
            `linestyle` = '-',
            `fill` = ``False``.

        Returns
        -------
        :c:`matplotlib.axes.Axes`
            The axes that the :c:`MaterialRegion2D` was plotted on.

        Examples
        --------
        >>> # initialize a mesh, then plot the boundaries
        >>> import matplotlib.pyplot as plt
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> fig = plt.figure()
        >>> ax = msh.plot_boundaries()
        >>> xmin, xmax, ymin, ymax = ax.axis('equal')
        >>> xtext = ax.set_xlabel('x')
        >>> ytext = ax.set_ylabel('y')
        >>> ttext = ax.set_title('PolyMesh2D Boundaries Test Plot')
        >>> plt.savefig('PolyMesh2D_boundaries_test_plot.png')
        """
        if ax is None or not isinstance(ax, plt.Axes):
            ax = plt.gca()
        if 'edgecolor' not in kwargs.keys():
            kwargs['edgecolor'] = 'black'
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 1.0
        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = '-'
        if 'fill' not in kwargs.keys():
            kwargs['fill'] = False
        ax.fill(self.vertices[self.boundary_vertices, 0],
                self.vertices[self.boundary_vertices, 1],
                **kwargs)
        return ax

    def plot_material_regions(self, ax=None):
        """Plot the :c:`PolyMesh2D` :a:`material_regions`.

        Parameters
        ----------
        ax : None | :c:`matplotlib.axes.Axes`
            The axes to plot on. If not provided, will try to get one using
            :m:`matplotlib.pyplot.gca`.

        Note
        ----
        This :m:`plot_material_regions` method calls the
        :m:`MaterialRegion2D.plot` method for each material region using
        default parameters. If you would like to modify the plotting
        parameters, use the :m:`MaterialRegion2D.plot` method directly.

        Returns
        -------
        :c:`matplotlib.axes.Axes`
            The axes that the :a:`material_regions` were plotted on.

        Examples
        --------
        >>> # initialize a mesh, then plot the boundaries and material regions
        >>> import matplotlib.pyplot as plt
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.add_vertices([[0, 0.5], [1, 0.5]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> sand = vcfempy.materials.Material('sand', color='xkcd:sand')
        >>> clay = vcfempy.materials.Material('clay', color='xkcd:clay')
        >>> sand_reg = vcfempy.meshgen.MaterialRegion2D(msh, [0, 4, 5, 3],
        ...                                             sand)
        >>> clay_reg = vcfempy.meshgen.MaterialRegion2D(msh, [4, 1, 2, 5],
        ...                                             clay)
        >>> fig = plt.figure()
        >>> ax = msh.plot_boundaries()
        >>> ax = msh.plot_material_regions()
        >>> ax = msh.plot_vertices()
        >>> xmin, xmax, ymin, ymax = ax.axis('equal')
        >>> xtext = ax.set_xlabel('x')
        >>> ytext = ax.set_ylabel('y')
        >>> ttext = ax.set_title('PolyMesh2D Material Regions Test Plot')
        >>> plt.savefig('PolyMesh2D_material_regions_test_plot.png')
        """
        if ax is None:
            ax = plt.gca()
        for mr in self.material_regions:
            mr.plot(ax)
        return ax

    def plot_vertices(self, ax=None, **kwargs):
        """Plot the :a:`vertices` of the :c:`PolyMesh2D` using
        :m:`matplotlib.pyplot.plot`.

        Parameters
        ----------
        ax : :c:`matplotlib.axes.Axes`, optional
            The axes to plot on. If not provided, will try to get one using
            :m:`matplotlib.pyplot.gca`.

        Other Parameters
        ----------------
        **kwargs : :c:`matplotlib.lines.Line2D` properties, optional
            Default values:
            `linewidth` = 0.0,
            `markeredgecolor` = 'black'
            `markerfacecolor` = 'white'
            `marker` = 's',
            `markersize` = 8.0.

        Returns
        -------
        :c:`matplotlib.axes.Axes`
            The axes that the :a:`vertices` were plotted on.

        Examples
        --------
        >>> # initialize a mesh and plot the nodes
        >>> import matplotlib.pyplot as plt
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock', color='xkcd:greenish')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, msh.boundary_vertices,
        ...                                       rock, 'rock region')
        >>> fig = plt.figure()
        >>> ax = msh.plot_boundaries()
        >>> ax = msh.plot_material_regions()
        >>> ax = msh.plot_vertices()
        >>> xmin, xmax, ymin, ymax = ax.axis('equal')
        >>> xtext = ax.set_xlabel('x')
        >>> ytext = ax.set_ylabel('y')
        >>> ttext = ax.set_title('PolyMesh2D Vertices Test Plot')
        >>> plt.savefig('PolyMesh2D_vertices_test_plot.png')
        """
        if ax is None or not isinstance(ax, plt.Axes):
            ax = plt.gca()
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 0.0
        if 'markeredgecolor' not in kwargs.keys():
            kwargs['markeredgecolor'] = 'black'
        if 'markerfacecolor' not in kwargs.keys():
            kwargs['markerfacecolor'] = 'black'
        if 'marker' not in kwargs.keys():
            kwargs['marker'] = 's'
        if 'markersize' not in kwargs.keys():
            kwargs['markersize'] = 8.0
        ax.plot(self.vertices[:, 0], self.vertices[:, 1], **kwargs)
        return ax

    def plot_mesh_edges(self, ax=None):
        if ax is None:
            ax = plt.gca()
        for edge in self.mesh_edges:
            edge.plot(ax)
        return ax

    def plot_mesh(self, ax=None, elements=True, interface_elements=True,
                  boundary_elements=True, element_quad_points=False):
        if ax is None:
            ax = plt.gca()
        if elements:
            for e in self.elements:
                e.plot(ax)
        if interface_elements:
            for e in self.interface_elements:
                e.plot(ax)
        if boundary_elements:
            for e in self.boundary_elements:
                e.plot(ax)
        if element_quad_points:
            for e in self.elements:
                e.plot_quad_points(ax)
        return ax

    def plot_nodes(self, ax=None, **kwargs):
        """Plot the :a:`nodes` of the :c:`PolyMesh2D` using
        :m:`matplotlib.pyplot.plot`.

        Parameters
        ----------
        ax : :c:`matplotlib.axes.Axes`, optional
            The axes to plot on. If not provided, will try to get one using
            :m:`matplotlib.pyplot.gca`.

        Other Parameters
        ----------------
        **kwargs : :c:`matplotlib.lines.Line2D` properties, optional
            Default values:
            `linewidth` = 0.0,
            `markeredgecolor` = 'black'
            `markerfacecolor` = 'white'
            `marker` = 'o',
            `markersize` = 4.0.

        Returns
        -------
        :c:`matplotlib.axes.Axes`
            The axes that the :a:`nodes` were plotted on.

        Examples
        --------
        >>> # initialize a mesh and plot the nodes
        >>> import matplotlib.pyplot as plt
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock', color='xkcd:greenish')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, msh.boundary_vertices,
        ...                                       rock, 'rock region')
        >>> msh.mesh_scale = 0.2
        >>> msh.mesh_rand = 0.2
        >>> msh.generate_mesh()
        >>> fig = plt.figure()
        >>> ax = msh.plot_mesh()
        >>> ax = msh.plot_vertices()
        >>> ax = msh.plot_nodes()
        >>> xmin, xmax, ymin, ymax = ax.axis('equal')
        >>> xtext = ax.set_xlabel('x')
        >>> ytext = ax.set_ylabel('y')
        >>> ttext = ax.set_title('PolyMesh2D Nodes Test Plot')
        >>> plt.savefig('PolyMesh2D_nodes_test_plot.png')
        """
        if ax is None or not isinstance(ax, plt.Axes):
            ax = plt.gca()
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 0.0
        if 'markeredgecolor' not in kwargs.keys():
            kwargs['markeredgecolor'] = 'black'
        if 'markerfacecolor' not in kwargs.keys():
            kwargs['markerfacecolor'] = 'white'
        if 'marker' not in kwargs.keys():
            kwargs['marker'] = 'o'
        if 'markersize' not in kwargs.keys():
            kwargs['markersize'] = 4.0
        ax.plot(self.nodes[:, 0], self.nodes[:, 1], **kwargs)
        return ax


class MaterialRegion2D():
    """A class for defining material regions and their attributes for meshes
    generated by a :c:`PolyMesh2D`..

    Parameters
    ----------
    mesh : :c:`PolyMesh2D`
        The parent mesh. Sets :a:`mesh`.
    vertices : list[int], optional
        Initial list of vertices defining the :c:`MaterialRegion2D`. Passed
        to :m:`insert_vertices`.
    material : :c:`vcfempy.materials.Material`, optional
        The material type of the :c:`MaterialRegion2D`. Sets :a:`material`.
    name : str, optional
        A descriptive name for the :c:`MaterialRegion2D`. If not provided,
        will be set to a default 'Unnamed Material Region {`k`}' where `k` is
        a counter for how many :c:`MaterialRegion2D` have been created.

    Other Parameters
    ----------------
    add_to_mesh : bool, optional, default=True
       Flag for whether to add the :c:`MaterialRegion2D` to its parent mesh.
       This is done by default when the :c:`MaterialRegion2D` is created.

    Examples
    --------
    >>> # initialize a mesh, no material regions added
    >>> import vcfempy.meshgen
    >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
    >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
    >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
    >>> print(msh.num_material_regions)
    0

    >>> # create a material region, this will add it to its parent mesh
    >>> import vcfempy.materials
    >>> rock_material = vcfempy.materials.Material('rock material')
    >>> rock_region = vcfempy.meshgen.MaterialRegion2D(msh, [0, 1, 2, 3],
    ...                                                rock_material,
    ...                                                'rock region')
    >>> print(msh.num_material_regions)
    1
    >>> print(rock_region in msh.material_regions)
    True
    >>> print(rock_region.name)
    rock region
    >>> print(rock_region.material.name)
    rock material
    >>> print(rock_region.vertices)
    [0, 1, 2, 3]

    >>> # generate a mesh, then change material region material
    >>> # this clears the mesh
    >>> msh.mesh_scale = 0.4
    >>> msh.add_seed_points([0.5, 0.5])
    >>> msh.generate_mesh()
    >>> print(msh.mesh_valid)
    True
    >>> rock_region.material = None
    >>> print(rock_region.material)
    None
    >>> print(msh.mesh_valid)
    False

    >>> # regenerate the mesh, then change the material region vertices
    >>> # this also clears the mesh
    >>> # note that the material region need not be fully inside the
    >>> # mesh boundaries
    >>> msh.generate_mesh()
    >>> print(msh.mesh_valid)
    True
    >>> msh.add_vertices([0.5, 1.5])
    >>> print(msh.mesh_valid)
    True
    >>> rock_region.insert_vertices(2, 4)
    >>> print(rock_region.vertices)
    [0, 1, 4, 2, 3]
    >>> print(msh.mesh_valid)
    False
    """

    _num_created = 0

    def __init__(self, mesh, vertices=None, material=None, name=None,
                 add_to_mesh=True):
        if not isinstance(mesh, PolyMesh2D):
            raise TypeError('type(mesh) must be vcfempy.meshgen.PolyMesh2D')
        self._mesh = mesh
        if add_to_mesh:
            self.mesh.add_material_region(self)

        if name is None:
            name = ('Unnamed Material Region '
                    + f'{MaterialRegion2D._num_created}')
        self.name = name
        MaterialRegion2D._num_created += 1

        self._vertices = []
        self.insert_vertices(0, vertices)

        self.material = material

    @property
    def name(self):
        """A descriptive name for the :c:`MaterialRegion2D`.

        Parameters
        ----------
        name : str
            The name of the :c:`MaterialRegion2D`. Will be cast to `str`
            regardless of type.

        Returns
        -------
        `str`
            The :a:`name` of the :c:`MaterialRegion2D`.

        Examples
        --------
        >>> # create a blank material region without a name (reset counter)
        >>> import vcfempy.meshgen
        >>> vcfempy.meshgen.MaterialRegion2D._num_created = 0
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh)
        >>> print(mr.name)
        Unnamed Material Region 0

        >>> # setting the name
        >>> mr.name = 'Rock region'
        >>> print(mr.name)
        Rock region

        >>> # changing the name property to non-str
        >>> # will be cast to str
        >>> mr.name = 1
        >>> print(mr.name)
        1
        >>> print(type(mr.name).__name__)
        str

        >>> # initialize a material region with a name
        >>> mr = vcfempy.meshgen.MaterialRegion2D(mesh=msh, name='new region')
        >>> print(mr.name)
        new region

        >>> # initialize another material region without a name
        >>> # notice that the "Unnamed" counter increases for every region
        >>> # created (including those that were assigned an initial name)
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh)
        >>> print(mr.name)
        Unnamed Material Region 2
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def mesh(self):
        """The parent :c:`PolyMesh2D` of the :c:`MaterialRegion2D`.

        Returns
        -------
        :c:`PolyMesh2D`
            The parent mesh object

        Note
        ----
        This property is immutable to ensure connection between a
        :c:`PolyMesh2D` and a :c:`MaterialRegion2D`.

        Examples
        --------
        >>> # create a mesh and a material region
        >>> # note that creating the material region requires a parent mesh
        >>> # and the material region will add itself to the list of parent
        >>> # mesh material regions by default
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh)
        >>> print(mr.mesh.name)
        test mesh
        >>> print(mr in msh.material_regions)
        True

        >>> # try to set parent mesh (immutable)
        >>> new_mesh = vcfempy.meshgen.PolyMesh2D()
        >>> mr.mesh = new_mesh
        Traceback (most recent call last):
            ...
        AttributeError: can't set attribute
        """
        return self._mesh

    @property
    def num_vertices(self):
        """Number of vertices defining the :c:`MaterialRegion2D` geometry.

        Returns
        -------
        `int`
            The number of :a:`vertices` in the :c:`MaterialRegion2D`.

        Examples
        --------
        >>> # creating a material region, no initial vertices provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh)
        >>> print(mr.num_vertices)
        0

        >>> # creating a material region, providing initial vertices
        >>> # these are indices referencing vertices in the parent mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> mr.insert_vertices(0, [k for k, _ in enumerate(new_verts)])
        >>> print(mr.num_vertices)
        4

        >>> # add a vertex and check num_vertices
        >>> msh.add_vertices([1.5, 0.5])
        >>> mr.insert_vertices(3, 4)
        >>> print(mr.num_vertices)
        5
        """
        return len(self.vertices)

    @property
    def vertices(self):
        """List of vertex indices defining the boundary of the
        :c:`MaterialRegion2D`.

        Returns
        -------
        `list[int]`
            The list of vertex indices referencing :a:`PolyMesh2D.vertices`
            of :a:`mesh`.

        Examples
        --------
        >>> # creating a material region, no initial vertices provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh)
        >>> print(mr.num_vertices)
        0

        >>> # creating a material region, providing initial vertices
        >>> # these are indices referencing vertices in the parent mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> mr.insert_vertices(0, [k for k, _ in enumerate(new_verts)])
        >>> print(mr.vertices)
        [0, 1, 2, 3]
        >>> print(msh.vertices[mr.vertices, :])
        [[0. 0.]
         [0. 1.]
         [1. 1.]
         [1. 0.]]

        >>> # add a vertex and check vertices
        >>> msh.add_vertices([1.5, 0.5])
        >>> mr.insert_vertices(3, 4)
        >>> print(mr.vertices)
        [0, 1, 2, 4, 3]
        >>> print(msh.vertices[mr.vertices, :])
        [[0.  0. ]
         [0.  1. ]
         [1.  1. ]
         [1.5 0.5]
         [1.  0. ]]
        """
        return self._vertices

    @property
    def material(self):
        """The :c:`vcfempy.materials.Material` assigned to the
        :c:`MaterialRegion2D`.

        Parameters
        ----------
        material : None | :c:`vcfempy.materials.Material`
            The material type to assign to the :c:`MaterialRegion2D`.

        Returns
        -------
        ``None`` | :c:`vcfempy.materials.Material`
            The material type assigned to the :c:`MaterialRegion2D`.

        Raises
        ------
        TypeError
            If **material** is not ``None`` or a
            :c:`vcfempy.materials.Material`.

        Examples
        --------
        >>> # create a material region, no material type assigned
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, msh.boundary_vertices,
        ...                                       name='rock region')
        >>> print(mr in msh.material_regions)
        True
        >>> print(mr.material)
        None

        >>> # assign a material type to the material region
        >>> rock = vcfempy.materials.Material('rock')
        >>> mr.material = rock
        >>> print(mr.material.name)
        rock

        >>> # changing material type of a material region resets the mesh
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.mesh_valid)
        True
        >>> mr.material = None
        >>> print(mr.material)
        None
        >>> print(msh.mesh_valid)
        False

        >>> # try to assign invalid materials to a material region
        >>> mr.material = 1
        Traceback (most recent call last):
            ...
        TypeError: type(material) not in [NoneType, vcfempy.materials.Material]
        >>> mr.material = 'rock'
        Traceback (most recent call last):
            ...
        TypeError: type(material) not in [NoneType, vcfempy.materials.Material]
        """
        return self._material

    @material.setter
    def material(self, material):
        if not isinstance(material, (type(None), mtl.Material)):
            raise TypeError('type(material) not in [NoneType, '
                            + 'vcfempy.materials.Material]')
        self._material = material
        self.mesh.mesh_valid = False

    def insert_vertices(self, index, vertices):
        """Insert one or more vertex indices to the :c:`MaterialRegion2D`.

        Parameters
        ----------
        index : int
            The index at which to insert the **vertices** into :a:`vertices`.
        vertices : int | list[int]
            The list of vertex indices to add to :a:`vertices`.

        Note
        -----
        Before inserting the values in **vertices**, an attempt is made to
        cast to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        TypeError
            If **index** cannot be interpreted as `int`.
        ValueError
            If **vertices** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **vertices** cannot be cast to `int`, are
            already in :a:`vertices`, are negative, or are
            >= :a:`mesh.num_vertices`.

        Examples
        --------
        >>> # create mesh and material region, add some vertices
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh)
        >>> mr.insert_vertices(0, [0, 1, 2, 3])
        >>> print(mr.vertices)
        [0, 1, 2, 3]

        >>> # add a single vertex and add it to the material region
        >>> msh.add_vertices([1.5, 0.5])
        >>> mr.insert_vertices(index=3, vertices=4)
        >>> print(mr.vertices)
        [0, 1, 2, 4, 3]

        >>> # add two more vertices and add them to the material region
        >>> msh.add_vertices([[0.25, 1.25], [0.75, 1.25]])
        >>> mr.insert_vertices(2, [5, 6])
        >>> print(mr.vertices)
        [0, 1, 5, 6, 2, 4, 3]

        >>> # the list of boundary vertices need not be 1d
        >>> # if not, it will be flattened
        >>> msh.add_vertices([[-0.5, 0.1], [-0.75, 0.25],
        ...                   [-0.75, 0.75], [-0.5, 0.9]])
        >>> mr.insert_vertices(1, [[7, 8], [9, 10]])
        >>> print(mr.vertices)
        [0, 7, 8, 9, 10, 1, 5, 6, 2, 4, 3]

        >>> # add no vertices, in two different ways
        >>> mr.insert_vertices(0, None)
        >>> mr.insert_vertices(0, [])
        >>> print(mr.vertices)
        [0, 7, 8, 9, 10, 1, 5, 6, 2, 4, 3]

        >>> # try to insert some invalid vertices
        >>> mr.insert_vertices(0, 'one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> mr.insert_vertices(0, 1)
        Traceback (most recent call last):
            ...
        ValueError: 1 is already a vertex
        >>> mr.insert_vertices(0, 11)
        Traceback (most recent call last):
            ...
        ValueError: vertex index 11 out of range
        >>> mr.insert_vertices(0, -1)
        Traceback (most recent call last):
            ...
        ValueError: vertex index -1 out of range
        >>> mr.insert_vertices(
        ...             0, [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        >>> msh.add_vertices([0.5, -0.5])
        >>> mr.insert_vertices('one', 11)
        Traceback (most recent call last):
            ...
        TypeError: 'str' object cannot be interpreted as an integer
        """
        if vertices is None:
            return
        vertices = np.array(vertices, dtype=int, ndmin=1)
        if len(vertices) == 0:
            return
        vertices = np.flip(vertices.ravel())
        for v in vertices:
            if v in self.vertices:
                raise ValueError(f'{v} is already a vertex')
            if v < 0 or v >= self.mesh.num_vertices:
                raise ValueError(f'vertex index {v} out of range')
            self.vertices.insert(index, int(v))
        self.mesh.mesh_valid = False

    def remove_vertices(self, remove_vertices):
        """Remove one or more vertex indices from the :c:`MaterialRegion2D`.

        Parameters
        ----------
        remove_vertices : int | list[int]
            The vertex or list of vertices to remove from :a:`vertices`.

        Note
        -----
        Before removing the values in **remove_vertices**, an attempt will be
        made to cast it to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        ValueError
            If **remove_vertices** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **remove_vertices** cannot be cast to `int` or
            are not in :a:`vertices`.

        Examples
        --------
        >>> # create mesh and material region, add/remove some vertices
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh)
        >>> mr.insert_vertices(0, [0, 1, 2, 3])
        >>> mr.remove_vertices(1)
        >>> print(mr.vertices)
        [0, 2, 3]

        >>> # remove multiple vertices
        >>> mr.insert_vertices(0, 1)
        >>> mr.remove_vertices([1, 3])
        >>> print(mr.vertices)
        [0, 2]

        >>> # the list of vertices to remove need not be 1d
        >>> # if not, it will be flattened
        >>> mr.insert_vertices(1, 1)
        >>> mr.insert_vertices(3, 3)
        >>> mr.remove_vertices([[0, 1], [2, 3]])
        >>> print(mr.vertices)
        []

        >>> # remove no vertices, in two different ways
        >>> mr.insert_vertices(0, [0, 1, 2, 3])
        >>> mr.remove_vertices(None)
        >>> mr.remove_vertices([])
        >>> print(mr.vertices)
        [0, 1, 2, 3]

        >>> # try to remove some invalid vertices
        >>> mr.remove_vertices('one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> mr.remove_vertices(4)
        Traceback (most recent call last):
            ...
        ValueError: list.remove(x): x not in list
        >>> mr.remove_vertices(
        ...                 [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        """
        if remove_vertices is None:
            return
        remove_vertices = np.array(remove_vertices, dtype=int, ndmin=1)
        if len(remove_vertices) == 0:
            return
        remove_vertices = remove_vertices.ravel()
        for rv in remove_vertices:
            self.vertices.remove(rv)
        self.mesh.mesh_valid = False

    def plot(self, ax=None, **kwargs):
        """Plot the :c:`MaterialRegion2D` using :m:`matplotlib.pyplot.fill`.

        Parameters
        ----------
        ax : None | :c:`matplotlib.axes.Axes`
            The axes to plot on. If not provided, will try to get one using
            :m:`matplotlib.pyplot.gca`.

        Other Parameters
        ----------------
        **kwargs : :c:`matplotlib.patches.Polygon` properties, optional
            Default values:
            `edgecolor` = :a:`material` `color` (or 'black' if :a:`material`
            is ``None``) with alpha = 1.0,
            `facecolor` = :a:`material` `color` (or 'black' if :a:`material`
            is ``None``) with alpha = 0.8,
            `linewidth` = 2.0,
            `linestyle` = '-'.

        Returns
        -------
        :c:`matplotlib.axes.Axes`
            The axes that the :c:`MaterialRegion2D` was plotted on.

        Examples
        --------
        >>> # initialize a mesh and a material region, then plot the
        >>> # material region
        >>> import matplotlib.pyplot as plt
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock', color='xkcd:stone')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, msh.boundary_vertices,
        ...                                       rock, 'rock region')
        >>> fig = plt.figure()
        >>> ax = mr.plot()
        >>> xmin, xmax, ymin, ymax = ax.axis('equal')
        >>> xtext = ax.set_xlabel('x')
        >>> ytext = ax.set_ylabel('y')
        >>> ttext = ax.set_title('MaterialRegion2D Test Plot')
        >>> leg = ax.legend(labels=[mr.name])
        >>> plt.savefig('MaterialRegion2D_test_plot.png')
        """
        if ax is None or not isinstance(ax, plt.Axes):
            ax = plt.gca()
        if self.material is not None:
            color = mplclr.to_rgb(self.material.color)
        else:
            color = mplclr.to_rgb('black')
        if 'edgecolor' not in kwargs.keys():
            kwargs['edgecolor'] = color + (1.0, )
        if 'facecolor' not in kwargs.keys():
            kwargs['facecolor'] = color + (0.8, )
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 2.0
        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = '-'
        ax.fill(self.mesh.vertices[self.vertices, 0],
                self.mesh.vertices[self.vertices, 1],
                **kwargs)
        return ax


class MeshEdge2D():
    """A class for defining edges to be preserved and their attributes for
    meshes generated by a :c:`PolyMesh2D`..

    Parameters
    ----------
    mesh : :c:`PolyMesh2D`
        The parent mesh. Sets :a:`mesh`.
    vertices : list[int], optional
        Initial list of vertices defining the :c:`MeshEdge2D`. Passed to
        :m:`insert_vertices`.
    material : :c:`vcfempy.materials.Material`, optional
        The material type of the :c:`MeshEdge2D`. Sets :a:`material`.
    name : str, optional
        A descriptive name for the :c:`MeshEdge2D`. If not provided, will be
        set to a default 'Unnamed Mesh Edge {`k`}' where `k` is a counter for
        how many :c:`MeshEdge2D` have been created.

    Other Parameters
    ----------------
    add_to_mesh : bool, optional, default=True
       Flag for whether to add the :c:`MeshEdge2D` to its parent mesh. This
       is done by default when the :c:`MeshEdge2D` is created.

    Examples
    --------
    >>> # initialize a mesh, no mesh edges added
    >>> import vcfempy.meshgen
    >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
    >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
    >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
    >>> print(msh.num_mesh_edges)
    0

    >>> # create a mesh edge, this will add it to its parent mesh
    >>> import vcfempy.materials
    >>> rj_material = vcfempy.materials.Material('rock joint material')
    >>> msh.add_vertices([[0.1, 0.1], [0.8, 0.8]])
    >>> rock_joint = vcfempy.meshgen.MeshEdge2D(msh, [4, 5], rj_material,
    ...                                         'rock joint')
    >>> print(msh.num_mesh_edges)
    1
    >>> print(rock_joint in msh.mesh_edges)
    True
    >>> print(rock_joint.name)
    rock joint
    >>> print(rock_joint.material.name)
    rock joint material
    >>> print(rock_joint.vertices)
    [4, 5]
    >>> print(msh.vertices[rock_joint.vertices, :])
    [[0.1 0.1]
     [0.8 0.8]]

    >>> # generate a mesh, then change mesh edge material
    >>> # this clears the mesh
    >>> msh.mesh_scale = 0.4
    >>> msh.add_seed_points([0.5, 0.5])
    >>> msh.generate_mesh()
    >>> print(msh.mesh_valid)
    True
    >>> rock_joint.material = None
    >>> print(rock_joint.material)
    None
    >>> print(msh.mesh_valid)
    False

    >>> # regenerate the mesh, then change the mesh edge vertices
    >>> # this also clears the mesh
    >>> msh.generate_mesh()
    >>> print(msh.mesh_valid)
    True
    >>> msh.add_vertices([0.5, 0.65])
    >>> print(msh.mesh_valid)
    True
    >>> rock_joint.insert_vertices(1, 6)
    >>> print(rock_joint.vertices)
    [4, 6, 5]
    >>> print(msh.mesh_valid)
    False
    """

    _num_created = 0

    def __init__(self, mesh, vertices=None, material=None, name=None,
                 add_to_mesh=True):
        if not isinstance(mesh, PolyMesh2D):
            raise TypeError('type(mesh) must be vcfempy.meshgen.PolyMesh2D')
        self._mesh = mesh
        if add_to_mesh:
            self.mesh.add_mesh_edge(self)

        if name is None:
            name = ('Unnamed Mesh Edge '
                    + f'{MeshEdge2D._num_created}')
        self.name = name
        MeshEdge2D._num_created += 1

        self._vertices = []
        self.insert_vertices(0, vertices)

        self.material = material

    @property
    def name(self):
        """A descriptive name for the :c:`MeshEdge2D`.

        Parameters
        ----------
        name : str
            The name of the :c:`MeshEdge2D`. Will be cast to `str`
            regardless of type.

        Returns
        -------
        `str`
            The :a:`name` of the :c:`MeshEdge2D`.

        Examples
        --------
        >>> # create a blank mesh edge without a name (reset counter)
        >>> import vcfempy.meshgen
        >>> vcfempy.meshgen.MeshEdge2D._num_created = 0
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> me = vcfempy.meshgen.MeshEdge2D(msh)
        >>> print(me.name)
        Unnamed Mesh Edge 0

        >>> # setting the name
        >>> me.name = 'Rock Joint'
        >>> print(me.name)
        Rock Joint

        >>> # changing the name property to non-str
        >>> # will be cast to str
        >>> me.name = 1
        >>> print(me.name)
        1
        >>> print(type(me.name).__name__)
        str

        >>> # initialize a mesh edge with a name
        >>> me = vcfempy.meshgen.MeshEdge2D(mesh=msh, name='The Edge')
        >>> print(me.name)
        The Edge

        >>> # initialize another mesh edge without a name
        >>> # notice that the "Unnamed" counter increases for every edge
        >>> # created (including those that were assigned an initial name)
        >>> me = vcfempy.meshgen.MeshEdge2D(msh)
        >>> print(me.name)
        Unnamed Mesh Edge 2
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def mesh(self):
        """The parent :c:`PolyMesh2D` of the :c:`MeshEdge2D`.

        Returns
        -------
        :c:`PolyMesh2D`
            The parent mesh object

        Note
        ----
        This property is immutable to ensure connection between a
        :c:`PolyMesh2D` and a :c:`MeshEdge2D`.

        Examples
        --------
        >>> # create a mesh and a mesh edge
        >>> # note that creating the mesh edge requires a parent mesh
        >>> # and the mesh edge will add itself to the list of parent
        >>> # mesh edges by default
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> me = vcfempy.meshgen.MeshEdge2D(msh)
        >>> print(me.mesh.name)
        test mesh
        >>> print(me in msh.mesh_edges)
        True

        >>> # try to set parent mesh (immutable)
        >>> new_mesh = vcfempy.meshgen.PolyMesh2D()
        >>> me.mesh = new_mesh
        Traceback (most recent call last):
            ...
        AttributeError: can't set attribute
        """
        return self._mesh

    @property
    def num_vertices(self):
        """Number of vertices defining the :c:`MeshEdge2D` geometry.

        Returns
        -------
        `int`
            The number of :a:`vertices` in the :c:`MeshEdge2D`.

        Examples
        --------
        >>> # creating a mesh edge, no initial vertices provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> me = vcfempy.meshgen.MeshEdge2D(msh)
        >>> print(me.num_vertices)
        0

        >>> # creating a mesh edge, providing initial vertices
        >>> # these are indices referencing vertices in the parent mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> msh.add_vertices([[0.1, 0.1], [0.8, 0.8]])
        >>> me = vcfempy.meshgen.MeshEdge2D(msh, [4, 5])
        >>> print(msh.num_mesh_edges)
        2
        >>> print(me.num_vertices)
        2

        >>> # add a vertex and check num_vertices
        >>> msh.add_vertices([0.5, 0.65])
        >>> me.insert_vertices(1, 6)
        >>> print(me.num_vertices)
        3
        """
        return len(self.vertices)

    @property
    def vertices(self):
        """The list of vertex indices in the :c:`MeshEdge2D`.

        Returns
        -------
        `list[int]`
            A list of vertex indices referencing :a:`PolyMesh2D.vertices`.

        Examples
        --------
        >>> # creating a mesh edge, no initial vertices provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> me = vcfempy.meshgen.MeshEdge2D(msh)
        >>> print(me.vertices)
        []

        >>> # creating a mesh edge, providing initial vertices
        >>> # these are indices referencing vertices in the parent mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> msh.add_vertices([[0.1, 0.1], [0.8, 0.8]])
        >>> me = vcfempy.meshgen.MeshEdge2D(msh, [4, 5])
        >>> print(me.vertices)
        [4, 5]
        >>> print(msh.vertices[me.vertices, :])
        [[0.1 0.1]
         [0.8 0.8]]

        >>> # add a vertex and check vertices
        >>> msh.add_vertices([0.5, 0.65])
        >>> me.insert_vertices(1, 6)
        >>> print(me.vertices)
        [4, 6, 5]
        >>> print(msh.vertices[me.vertices, :])
        [[0.1  0.1 ]
         [0.5  0.65]
         [0.8  0.8 ]]
        """
        return self._vertices

    @property
    def material(self):
        """The :c:`vcfempy.materials.Material` assigned to the
        :c:`MeshEdge2D`.

        Parameters
        ----------
        material : None | :c:`vcfempy.materials.Material`
            The material type to assign to the :c:`MeshEdge2D`.

        Returns
        -------
        ``None`` | :c:`vcfempy.materials.Material`
            The material type assigned to the :c:`MeshEdge2D`.

        Raises
        ------
        TypeError
            If **material** is not ``None`` or a
            :c:`vcfempy.materials.Material`.

        Examples
        --------
        >>> # create a mesh edge, no material type assigned
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> me = vcfempy.meshgen.MeshEdge2D(msh)
        >>> msh.add_vertices([[0.1, 0.1], [0.8, 0.8]])
        >>> me.insert_vertices(0, [4, 5])
        >>> print(me.material)
        None

        >>> # create a mesh edge, assigning a material type
        >>> rock_joint = vcfempy.materials.Material('rock joint')
        >>> rj_edge = vcfempy.meshgen.MeshEdge2D(msh, material=rock_joint)
        >>> msh.add_vertices([[0.1, 0.4], [0.3, 0.9]])
        >>> rj_edge.insert_vertices(0, [6, 7])
        >>> print(rj_edge in msh.mesh_edges)
        True
        >>> print(rj_edge.material.name)
        rock joint

        >>> # assign a new material to an edge
        >>> sandy_joint = vcfempy.materials.Material('sandy joint')
        >>> me.material = sandy_joint
        >>> print(me in msh.mesh_edges)
        True
        >>> print(me.material.name)
        sandy joint

        >>> # changing material type of an edge resets the mesh
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.mesh_valid)
        True
        >>> me.material = None
        >>> print(me.material)
        None
        >>> print(msh.mesh_valid)
        False

        >>> # try to assign invalid materials to an edge
        >>> me.material = 1
        Traceback (most recent call last):
            ...
        TypeError: type(material) not in [NoneType, vcfempy.materials.Material]
        >>> me.material = 'rock joint'
        Traceback (most recent call last):
            ...
        TypeError: type(material) not in [NoneType, vcfempy.materials.Material]
        """
        return self._material

    @material.setter
    def material(self, material):
        if not isinstance(material, (type(None), mtl.Material)):
            raise TypeError('type(material) not in [NoneType, '
                            + 'vcfempy.materials.Material]')
        self._material = material
        self.mesh.mesh_valid = False

    def insert_vertices(self, index, vertices):
        """Insert one or more vertex indices to the :c:`MeshEdge2D`.

        Parameters
        ----------
        index : int
            The index at which to insert the **vertices** into :a:`vertices`.
        vertices : int | list[int]
            The list of vertex indices to add to :a:`vertices`.

        Note
        -----
        Before inserting the values in **vertices**, an attempt is made to
        cast to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        TypeError
            If **index** cannot be interpreted as `int`.
        ValueError
            If **vertices** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **vertices** cannot be cast to `int`, are
            already in :a:`vertices`, are negative, or are
            >= :a:`mesh.num_vertices`.

        Examples
        --------
        >>> # create mesh and material region, add some vertices
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.add_vertices([[0.1, 0.1], [0.8, 0.8]])
        >>> me = vcfempy.meshgen.MeshEdge2D(msh)
        >>> me.insert_vertices(0, [4, 5])
        >>> print(me.vertices)
        [4, 5]

        >>> # add a single vertex and add it to the mesh edge
        >>> msh.add_vertices([0.5, 0.9])
        >>> me.insert_vertices(index=1, vertices=6)
        >>> print(me.vertices)
        [4, 6, 5]

        >>> # add two more vertices and add them to the mesh edge
        >>> msh.add_vertices([[0.25, 0.65], [0.35, 0.75]])
        >>> me.insert_vertices(1, [7, 8])
        >>> print(me.vertices)
        [4, 7, 8, 6, 5]

        >>> # the list of boundary vertices need not be 1d
        >>> # if not, it will be flattened
        >>> msh.add_vertices([[0.55, 0.85], [0.6, 0.75],
        ...                   [0.65, 0.85], [0.75, 0.75]])
        >>> me.insert_vertices(4, [[9, 10], [11, 12]])
        >>> print(me.vertices)
        [4, 7, 8, 6, 9, 10, 11, 12, 5]

        >>> # add no vertices, in two different ways
        >>> me.insert_vertices(0, None)
        >>> me.insert_vertices(0, [])
        >>> print(me.vertices)
        [4, 7, 8, 6, 9, 10, 11, 12, 5]

        >>> # try to insert some invalid vertices
        >>> me.insert_vertices(0, 'one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> me.insert_vertices(0, 6)
        Traceback (most recent call last):
            ...
        ValueError: 6 is already a vertex
        >>> me.insert_vertices(0, 13)
        Traceback (most recent call last):
            ...
        ValueError: vertex index 13 out of range
        >>> me.insert_vertices(0, -1)
        Traceback (most recent call last):
            ...
        ValueError: vertex index -1 out of range
        >>> me.insert_vertices(
        ...             0, [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        >>> msh.add_vertices([0.5, -0.5])
        >>> me.insert_vertices('one', 13)
        Traceback (most recent call last):
            ...
        TypeError: 'str' object cannot be interpreted as an integer
        """
        if vertices is None:
            return
        vertices = np.array(vertices, dtype=int, ndmin=1)
        if len(vertices) == 0:
            return
        vertices = np.flip(vertices.ravel())
        for v in vertices:
            if v in self.vertices:
                raise ValueError(f'{v} is already a vertex')
            if v < 0 or v >= self.mesh.num_vertices:
                raise ValueError(f'vertex index {v} out of range')
            self.vertices.insert(index, int(v))
        self.mesh.mesh_valid = False

    def remove_vertices(self, remove_vertices):
        """Remove one or more vertex indices from the :c:`MeshEdge2D`.

        Parameters
        ----------
        remove_vertices : int | list[int]
            The vertex or list of vertices to remove from :a:`vertices`.

        Note
        -----
        Before removing the values in **remove_vertices**, an attempt will be
        made to cast it to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        ValueError
            If **remove_vertices** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **remove_vertices** cannot be cast to `int` or
            are not in :a:`vertices`.

        Examples
        --------
        >>> # create mesh and mesh edge, add/remove some vertices
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.add_vertices([[0.1, 0.1], [0.5, 0.65], [0.8, 0.8]])
        >>> me = vcfempy.meshgen.MeshEdge2D(msh)
        >>> me.insert_vertices(0, [4, 5, 6])
        >>> me.remove_vertices(5)
        >>> print(me.vertices)
        [4, 6]

        >>> # remove multiple vertices
        >>> msh.add_vertices([0.25, 0.9])
        >>> me.insert_vertices(1, [7, 5])
        >>> me.remove_vertices([4, 6])
        >>> print(me.vertices)
        [7, 5]

        >>> # the list of vertices to remove need not be 1d
        >>> # if not, it will be flattened
        >>> me.insert_vertices(0, 4)
        >>> me.insert_vertices(4, 6)
        >>> me.remove_vertices([[4, 5], [6, 7]])
        >>> print(me.vertices)
        []

        >>> # remove no vertices, in two different ways
        >>> me.insert_vertices(0, [4, 7, 5, 6])
        >>> me.remove_vertices(None)
        >>> me.remove_vertices([])
        >>> print(me.vertices)
        [4, 7, 5, 6]

        >>> # try to remove some invalid vertices
        >>> me.remove_vertices('one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> me.remove_vertices(1)
        Traceback (most recent call last):
            ...
        ValueError: list.remove(x): x not in list
        >>> me.remove_vertices(
        ...                 [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        """
        if remove_vertices is None:
            return
        remove_vertices = np.array(remove_vertices, dtype=int, ndmin=1)
        if len(remove_vertices) == 0:
            return
        remove_vertices = np.unique(remove_vertices)
        for rv in remove_vertices:
            self.vertices.remove(rv)
        self.mesh.mesh_valid = False

    def plot(self, ax=None, **kwargs):
        """Plot the :c:`MeshEdge2D`.

        Parameters
        ----------
        ax : None | :c:`matplotlib.axes.Axes`
            The axes to plot on. If not provided, will try to get one using
            :m:`matplotlib.pyplot.gca`.

        Other Parameters
        ----------------
        **kwargs : :c:`matplotlib.pyplot.Line2D` properties, optional
            Default values:
            `linewidth` = 3.0,
            `linestyle` = '--',
            `color` = :a:`material` `color` (or 'black' if :a:`material` is
            ``None``),
            `marker` = 's',
            `markersize` = 8.0,
            `markeredgecolor` = 'black',
            `markerfacecolor` = :a:`material` `color` (or 'black' if
            :a:`material` is ``None``)

        Returns
        -------
        :c:`matplotlib.axes.Axes`
            The axes that the :c:`MeshEdge2D` was plotted on.

        Examples
        --------
        >>> # initialize a mesh, a material region, and a mesh edge, then
        >>> # plot the material region and mesh edge
        >>> import matplotlib.pyplot as plt
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock', color='xkcd:stone')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, msh.boundary_vertices,
        ...                                       rock, 'rock region')
        >>> rock_joint = vcfempy.materials.Material('rock joint',
        ...                                         color='xkcd:greenish')
        >>> msh.add_vertices([[0.1, 0.1], [0.5, 0.65], [0.8, 0.8]])
        >>> me = vcfempy.meshgen.MeshEdge2D(msh, [4, 5, 6], rock_joint,
        ...                                 'rock joint')
        >>> fig = plt.figure()
        >>> ax = mr.plot()
        >>> ax = me.plot()
        >>> xmin, xmax, ymin, ymax = ax.axis('equal')
        >>> xtext = ax.set_xlabel('x')
        >>> ytext = ax.set_ylabel('y')
        >>> ttext = ax.set_title('MeshEdge2D Test Plot')
        >>> leg = ax.legend(labels=[mr.name, me.name])
        >>> plt.savefig('MeshEdge2D_test_plot.png')
        """
        if ax is None or not isinstance(ax, plt.Axes):
            ax = plt.gca()
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 3.0
        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = '--'
        if self.material is not None:
            kwargs['color'] = self.material.color
        elif 'color' not in kwargs.keys():
            kwargs['color'] = 'black'
        if 'marker' not in kwargs.keys():
            kwargs['marker'] = 's'
        if 'markersize' not in kwargs.keys():
            kwargs['markersize'] = 8.0
        if 'markeredgecolor' not in kwargs.keys():
            kwargs['markeredgecolor'] = 'black'
        if self.material is not None:
            kwargs['markerfacecolor'] = self.material.color
        elif 'markerfacecolor' not in kwargs.keys():
            kwargs['markerfacecolor'] = 'black'
        ax.plot(self.mesh.vertices[self.vertices, 0],
                self.mesh.vertices[self.vertices, 1],
                **kwargs)
        return ax


class PolyElement2D():
    """A class for polygonal element geometry and quadrature generation. Used
    by :c:`PolyMesh2D` to generate polygonal meshes.

    Parameters
    ----------
    mesh : :c:`vcfempy.meshgen.PolyMesh2D`
        The parent mesh.
    nodes : list[int], optional
        The list of node indices from the parent mesh. Can be in CW or CCW
        order.
    material : :c:`vcfempy.materials.Material`, optional
        The material type assigned to the element.

    Examples
    --------
    >>> # create a simple mesh and check the element properties
    >>> import vcfempy.materials
    >>> import vcfempy.meshgen
    >>> msh = vcfempy.meshgen.PolyMesh2D()
    >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
    >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
    >>> rock = vcfempy.materials.Material('rock')
    >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, [0, 1, 2, 3], rock)
    >>> msh.mesh_scale = 0.4
    >>> msh.add_seed_points([0.5, 0.5])
    >>> msh.generate_mesh()
    >>> print(msh.elements[0].mesh is msh)
    True
    >>> print(msh.elements[0].num_nodes)
    4
    >>> print(msh.elements[0].nodes)
    [15, 14, 6, 10]
    >>> print(msh.elements[0].material.name)
    rock
    >>> print(np.round(msh.elements[0].area, 14))
    0.09
    >>> print(msh.elements[0].centroid.round(14))
    [0.5 0.5]
    >>> print(msh.elements[0].quad_points.round(14))
    [[-0.1125 -0.1125]
     [-0.1125  0.1125]
     [ 0.1125  0.1125]
     [ 0.1125 -0.1125]
     [-0.075   0.    ]
     [ 0.      0.075 ]
     [ 0.075   0.    ]
     [ 0.     -0.075 ]
     [ 0.      0.    ]]
    >>> print(msh.elements[0].quad_weights.round(14))
    [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
     0.10083037 0.10083037 0.09371293]
    >>> print(np.sum(msh.elements[0].quad_weights).round(14))
    1.0
    """

    def __init__(self, mesh, nodes=None, material=None):
        # initialize parent mesh
        if not isinstance(mesh, PolyMesh2D):
            raise TypeError('type(mesh) is not vcfempy.meshgen.PolyMesh2D')
        self._mesh = mesh

        # initialize nodes
        self._nodes = []
        self.insert_nodes(0, nodes)

        # initialize material
        self.material = material

        # initialize geometry and quadrature attributes
        self.invalidate_properties()

    @property
    def mesh(self):
        """The parent :c:`PolyMesh2D`.

        Returns
        -------
        :c:`PolyMesh2D`
            The parent mesh assigned to the :c:`PolyElement2D`.

        Note
        ----
        The :a:`mesh` is immutable and can only be assigned when the
        :c:`PolyElement2D` is created. A :c:`PolyElement2D` should not
        usually be created explicitly, but rather should be created indirectly
        by calling the :m:`PolyMesh2D.generate_mesh` method.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0].mesh is msh)
        True
        """
        return self._mesh

    @property
    def material(self):
        """Material type assigned to the :c:`PolyElement2D`.

        Parameters
        ----------
        material : ``None`` | :c:`vcfempy.materials.Material`
            The material to assign to the :c:`PolyElement2D`.

        Returns
        -------
        ``None`` | :c:`vcfempy.materials.Material`
            The material assigned to the :c:`PolyElement2D`.

        Raises
        ------
        TypeError
            If type(material) not in [`NoneType`,
            :c:`vcfempy.materials.Material`]

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, [0, 1, 2, 3], rock)
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0].material.name)
        rock
        """
        return self._material

    @material.setter
    def material(self, material):
        if not isinstance(material, (type(None), mtl.Material)):
            raise TypeError('type(material) not in [NoneType, '
                            + 'vcfempy.materials.Material]')
        self._material = material

    @property
    def num_nodes(self):
        """Number of nodes in the :c:`PolyElement2D`.

        Returns
        -------
        `int`
            The number of nodes in the :c:`PolyElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0].num_nodes)
        4
        """
        return len(self.nodes)

    @property
    def nodes(self):
        """List of node indices in the :c:`PolyElement2D`. References the
        :a:`PolyMesh2D.nodes` of the parent :a:`mesh`.

        Returns
        -------
        `list[int]`
            The list of node indices in the :c:`PolyElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0].nodes)
        [15, 14, 6, 10]
        """
        return self._nodes

    def insert_nodes(self, index, nodes):
        """Insert one or more node indices to the :c:`PolyElement2D`.

        Parameters
        ----------
        index : int
            The index at which to insert the **nodes** into :a:`nodes`.
        nodes : int | list[int]
            The list of node indices to add to :a:`nodes`.

        Note
        -----
        Before inserting the values in **nodes**, an attempt is made to
        cast to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        TypeError
            If **index** cannot be interpreted as `int`.
        ValueError
            If **nodes** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **nodes** cannot be cast to `int`, are already
            in :a:`nodes`, are negative, or are >= :a:`mesh.num_nodes`.

        Examples
        --------
        >>> # create a simple mesh
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.nodes.round(14))
        [[-0.    1.  ]
         [ 0.35  1.  ]
         [ 0.    0.  ]
         [ 0.35  0.  ]
         [ 1.    1.  ]
         [ 0.65  1.  ]
         [ 0.65  0.65]
         [ 1.    0.65]
         [ 1.   -0.  ]
         [ 0.65 -0.  ]
         [ 0.65  0.35]
         [ 1.    0.35]
         [ 0.    0.65]
         [ 0.    0.35]
         [ 0.35  0.65]
         [ 0.35  0.35]]

        >>> # create a new element
        >>> # note, this is normally not done explicitly, but is shown here
        >>> # for testing and documentation
        >>> e = vcfempy.meshgen.PolyElement2D(msh)
        >>> print(e.nodes)
        []
        >>> e.insert_nodes(0, [0, 1, 14, 12])
        >>> print(e.nodes)
        [0, 1, 14, 12]
        >>> print(np.round(e.area, 14))
        0.1225

        >>> # insert no nodes in multiple ways
        >>> e.insert_nodes(0, None)
        >>> e.insert_nodes(0, [])
        >>> print(e.nodes)
        [0, 1, 14, 12]

        >>> # try to insert some invalid nodes
        >>> e.insert_nodes(0, 'one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> e.insert_nodes(0, 1)
        Traceback (most recent call last):
            ...
        ValueError: 1 is already a node
        >>> e.insert_nodes(0, 16)
        Traceback (most recent call last):
            ...
        ValueError: node index 16 out of range
        >>> e.insert_nodes(0, -1)
        Traceback (most recent call last):
            ...
        ValueError: node index -1 out of range
        >>> e.insert_nodes(
        ...             0, [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        >>> e.insert_nodes('one', 2)
        Traceback (most recent call last):
            ...
        TypeError: 'str' object cannot be interpreted as an integer
        """
        if nodes is None:
            return
        nodes = np.array(nodes, dtype=int, ndmin=1)
        if len(nodes) == 0:
            return
        self.invalidate_properties()
        nodes = np.flip(nodes.ravel())
        for n in nodes:
            if n in self.nodes:
                raise ValueError(f'{n} is already a node')
            if n < 0 or n >= self.mesh.num_nodes:
                raise ValueError(f'node index {n} out of range')
            self.nodes.insert(index, int(n))

    def remove_nodes(self, remove_nodes):
        """Remove one or more node indices from the :c:`PolyElement2D`.

        Parameters
        ----------
        remove_nodes : int | list[int]
            The node or list of nodes to remove from :a:`nodes`.

        Note
        -----
        Before removing the values in **remove_nodes**, an attempt will be
        made to cast it to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        ValueError
            If **remove_nodes** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **remove_nodes** cannot be cast to `int` or
            are not in :a:`nodes`.

        Examples
        --------
        >>> # create a simple mesh, and remove a node from an element
        >>> # this should not normally be done explicitly unless you know
        >>> # what you are doing
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.02
        >>> msh.generate_mesh()
        >>> print(msh.elements[0].nodes)
        [714, 710, 711, 712, 713]
        >>> msh.elements[0].remove_nodes(714)
        >>> print(msh.elements[0].nodes)
        [710, 711, 712, 713]

        >>> # remove multiple nodes
        >>> print(msh.elements[1].nodes)
        [1328, 276, 81, 714, 710, 1327]
        >>> msh.elements[1].remove_nodes([276, 81])
        >>> print(msh.elements[1].nodes)
        [1328, 714, 710, 1327]

        >>> # the list of nodes to remove need not be 1d
        >>> # if not, it will be flattened
        >>> i7 = msh.num_nodes_per_element.index(7)
        >>> print(msh.elements[i7].nodes)
        [446, 162, 7, 9, 8, 10, 445]
        >>> msh.elements[i7].remove_nodes([[7, 8], [9, 10]])
        >>> print(msh.elements[i7].nodes)
        [446, 162, 445]

        >>> # remove no nodes, in two different ways
        >>> msh.elements[0].remove_nodes(None)
        >>> msh.elements[0].remove_nodes([])
        >>> print(msh.elements[0].nodes)
        [710, 711, 712, 713]

        >>> # try to remove some invalid nodes
        >>> msh.elements[0].remove_nodes('one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> msh.elements[0].remove_nodes(4)
        Traceback (most recent call last):
            ...
        ValueError: list.remove(x): x not in list
        >>> msh.elements[0].remove_nodes(
        ...                 [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        """
        if remove_nodes is None:
            return
        remove_nodes = np.array(remove_nodes, dtype=int, ndmin=1)
        if len(remove_nodes) == 0:
            return
        self.invalidate_properties()
        remove_nodes = remove_nodes.ravel()
        for rn in remove_nodes:
            self.nodes.remove(rn)

    @property
    def area(self):
        """The area of the :c:`PolyElement2D`.

        Returns
        -------
        `float`
            The area of the :c:`PolyElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(np.round(msh.elements[0].area, 14))
        0.09
        """
        if self._area is None:
            self._area = shp.Polygon(self.mesh.nodes[self.nodes]).area
        return self._area

    @property
    def centroid(self):
        """The centroid coordinates of the :c:`PolyElement2D`.

        Returns
        -------
        `numpy.ndarray`, shape = (2, )
            The coordinates of the centroid of the :c:`PolyElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0].centroid)
        [0.5 0.5]
        """
        if self._centroid is None:
            c = shp.Polygon(self.mesh.nodes[self.nodes]).centroid
            self._centroid = np.array([c.x, c.y], dtype=float)
        return self._centroid

    @property
    def num_quad_points(self):
        """The number of quadrature points in the :c:`PolyElement2D`.

        Returns
        -------
        `int`
            The number of quadrature points in the :c:`PolyElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0].num_quad_points)
        9
        """
        if self._quad_points is None:
            self.generate_quadrature()
        return len(self.quad_points)

    @property
    def quad_points(self):
        """The quadrature point coordinates for the :c:`PolyElement2D`.

        Returns
        -------
        `numpy.ndarray`, shape = (:a:`num_quad_points`, 2)
            The coordinates of the quadrature points for the
            :c:`PolyElement2D`.

        Note
        ----
        The quadrature point coordinates are provided in a local coordinate
        system with the :a:`centroid` at the origin.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0].quad_points.round(14))
        [[-0.1125 -0.1125]
         [-0.1125  0.1125]
         [ 0.1125  0.1125]
         [ 0.1125 -0.1125]
         [-0.075   0.    ]
         [ 0.      0.075 ]
         [ 0.075   0.    ]
         [ 0.     -0.075 ]
         [ 0.      0.    ]]
        """
        if self._quad_points is None:
            self.generate_quadrature()
        return self._quad_points

    @property
    def quad_weights(self):
        """The quadrature point weights for the :c:`PolyElement2D`.

        Returns
        -------
        `numpy.ndarray`, shape = (:a:`num_quad_points`, )
            The weights of the quadrature points in the :c:`PolyElement2D`.

        Note
        ----
        The quadrature point weights should always sum to ``1.0``.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0].quad_weights)
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        >>> print(np.sum(msh.elements[0].quad_weights).round(14))
        1.0
        """
        if self._quad_weights is None:
            self.generate_quadrature()
        return self._quad_weights

    @property
    def num_quad_integrals(self):
        """The number of quadrature basis function integrals used in
        generating the quadrature for the :c:`PolyElement2D`.

        Returns
        -------
        `int`
            The number of quadrature basis function integrals for the
            :c:`PolyElement2D`.

        Note
        ----
        The number of quadrature integrals corresponds to the number of basis
        functions used to develop the quadrature rule. The number depends on
        the order of interpolation for the flux field in the hybrid finite
        element approach. The flux field interpolation uses complete
        polynomials in 2D and the order depends on the number of nodes in the
        element (and whether :a:`PolyMesh2D.high_order_quadrature` is ``True``
        for the parent :a:`mesh`). In integrating finite element matrices, it
        is necessary to integrate terms containing the square of the flux
        field basis functions, so the quadrature rule must increment the
        maximum order of basis function by 2 for each increase in order of
        flux field interpolation. The number of basis functions is as follows:
        [3 nodes: constant: 1 basis function, 4-5 nodes: quadratic: 6 basis
        functions, 6-7 nodes: 4th order: 15 basis functions, 8-10 nodes:
        6th order: 28 basis functions, :a:`PolyMesh2D.high_order_quadrature`
        for :a:`mesh`: 6th order: 28 basis functions].

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.02
        >>> msh.mesh_rand = 0.7
        >>> msh.generate_mesh()
        >>> i3 = msh.num_nodes_per_element.index(3)
        >>> print(msh.elements[i3].num_quad_integrals)
        1
        >>> i5 = msh.num_nodes_per_element.index(5)
        >>> print(msh.elements[i5].num_quad_integrals)
        6
        >>> i7 = msh.num_nodes_per_element.index(7)
        >>> print(msh.elements[i7].num_quad_integrals)
        15
        >>> i8 = msh.num_nodes_per_element.index(8)
        >>> print(msh.elements[i8].num_quad_integrals)
        28

        >>> # set high order quadrature and check number of basis functions
        >>> msh.high_order_quadrature = True
        >>> msh.generate_mesh()
        >>> i3 = msh.num_nodes_per_element.index(3)
        >>> print(msh.elements[i3].num_quad_integrals)
        28
        >>> i5 = msh.num_nodes_per_element.index(5)
        >>> print(msh.elements[i5].num_quad_integrals)
        28
        >>> i7 = msh.num_nodes_per_element.index(7)
        >>> print(msh.elements[i7].num_quad_integrals)
        28
        >>> i8 = msh.num_nodes_per_element.index(8)
        >>> print(msh.elements[i8].num_quad_integrals)
        28
        """
        if self._quad_integrals is None:
            self.generate_quadrature()
        return len(self.quad_integrals)

    @property
    def quad_integrals(self):
        """The quadrature basis function integrals for the :c:`PolyElement2D`.

        Returns
        -------
        `numpy.ndarray`, size = (:a:`num_quad_integrals`, )
            The values of the element quadrature basis function integrals for
            the :c:`PolyElement2D`.

        See Also
        --------
        :a:`num_quad_integrals`
            For explanatory **Note** on number of basis functions.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> # note that the first basis function integral is always equal
        >>> # to the element area
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0].quad_integrals.round(14))
        [ 0.09     -0.        0.        0.000675  0.        0.000675]
        >>> print(np.round(msh.elements[0].area, 14))
        0.09
        """
        if self._quad_integrals is None:
            self.generate_quadrature()
        return self._quad_integrals

    def invalidate_properties(self):
        """Resets cached values of computed attributes :a:`area`,
        :a:`centroid`, :a:`quad_points`, :a:`quad_weights`, and
        :a:`quad_integrals` for the :c:`PolyElement2D`.

        Note
        ----
        The :m:`invalidate_properties` method should be called whenever
        :a:`nodes` is changed. This is done by :m:`insert_nodes` and
        :m:`remove_nodes`, but needs to be done explicitly if making manual
        changes to :a:`nodes`.

        Examples
        --------
        >>> # create a simple mesh, check the element properties
        >>> # invalidate properties and check the values of (private) cache
        >>> # attributes
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0]._area)
        None
        >>> print(msh.elements[0]._centroid)
        None
        >>> print(msh.elements[0]._quad_points)
        None
        >>> print(msh.elements[0]._quad_weights)
        None
        >>> print(msh.elements[0]._quad_integrals)
        None
        >>> print(np.round(msh.elements[0].area, 14))
        0.09
        >>> print(msh.elements[0].centroid.round(14))
        [0.5 0.5]
        >>> print(msh.elements[0].quad_points.round(14))
        [[-0.1125 -0.1125]
         [-0.1125  0.1125]
         [ 0.1125  0.1125]
         [ 0.1125 -0.1125]
         [-0.075   0.    ]
         [ 0.      0.075 ]
         [ 0.075   0.    ]
         [ 0.     -0.075 ]
         [ 0.      0.    ]]
        >>> print(msh.elements[0].quad_weights.round(14))
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        >>> print(msh.elements[0].quad_integrals.round(14))
        [ 0.09     -0.        0.        0.000675  0.        0.000675]
        >>> msh.elements[0].invalidate_properties()
        >>> print(msh.elements[0]._area)
        None
        >>> print(msh.elements[0]._centroid)
        None
        >>> print(msh.elements[0]._quad_points)
        None
        >>> print(msh.elements[0]._quad_weights)
        None
        >>> print(msh.elements[0]._quad_integrals)
        None
        """
        self._area = None
        self._centroid = None
        self._quad_points = None
        self._quad_weights = None
        self._quad_integrals = None

    def generate_quadrature(self):
        """Generate quadrature points and weights for the :c:`PolyElement2D`.

        Note
        ----
        The :m:`generate_quadrature` method determines the correct (private)
        :m:`_quadconX` method to call depending on the values of
        :a:`num_nodes` and :a:`PolyMesh2D.high_order_quadrature` for
        :a:`mesh`. Elements with 8 or more nodes (or
        :a:`PolyMesh2D.high_order_quadrature` set) call :m:`_quadcon10`,
        elements with 6-7 nodes call :m:`_quadcon7`, elements with 4-5 nodes
        call :m:`_quadcon5`, and elements with 3 nodes call :m:`_quadcon3`.
        This method does not have a direct return value, but sets the cached
        values of private attributes corresponding to :a:`area`,
        :a:`centroid`, :a:`quad_points`, :a:`quad_weights`, and
        :a:`quad_integrals`.

        Examples
        --------
        >>> # create a simple mesh and check the element (private) attributes
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.elements[0]._area)
        None
        >>> print(msh.elements[0]._centroid)
        None
        >>> print(msh.elements[0]._quad_points)
        None
        >>> print(msh.elements[0]._quad_weights)
        None
        >>> print(msh.elements[0]._quad_integrals)
        None
        >>> msh.elements[0].generate_quadrature()
        >>> print(np.round(msh.elements[0]._area, 14))
        0.09
        >>> print(msh.elements[0]._centroid.round(14))
        [0.5 0.5]
        >>> print(msh.elements[0]._quad_points.round(14))
        [[-0.1125 -0.1125]
         [-0.1125  0.1125]
         [ 0.1125  0.1125]
         [ 0.1125 -0.1125]
         [-0.075   0.    ]
         [ 0.      0.075 ]
         [ 0.075   0.    ]
         [ 0.     -0.075 ]
         [ 0.      0.    ]]
        >>> print(msh.elements[0]._quad_weights.round(14))
        [0.1257414  0.1257414  0.1257414  0.1257414  0.10083037 0.10083037
         0.10083037 0.10083037 0.09371293]
        >>> print(msh.elements[0]._quad_integrals.round(14))
        [ 0.09     -0.        0.        0.000675  0.        0.000675]
        """

        n = self.num_nodes

        if self.mesh.high_order_quadrature or n > 7:
            self._quadcon10()
        elif n > 5:
            self._quadcon7()
        elif n > 3:
            self._quadcon5()
        else:
            self._quadcon3()

    def _quadcon3(self):
        # only require linear integration over a triangle, one integration
        # point is sufficient, ensure that area and centroid are set
        _ = self.centroid
        self._quad_points = np.zeros((1, 2))
        self._quad_weights = np.array([1.])
        self._quad_integrals = np.array([self.area])

    def _quadcon5(self):
        vertices = self.mesh.nodes[self.nodes] - self.centroid
        cent = np.zeros(2)
        # integrate basis functions
        # f(x,y) = {1, x, y, x**2, x*y, y**2}
        # using subdivision of polygon into triangles
        # each triangle is integrated using Gaussian quadrature
        # as per: Cowper, G.R. 1973. Gaussian quadrature formulas for
        #             triangles, International Journal for Numerical
        #             Methods 7(3): 405-408, doi: 10.1002/nme.1620070316
        # here, use the 3-point formula  with degree of precision 2
        N = np.array([[0.66666_66666_66667,
                       0.16666_66666_66667,
                       0.16666_66666_66667],
                      [0.16666_66666_66667,
                       0.66666_66666_66667,
                       0.16666_66666_66667],
                      [0.16666_66666_66667,
                       0.16666_66666_66667,
                       0.66666_66666_66667]])
        w = np.array([0.33333_33333_33333,
                      0.33333_33333_33333,
                      0.33333_33333_33333])
        nphi = 6
        phi = np.zeros(nphi)
        # perform Gaussian integration over triangles
        n = len(vertices)
        for k, v0 in enumerate(vertices):
            v1 = vertices[(k+1) % n]
            x = np.vstack([cent, v0, v1])
            area = shp.Polygon(x).area
            for wj, Nj in zip(w, N):
                xj = Nj @ x
                phi += area * wj * np.array([1.,
                                             xj[0],
                                             xj[1],
                                             xj[0]**2,
                                             xj[0] * xj[1],
                                             xj[1]**2])
        # initialize polygon integration points
        # this produces a 9-point integration rule for quadrilaterals
        # and an 11-point integration rule for pentagons
        xq0 = []
        for v in vertices:
            d = cent - v
            xq0.append(v + 0.25*d)
        xq0 = np.array(xq0)
        mid_xq0 = []
        nq0 = len(xq0)
        for k, x0 in enumerate(xq0):
            x1 = xq0[(k+1) % nq0]
            mid_xq0.append((x0+x1+cent)/3)
        xq = np.vstack([xq0, mid_xq0, cent])
        nq = len(xq)
        # evaluate basis functions at integration points
        PHI = np.array([np.ones(nq),
                        xq[:, 0],
                        xq[:, 1],
                        xq[:, 0]**2,
                        xq[:, 0] * xq[:, 1],
                        xq[:, 1]**2])
        # solve for the quadrature coefficients and normalize integration
        # point weights, this is a least squares solution
        wq = np.linalg.lstsq(PHI, phi, rcond=None)[0] / self.area
        # set element quadrature (private) attributes
        self._quad_points = xq
        self._quad_weights = wq
        self._quad_integrals = phi

    def _quadcon7(self):
        vertices = self.mesh.nodes[self.nodes] - self.centroid
        cent = np.zeros(2)
        # integrate basis functions
        # f(x,y) = { 1,
        #            x, y,
        #            x**2, x * y, y**2,
        #            x**3, x**2 * y, x * y**2, y**3,
        #            x**4, x**3 * y, x**2 * y**2, x * y**3, y**4}
        # using subdivision of polygon into triangles
        # each triangle is integrated using Gaussian quadrature
        # as per: Cowper, G.R. 1973. Gaussian quadrature formulas for
        #             triangles, International Journal for Numerical
        #             Methods 7(3): 405-408, doi: 10.1002/nme.1620070316
        # here, use the 6-point formula  with degree of precision 4
        N = np.array([[0.81684_75729_80459,
                       0.09157_62135_09771,
                       0.09157_62135_09771],
                      [0.09157_62135_09771,
                       0.81684_75729_80459,
                       0.09157_62135_09771],
                      [0.09157_62135_09771,
                       0.09157_62135_09771,
                       0.81684_75729_80459],
                      [0.10810_30181_68070,
                       0.44594_84909_15965,
                       0.44594_84909_15965],
                      [0.44594_84909_15965,
                       0.10810_30181_68070,
                       0.44594_84909_15965],
                      [0.44594_84909_15965,
                       0.44594_84909_15965,
                       0.10810_30181_68070]])
        w = np.array([0.10995_17436_55322,
                      0.10995_17436_55322,
                      0.10995_17436_55322,
                      0.22338_15896_78011,
                      0.22338_15896_78011,
                      0.22338_15896_78011])
        nphi = 15
        phi = np.zeros(nphi)
        # perform Gaussian integration over triangles
        n = len(vertices)
        for k, v0 in enumerate(vertices):
            v1 = vertices[(k+1) % n]
            x = np.vstack([cent, v0, v1])
            area = shp.Polygon(x).area
            for wj, Nj in zip(w, N):
                xj = Nj @ x
                phi += area * wj * np.array([1.,
                                             xj[0],
                                             xj[1],
                                             xj[0]**2,
                                             xj[0] * xj[1],
                                             xj[1]**2,
                                             xj[0]**3,
                                             xj[0]**2 * xj[1],
                                             xj[0] * xj[1]**2,
                                             xj[1]**3,
                                             xj[0]**4,
                                             xj[0]**3 * xj[1],
                                             xj[0]**2 * xj[1]**2,
                                             xj[0] * xj[1]**3,
                                             xj[1]**4])
        # initialize polygon integration points
        # this produces a 19-point integration rule for hexagons
        # and a 22-point integration rule for heptagons
        xq0 = []
        for v in vertices:
            d = cent - v
            xq0.append(v + 0.25*d)
        xq0 = np.array(xq0)
        mid_xq0 = []
        nq0 = len(xq0)
        for k, x0 in enumerate(xq0):
            x1 = xq0[(k+1) % nq0]
            mid_xq0.append(0.5*(x0+x1))
        mid_xq0 = np.array(mid_xq0)
        tri_xq0 = []
        for x in mid_xq0:
            tri_xq0.append(0.5*(cent + x))
        xq = np.vstack([xq0, mid_xq0, tri_xq0, cent])
        nq = len(xq)
        # evaluate basis functions at integration points
        PHI = np.array([np.ones(nq),
                        xq[:, 0],
                        xq[:, 1],
                        xq[:, 0]**2,
                        xq[:, 0] * xq[:, 1],
                        xq[:, 1]**2,
                        xq[:, 0]**3,
                        xq[:, 0]**2 * xq[:, 1],
                        xq[:, 0] * xq[:, 1]**2,
                        xq[:, 1]**3,
                        xq[:, 0]**4,
                        xq[:, 0]**3 * xq[:, 1],
                        xq[:, 0]**2 * xq[:, 1]**2,
                        xq[:, 0] * xq[:, 1]**3,
                        xq[:, 1]**4])
        # solve for the quadrature coefficients and normalize integration
        # point weights, this is a least squares solution
        wq = np.linalg.lstsq(PHI, phi, rcond=None)[0] / self.area
        # set element quadrature (private) attributes
        self._quad_points = xq
        self._quad_weights = wq
        self._quad_integrals = phi

    def _quadcon10(self):
        vertices = self.mesh.nodes[self.nodes] - self.centroid
        cent = np.zeros(2)
        # integrate basis functions
        # f(x,y) = { 1,
        #            x, y,
        #            x**2, x * y, y**2,
        #            x**3, x**2 * y, x * y**2, y**3,
        #            x**4, x**3 * y, x**2 * y**2, x * y**3, y**4,
        #            x**5, x**4 * y, x**3 * y**2, x**2 * y**3, x * y**4, y**5,
        #            x**6, x**5 * y, x**4 * y**2, x**3 * y**3,
        #            x**2 * y**4, x * y**5, y**6}
        # using subdivision of polygon into triangles
        # each triangle is integrated using Gaussian quadrature
        # as per: Cowper, G.R. 1973. Gaussian quadrature formulas for
        #             triangles, International Journal for Numerical
        #             Methods 7(3): 405-408, doi: 10.1002/nme.1620070316
        # here, use the 12-point formula  with degree of precision 6
        N = np.array([[0.87382_19710_16996,
                       0.06308_90144_91502,
                       0.06308_90144_91502],
                      [0.06308_90144_91502,
                       0.87382_19710_16996,
                       0.06308_90144_91502],
                      [0.06308_90144_91502,
                       0.06308_90144_91502,
                       0.87382_19710_16996],
                      [0.50142_65096_58179,
                       0.24928_67451_70911,
                       0.24928_67451_70911],
                      [0.24928_67451_70911,
                       0.50142_65096_58179,
                       0.24928_67451_70911],
                      [0.24928_67451_70911,
                       0.24928_67451_70911,
                       0.50142_65096_58179],
                      [0.63650_24991_21399,
                       0.31035_24510_33785,
                       0.05314_50498_44816],
                      [0.63650_24991_21399,
                       0.05314_50498_44816,
                       0.31035_24510_33785],
                      [0.31035_24510_33785,
                       0.63650_24991_21399,
                       0.05314_50498_44816],
                      [0.31035_24510_33785,
                       0.05314_50498_44816,
                       0.63650_24991_21399],
                      [0.05314_50498_44816,
                       0.63650_24991_21399,
                       0.31035_24510_33785],
                      [0.05314_50498_44816,
                       0.31035_24510_33785,
                       0.63650_24991_21399]])
        w = np.array([0.05084_49063_70207,
                      0.05084_49063_70207,
                      0.05084_49063_70207,
                      0.11678_62757_26379,
                      0.11678_62757_26379,
                      0.11678_62757_26379,
                      0.08285_10756_18374,
                      0.08285_10756_18374,
                      0.08285_10756_18374,
                      0.08285_10756_18374,
                      0.08285_10756_18374,
                      0.08285_10756_18374])
        nphi = 28
        phi = np.zeros(nphi)
        # perform Gaussian integration over triangles
        n = len(vertices)
        for k, v0 in enumerate(vertices):
            v1 = vertices[(k+1) % n]
            x = np.vstack([cent, v0, v1])
            area = shp.Polygon(x).area
            for wj, Nj in zip(w, N):
                xj = Nj @ x
                phi += area * wj * np.array([1.,
                                             xj[0],
                                             xj[1],
                                             xj[0]**2,
                                             xj[0] * xj[1],
                                             xj[1]**2,
                                             xj[0]**3,
                                             xj[0]**2 * xj[1],
                                             xj[0] * xj[1]**2,
                                             xj[1]**3,
                                             xj[0]**4,
                                             xj[0]**3 * xj[1],
                                             xj[0]**2 * xj[1]**2,
                                             xj[0] * xj[1]**3,
                                             xj[1]**4,
                                             xj[0]**5,
                                             xj[0]**4 * xj[1],
                                             xj[0]**3 * xj[1]**2,
                                             xj[0]**2 * xj[1]**3,
                                             xj[0] * xj[1]**4,
                                             xj[1]**5,
                                             xj[0]**6,
                                             xj[0]**5 * xj[1],
                                             xj[0]**4 * xj[1]**2,
                                             xj[0]**3 * xj[1]**3,
                                             xj[0]**2 * xj[1]**4,
                                             xj[0] * xj[1]**5,
                                             xj[1]**6])
        # initialize polygon integration points
        # this produces a 33-point integration rule for octagons,
        # a 37-point integration rule for nonagons, and
        # a 41-point integration rule for decagons
        xq0 = []
        for v in vertices:
            d = cent - v
            xq0.append(v + 0.15*d)
        xq0 = np.array(xq0)
        Ntri = np.array([[0.6, 0.2, 0.2],
                         [0.2, 0.6, 0.2],
                         [0.2, 0.2, 0.6]])
        tri_xq0 = []
        nq0 = len(xq0)
        for k, x0 in enumerate(xq0):
            x1 = xq0[(k+1) % nq0]
            x = np.vstack([x0, x1, cent])
            for Nj in Ntri:
                tri_xq0.append(Nj @ x)
        xq = np.vstack([xq0, tri_xq0, cent])
        nq = len(xq)
        # evaluate basis functions at integration points
        PHI = np.array([np.ones(nq),
                        xq[:, 0],
                        xq[:, 1],
                        xq[:, 0]**2,
                        xq[:, 0]*xq[:, 1],
                        xq[:, 1]**2,
                        xq[:, 0]**3,
                        xq[:, 0]**2 * xq[:, 1],
                        xq[:, 0] * xq[:, 1]**2,
                        xq[:, 1]**3,
                        xq[:, 0]**4,
                        xq[:, 0]**3 * xq[:, 1],
                        xq[:, 0]**2 * xq[:, 1]**2,
                        xq[:, 0] * xq[:, 1]**3,
                        xq[:, 1]**4,
                        xq[:, 0]**5,
                        xq[:, 0]**4 * xq[:, 1],
                        xq[:, 0]**3 * xq[:, 1]**2,
                        xq[:, 0]**2 * xq[:, 1]**3,
                        xq[:, 0] * xq[:, 1]**4,
                        xq[:, 1]**5,
                        xq[:, 0]**6,
                        xq[:, 0]**5 * xq[:, 1],
                        xq[:, 0]**4 * xq[:, 1]**2,
                        xq[:, 0]**3 * xq[:, 1]**3,
                        xq[:, 0]**2 * xq[:, 1]**4,
                        xq[:, 0] * xq[:, 1]**5,
                        xq[:, 1]**6])
        # solve for the quadrature coefficients and normalize integration
        # point weights, this is a least squares solution
        wq = np.linalg.lstsq(PHI, phi, rcond=None)[0] / self.area
        # set element quadrature (private) attributes
        self._quad_points = xq
        self._quad_weights = wq
        self._quad_integrals = phi

    def plot(self, ax=None, **kwargs):
        """Plot the :c:`PolyElement2D` using :m:`matplotlib.pyplot.fill`.

        Parameters
        ----------
        ax : :c:`matplotlib.axes.Axes`, optional
            The axes to plot on. If not provided, will try to get one using
            :m:`matplotlib.pyplot.gca`.

        Other Parameters
        ----------------
        **kwargs : :c:`matplotlib.patches.Polygon` properties, optional
            Default values:
            `edgecolor` = :a:`material` `color` (or 'black' if :a:`material`
            is ``None``) with alpha = 1.0,
            `facecolor` = :a:`material` `color` (or 'black' if :a:`material`
            is ``None``) with alpha = 0.6,
            `linewidth` = 1.0,
            `linestyle` = ':'.

        Returns
        -------
        :c:`matplotlib.axes.Axes`
            The axes that the :c:`PolyElement2D` was plotted on.

        Examples
        --------
        >>> # initialize a mesh and material region, then plot the elements
        >>> import matplotlib.pyplot as plt
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock', color='xkcd:clay')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, msh.boundary_vertices,
        ...                                       rock, 'rock region')
        >>> msh.mesh_scale = 0.2
        >>> msh.mesh_rand = 0.2
        >>> msh.generate_mesh()
        >>> fig = plt.figure()
        >>> for e in msh.elements:
        ...     ax = e.plot()
        >>> xmin, xmax, ymin, ymax = ax.axis('equal')
        >>> xtext = ax.set_xlabel('x')
        >>> ytext = ax.set_ylabel('y')
        >>> ttext = ax.set_title('PolyElement2D Test Plot')
        >>> plt.savefig('PolyElement2D_test_plot.png')
        """
        if ax is None or not isinstance(ax, plt.Axes):
            ax = plt.gca()
        if self.material is not None:
            color = mplclr.to_rgb(self.material.color)
        else:
            color = mplclr.to_rgb('black')
        if 'edgecolor' not in kwargs.keys():
            kwargs['edgecolor'] = color + (1.0, )
        if 'facecolor' not in kwargs.keys():
            kwargs['facecolor'] = color + (0.6, )
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 1.0
        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = ':'
        ax.fill(self.mesh.nodes[self.nodes, 0],
                self.mesh.nodes[self.nodes, 1],
                **kwargs)
        return ax

    def plot_quad_points(self, ax=None, **kwargs):
        """Plot the :a:`quad_points` of the :c:`PolyElement2D` using
        :m:`matplotlib.pyplot.plot`.

        Parameters
        ----------
        ax : :c:`matplotlib.axes.Axes`, optional
            The axes to plot on. If not provided, will try to get one using
            :m:`matplotlib.pyplot.gca`.

        Other Parameters
        ----------------
        **kwargs : :c:`matplotlib.lines.Line2D` properties, optional
            Default values:
            `linewidth` = 0.0,
            `markeredgecolor` = :a:`material` `color` with alpha = 1.0 (or
            'black' if :a:`material` is ``None``),
            `markerfacecolor` = :a:`material` `color` with alpha = 1.0 (or
            'black' if :a:`material` is ``None``),
            `marker` = 'P',
            `markersize` = 6.0.

        Returns
        -------
        :c:`matplotlib.axes.Axes`
            The axes that the :a:`quad_points` were plotted on.

        Examples
        --------
        >>> # initialize a mesh and material region, then plot the elements
        >>> # and quad points
        >>> import matplotlib.pyplot as plt
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock', color='xkcd:greenish')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, msh.boundary_vertices,
        ...                                       rock, 'rock region')
        >>> msh.mesh_scale = 0.2
        >>> msh.mesh_rand = 0.2
        >>> msh.generate_mesh()
        >>> fig = plt.figure()
        >>> for e in msh.elements:
        ...     ax = e.plot()
        ...     ax = e.plot_quad_points()
        >>> xmin, xmax, ymin, ymax = ax.axis('equal')
        >>> xtext = ax.set_xlabel('x')
        >>> ytext = ax.set_ylabel('y')
        >>> ttext = ax.set_title('PolyElement2D Test Plot')
        >>> plt.savefig('PolyElement2D_quad_points_test_plot.png')
        """
        if ax is None or not isinstance(ax, plt.Axes):
            ax = plt.gca()
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 0.0
        if self.material is not None:
            color = mplclr.to_rgb(self.material.color)
            kwargs['markeredgecolor'] = color + (1.0, )
            kwargs['markerfacecolor'] = color + (1.0, )
        else:
            if 'markeredgecolor' not in kwargs.keys():
                kwargs['markeredgecolor'] = 'black'
            if 'markerfacecolor' not in kwargs.keys():
                kwargs['markerfacecolor'] = 'black'
        if 'marker' not in kwargs.keys():
            kwargs['marker'] = 'P'
        if 'markersize' not in kwargs.keys():
            kwargs['markersize'] = 6.0
        ax.plot(self.quad_points[:, 0] + self.centroid[0],
                self.quad_points[:, 1] + self.centroid[1],
                **kwargs)
        return ax


class InterfaceElement2D():
    """A class for interfaces between neighboring :c:`PolyElement2D` elements
    in a :c:`PolyMesh2D`.

    Parameters
    ----------
    mesh : :c:`PolyMesh2D`
        The parent mesh.
    nodes : list[int], optional
        Initial list of node indices from the parent mesh that are contained
        in the :c:`InterfaceElement2D`.
    material : :c:`vcfempy.materials.Material`, optional
        The material type to assign to the :c:`InterfaceElement2D`.
    neighbors: `list` of :c:`PolyElement2D`, optional
        List of neighboring :c:`PolyElement2D` from the parent mesh.
    width : float, optional
        The element width in the direction normal to the length of the
        :c:`InterfaceElement2D`.

    Examples
    --------
    >>> # create a simple mesh and check the element properties
    >>> import vcfempy.materials
    >>> import vcfempy.meshgen
    >>> msh = vcfempy.meshgen.PolyMesh2D()
    >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
    >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
    >>> rock = vcfempy.materials.Material('rock')
    >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, [0, 1, 2, 3], rock)
    >>> msh.mesh_scale = 0.4
    >>> msh.add_seed_points([0.5, 0.5])
    >>> msh.generate_mesh()
    >>> print(msh.interface_elements[0].mesh is msh)
    True
    >>> print(msh.interface_elements[0].num_nodes)
    2
    >>> print(msh.interface_elements[0].nodes)
    [5, 6]
    >>> print(msh.interface_elements[0].material.name)
    rock
    >>> print(np.round(msh.interface_elements[0].length, 14))
    0.35
    >>> print(msh.interface_elements[0].centroid.round(14))
    [0.65  0.825]
    """

    def __init__(self, mesh, nodes=None, material=None, neighbors=None,
                 width=0.0):
        if not isinstance(mesh, PolyMesh2D):
            raise TypeError('type(mesh) is not vcfempy.meshgen.PolyMesh2D')
        self._mesh = mesh

        self._material = None
        self.material = material

        self.width = width

        self.invalidate_properties()

        self._nodes = []
        self.insert_nodes(0, nodes)

        self._neighbors = []
        self.add_neighbors(neighbors)

    @property
    def mesh(self):
        """The parent :c:`PolyMesh2D`.

        Returns
        -------
        :c:`PolyMesh2D`
            The parent mesh assigned to the :c:`InterfaceElement2D`.

        Note
        ----
        The :a:`mesh` is immutable and can only be assigned when the
        :c:`InterfaceElement2D` is created. An :c:`InterfaceElement2D` should
        not usually be created explicitly, but rather should be created
        indirectly by calling the :m:`PolyMesh2D.generate_mesh` method.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.interface_elements[0].mesh is msh)
        True
        """
        return self._mesh

    @property
    def material(self):
        """Material type assigned to the :c:`InterfaceElement2D`.

        Parameters
        ----------
        material : ``None`` | :c:`vcfempy.materials.Material`
            The material to assign to the :c:`InterfaceElement2D`.

        Returns
        -------
        ``None`` | :c:`vcfempy.materials.Material`
            The material assigned to the :c:`InterfaceElement2D`.

        Raises
        ------
        TypeError
            If type(material) not in [`NoneType`,
            :c:`vcfempy.materials.Material`]

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, [0, 1, 2, 3], rock)
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.interface_elements[0].material.name)
        rock
        """
        return self._material

    @material.setter
    def material(self, material):
        if not isinstance(material, (type(None), mtl.Material)):
            raise TypeError('type(material) not in [NoneType, '
                            + 'vcfempy.materials.Material]')
        self._material = material

    @property
    def width(self):
        """The width of the :c:`InterfaceElement2D` in the normal direction.

        Parameters
        ----------
        val : float
            The width to assign to the :c:`InterfaceElement2D`.

        Returns
        -------
        `float`
            The width of the :c:`InterfaceElement2D`.

        Raises
        ------
        TypeError
            If **val** is not a `str` or a number.
        ValueError
            If **val** cannot be cast to `float` or is < 0.0.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, [0, 1, 2, 3], rock)
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.interface_elements[0].width)
        0.0
        >>> msh.interface_elements[0].width = 0.5
        >>> print(msh.interface_elements[0].width)
        0.5
        >>> msh.interface_elements[0].width = None
        Traceback (most recent call last):
            ...
        TypeError: float() argument must be a string or a number, \
not 'NoneType'
        >>> msh.interface_elements[0].width = -0.1
        Traceback (most recent call last):
            ...
        ValueError: width must be >= 0.0
        >>> msh.interface_elements[0].width = 'abc'
        Traceback (most recent call last):
            ...
        ValueError: could not convert string to float: 'abc'
        """
        return self._width

    @width.setter
    def width(self, val):
        val = float(val)
        if val < 0.0:
            raise ValueError('width must be >= 0.0')
        self._width = val
        self.invalidate_properties()

    @property
    def num_nodes(self):
        """Number of nodes in the :c:`InterfaceElement2D`.

        Returns
        -------
        `int`
            The number of nodes in the :c:`InterfaceElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.interface_elements[0].num_nodes)
        2
        """
        return len(self.nodes)

    @property
    def nodes(self):
        """List of node indices in the :c:`InterfaceElement2D`. References the
        :a:`PolyMesh2D.nodes` of the parent :a:`mesh`.

        Returns
        -------
        `list[int]`
            The list of node indices in the :c:`InterfaceElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.interface_elements[0].nodes)
        [5, 6]
        """
        return self._nodes

    def insert_nodes(self, index, nodes):
        """Insert node indices to the :c:`InterfaceElement2D`.

        Parameters
        ----------
        index : int
            The index at which to insert the **nodes** into :a:`nodes`.
        nodes : list[int]
            The list of node indices to add to :a:`nodes`. Length must be a
            multiple of 2 and the maximum number of :a:`nodes` is 4.

        Note
        -----
        Before inserting the values in **nodes**, an attempt is made to
        cast to a flattened `numpy.ndarray` of `int`. The new number of nodes
        after insertion must be 0, 2, or 4.

        Raises
        ------
        TypeError
            If **index** cannot be interpreted as `int`.
        ValueError
            If **nodes** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **nodes** cannot be cast to `int`, are already
            in :a:`nodes`, are negative, are >= :a:`mesh.num_nodes`, or
            the list of nodes to be added would result in :a:`nodes` having
            a length other than 0, 2, or 4.

        Examples
        --------
        >>> # create a simple mesh
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.nodes.round(14))
        [[-0.    1.  ]
         [ 0.35  1.  ]
         [ 0.    0.  ]
         [ 0.35  0.  ]
         [ 1.    1.  ]
         [ 0.65  1.  ]
         [ 0.65  0.65]
         [ 1.    0.65]
         [ 1.   -0.  ]
         [ 0.65 -0.  ]
         [ 0.65  0.35]
         [ 1.    0.35]
         [ 0.    0.65]
         [ 0.    0.35]
         [ 0.35  0.65]
         [ 0.35  0.35]]

        >>> # create a new element
        >>> # note, this is normally not done explicitly, but is shown here
        >>> # for testing and documentation
        >>> e = vcfempy.meshgen.InterfaceElement2D(msh)
        >>> print(e.nodes)
        []
        >>> e.insert_nodes(0, [1, 14])
        >>> print(e.nodes)
        [1, 14]
        >>> print(np.round(e.length, 14))
        0.35

        >>> # insert no nodes in multiple ways
        >>> e.insert_nodes(0, None)
        >>> e.insert_nodes(0, [])
        >>> print(e.nodes)
        [1, 14]

        >>> # try to insert some invalid nodes
        >>> e.insert_nodes(0, 'one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> e.insert_nodes(0, [1, 14])
        Traceback (most recent call last):
            ...
        ValueError: 14 is already a node
        >>> e.insert_nodes(0, [14, 2])
        Traceback (most recent call last):
            ...
        ValueError: 14 is already a node
        >>> print(e.nodes)
        [1, 14]
        >>> e.insert_nodes(0, [16, 17])
        Traceback (most recent call last):
            ...
        ValueError: node index 17 out of range
        >>> e.insert_nodes(0, [-1, -2])
        Traceback (most recent call last):
            ...
        ValueError: node index -2 out of range
        >>> e.insert_nodes(
        ...             0, [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        >>> e.insert_nodes('one', [2, 3])
        Traceback (most recent call last):
            ...
        TypeError: 'str' object cannot be interpreted as an integer
        >>> e.insert_nodes(0, [2])
        Traceback (most recent call last):
            ...
        ValueError: number of nodes in InterfaceElement2D \
can only be 0, 2, or 4
        >>> e.insert_nodes(0, [2, 3, 4, 5])
        Traceback (most recent call last):
            ...
        ValueError: number of nodes in InterfaceElement2D \
can only be 0, 2, or 4
        """
        if nodes is None:
            return
        nodes = np.array(nodes, dtype=int, ndmin=1)
        if len(nodes) == 0:
            return
        new_num_nodes = self.num_nodes + len(nodes)
        if new_num_nodes % 2 or new_num_nodes > 4:
            raise ValueError('number of nodes in InterfaceElement2D '
                             + 'can only be 0, 2, or 4')
        self.invalidate_properties()
        old_nodes = list(self.nodes)
        nodes = np.flip(nodes.ravel())
        try:
            for n in nodes:
                if n in self.nodes:
                    raise ValueError(f'{n} is already a node')
                if n < 0 or n >= self.mesh.num_nodes:
                    raise ValueError(f'node index {n} out of range')
                self.nodes.insert(index, int(n))
        except ValueError:
            self._nodes = old_nodes
            raise

    def remove_nodes(self, remove_nodes):
        """Remove node indices from the :c:`InterfaceElement2D`.

        Parameters
        ----------
        remove_nodes : list[int]
            The list of nodes to remove from :a:`nodes`.

        Note
        -----
        Before removing the values in **remove_nodes**, an attempt will be
        made to cast it to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        ValueError
            If **remove_nodes** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **remove_nodes** cannot be cast to `int` or
            are not in :a:`nodes`.
            If the number of **remove_nodes** would result in the length of
            :a:`nodes` being other than 0, 2, or 4.

        Examples
        --------
        >>> # create a simple mesh, and remove nodes from an element
        >>> # this should not normally be done explicitly unless you know
        >>> # what you are doing
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.interface_elements[0].nodes)
        [5, 6]
        >>> msh.interface_elements[0].remove_nodes([5, 6])
        >>> print(msh.interface_elements[0].nodes)
        []

        >>> # remove no nodes, in two different ways
        >>> msh.interface_elements[0].insert_nodes(0, [5, 6])
        >>> msh.interface_elements[0].remove_nodes(None)
        >>> msh.interface_elements[0].remove_nodes([])
        >>> print(msh.interface_elements[0].nodes)
        [5, 6]

        >>> # try to remove some invalid nodes
        >>> msh.interface_elements[0].remove_nodes('one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> msh.interface_elements[0].remove_nodes([5, 8])
        Traceback (most recent call last):
            ...
        ValueError: list.remove(x): x not in list
        >>> print(msh.interface_elements[0].nodes)
        [5, 6]
        >>> msh.interface_elements[0].remove_nodes(4)
        Traceback (most recent call last):
            ...
        ValueError: number of nodes in InterfaceElement2D \
can only be 0, 2, or 4
        >>> msh.interface_elements[0].remove_nodes(
        ...                 [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        """
        if remove_nodes is None:
            return
        remove_nodes = np.array(remove_nodes, dtype=int, ndmin=1)
        if len(remove_nodes) == 0:
            return
        new_num_nodes = self.num_nodes - len(remove_nodes)
        if new_num_nodes % 2 or new_num_nodes < 0:
            raise ValueError('number of nodes in InterfaceElement2D '
                             + 'can only be 0, 2, or 4')
        self.invalidate_properties()
        remove_nodes = remove_nodes.ravel()
        old_nodes = list(self.nodes)
        try:
            for rn in remove_nodes:
                self.nodes.remove(rn)
        except ValueError:
            self._nodes = old_nodes
            raise

    @property
    def num_neighbors(self):
        """Number of neighboring :c:`PolyElement2D` elements to the
        :c:`InterfaceElement2D`.

        Returns
        -------
        `int`
            The number of neighbors to the :c:`InterfaceElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.interface_elements[0].num_neighbors)
        2
        """
        return len(self.neighbors)

    @property
    def neighbors(self):
        """The list of neighboring :c:`PolyElement2D` elements to the
        :c:`InterfaceElement2D`.

        Returns
        -------
        `list` of :c:`PolyElement2D`
            The list of neighbors to the :c:`InterfaceElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print([msh.elements.index(n)
        ...        for n in msh.interface_elements[0].neighbors])
        [3, 6]
        """
        return self._neighbors

    def add_neighbors(self, neighbors):
        """Add neighboring :c:`PolyElement2D` to the :c:`InterfaceElement2D`.

        Parameters
        ----------
        neighbors : `list` of :c:`PolyElement2D`
            The list of neighboring :c:`PolyElement2D` to the
            :c:`InterfaceElement2D`.

        Note
        -----
        An :c:`InterfaceElement2D` can only have 0 or 2 neighbors.

        Raises
        ------
        ValueError
            If any values in **neighbors** are not :c:`PolyElement2D`, are
            already in :a:`neighbors`, do not have the same :a:`mesh` as the
            :c:`InterfaceElement2D`, or the list of neighbors to be added
            would result in :a:`neighbors` having a length other than 0 or 2.

        Examples
        --------
        >>> # create a simple mesh
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()

        >>> # create a new element
        >>> # note, this is normally not done explicitly, but is shown here
        >>> # for testing and documentation
        >>> e = vcfempy.meshgen.InterfaceElement2D(msh)
        >>> print(e.neighbors)
        []
        >>> e.add_neighbors(msh.elements[0:2])
        >>> print([msh.elements.index(n) for n in e.neighbors])
        [0, 1]

        >>> # add no neighbors in multiple ways
        >>> e.add_neighbors(None)
        >>> e.add_neighbors([])
        >>> print([msh.elements.index(n) for n in e.neighbors])
        [0, 1]

        >>> # try to add some invalid neighbors
        >>> e.add_neighbors(msh.elements[0:2])
        Traceback (most recent call last):
            ...
        ValueError: number of neighbors to InterfaceElement2D \
can only be 0 or 2
        """
        if neighbors is None:
            return
        if len(neighbors) == 0:
            return
        new_num_neighbors = self.num_neighbors + len(neighbors)
        if new_num_neighbors % 2 or new_num_neighbors > 2:
            raise ValueError('number of neighbors to InterfaceElement2D '
                             + 'can only be 0 or 2')
        old_neighbors = list(self.neighbors)
        try:
            for n in neighbors:
                if n.mesh is not self.mesh:
                    raise ValueError(f'{n} does not have the same parent mesh')
                if n in self.neighbors:
                    raise ValueError(f'element {self.mesh.elements.index(n)}'
                                     + ' is already a neighbor')
                self.neighbors.append(n)
        except ValueError:
            self._neighbors = old_neighbors
            raise

    @property
    def length(self):
        """The length of the :c:`InterfaceElement2D`.

        Returns
        -------
        `float`
            The length of the :c:`InterfaceElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(np.round(msh.interface_elements[0].length, 14))
        0.35
        """
        if self._length is None and self.num_nodes:
            self._length = shp.LineString(
                    self.mesh.nodes[self.nodes[0:2]]).length
        return self._length

    @property
    def area(self):
        """The area of the :c:`InterfaceElement2D`.

        Returns
        -------
        `float`
            The area of the :c:`InterfaceElement2D`.

        Note
        ----
        The method uses the :a:`length` and :a:`width` properties to
        calculate the area. It does not use the polygon formed by the
        :a:`nodes`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(np.round(msh.interface_elements[0].area, 14))
        0.0
        """
        if self._area is None:
            self._area = self.length * self.width
        return self._area

    @property
    def centroid(self):
        """The centroid coordinates of the :c:`InterfaceElement2D`.

        Returns
        -------
        `numpy.ndarray`, shape = (2, )
            The coordinates of the centroid of the :c:`InterfaceElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.interface_elements[0].centroid)
        [0.65  0.825]
        """
        if self._centroid is None:
            if self.num_nodes:
                if self.num_nodes == 2:
                    c = shp.LineString(self.mesh.nodes[self.nodes]).centroid
                elif self.num_nodes == 4:
                    c = shp.Polygon(self.mesh.nodes[self.nodes]).centroid
                self._centroid = np.array([c.x, c.y], dtype=float)
        return self._centroid

    def invalidate_properties(self):
        """Resets cached value of computed attributes :a:`length`, :a:`area`,
        and :a:`centroid`.

        Note
        ----
        The :m:`invalidate_properties` method should be called whenever
        :a:`nodes` or :a:`width` are changed. This is done by
        :m:`insert_nodes`, :m:`remove_nodes`, and the setter for :a:`width`,
        but needs to be done explicitly if making manual changes to
        :a:`nodes`.

        Examples
        --------
        >>> # create a simple mesh, check the element properties
        >>> # invalidate properties and check the values of (private) cache
        >>> # attributes
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.interface_elements[0].width)
        0.0
        >>> print(msh.interface_elements[0]._length)
        None
        >>> print(msh.interface_elements[0]._area)
        None
        >>> print(msh.interface_elements[0]._centroid)
        None
        >>> print(np.round(msh.interface_elements[0].length, 14))
        0.35
        >>> print(msh.interface_elements[0].area)
        0.0
        >>> print(msh.interface_elements[0].centroid.round(14))
        [0.65  0.825]
        >>> msh.interface_elements[0].invalidate_properties()
        >>> print(msh.interface_elements[0]._length)
        None
        >>> print(msh.interface_elements[0]._area)
        None
        >>> print(msh.interface_elements[0]._centroid)
        None
        """
        self._length = None
        self._centroid = None
        self._area = None

    def plot(self, ax=None, **kwargs):
        """Plot the :c:`InterfaceElement2D` using :m:`matplotlib.pyplot.fill`.

        Parameters
        ----------
        ax : None | :c:`matplotlib.axes.Axes`
            The axes to plot on. If not provided, will try to get one using
            :m:`matplotlib.pyplot.gca`.

        Other Parameters
        ----------------
        **kwargs : :c:`matplotlib.patches.Polygon` properties, optional
            Default values:
            `edgecolor` = :a:`material` `color` with alpha = 1.0 (or 'black'
            with alpha = 1.0 if :a:`material` is ``None``),
            `facecolor` = :a:`material` `color` with alpha = 0.6 (or 'black'
            with alpha = 0.6 if :a:`material` is ``None``),
            `linewidth` = 2.0,
            `linestyle` = '--'.

        Returns
        -------
        :c:`matplotlib.axes.Axes`
            The axes that the :c:`InterfaceElement2D` was plotted on.

        Examples
        --------
        >>> # initialize a mesh and a material region, then generate a mesh
        >>> # and plot the elements and interface elements
        >>> import matplotlib.pyplot as plt
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock', color='xkcd:clay')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, msh.boundary_vertices,
        ...                                       rock, 'rock region')
        >>> msh.mesh_scale = 0.2
        >>> msh.mesh_rand = 0.2
        >>> msh.generate_mesh()
        >>> fig = plt.figure()
        >>> for e in msh.elements:
        ...     ax = e.plot(edgecolor=None)
        ...     ax = e.plot_quad_points()
        >>> for e in msh.interface_elements:
        ...     ax = e.plot()
        >>> xmin, xmax, ymin, ymax = ax.axis('equal')
        >>> xtext = ax.set_xlabel('x')
        >>> ytext = ax.set_ylabel('y')
        >>> ttext = ax.set_title('InterfaceElement2D Test Plot')
        >>> plt.savefig('InterfaceElement2D_test_plot.png')
        """
        if ax is None or not isinstance(ax, plt.Axes):
            ax = plt.gca()
        if self.material is not None:
            color = mplclr.to_rgb(self.material.color)
        else:
            color = mplclr.to_rgb('black')
        if 'facecolor' not in kwargs.keys():
            kwargs['facecolor'] = color + (0.6, )
        if 'edgecolor' not in kwargs.keys():
            kwargs['edgecolor'] = color + (1.0, )
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 2.0
        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = '--'
        ax.fill(self.mesh.nodes[self.nodes, 0],
                self.mesh.nodes[self.nodes, 1],
                **kwargs)
        return ax


class BoundaryElement2D():
    """A class for interfaces between :c:`PolyElement2D` elements and the
    boundaries in a :c:`PolyMesh2D`.

    Parameters
    ----------
    mesh : :c:`PolyMesh2D`
        The parent mesh
    nodes : list[int], optional
        Initial list of node indices from the parent mesh that are contained
        in the :c:`BoundaryElement2D`
    neighbor: :c:`PolyElement2D`, optional
        The neighboring :c:`PolyElement2D` from the parent mesh

    Examples
    --------
    """

    def __init__(self, mesh, nodes=None, neighbor=None):
        if not isinstance(mesh, PolyMesh2D):
            raise TypeError('type(mesh) is not vcfempy.meshgen.PolyMesh2D')
        self._mesh = mesh

        self.invalidate_properties()

        self._nodes = []
        self.insert_nodes(0, nodes)

        if neighbor is None:
            self._neighbor = None
        else:
            self.neighbor = neighbor

    @property
    def mesh(self):
        """The parent :c:`PolyMesh2D`.

        Returns
        -------
        :c:`PolyMesh2D`
            The parent mesh assigned to the :c:`BoundaryElement2D`.

        Note
        ----
        The :a:`mesh` is immutable and can only be assigned when the
        :c:`BoundaryElement2D` is created. An :c:`BoundaryElement2D` should
        not usually be created explicitly, but rather should be created
        indirectly by calling the :m:`PolyMesh2D.generate_mesh` method.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.boundary_elements[0].mesh is msh)
        True
        """
        return self._mesh

    @property
    def num_nodes(self):
        """Number of nodes in the :c:`BoundaryElement2D`.

        Returns
        -------
        `int`
            The number of nodes in the :c:`BoundaryElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.boundary_elements[0].num_nodes)
        2
        """
        return len(self.nodes)

    @property
    def nodes(self):
        """List of node indices in the :c:`BoundaryElement2D`. References the
        :a:`PolyMesh2D.nodes` of the parent :a:`mesh`.

        Returns
        -------
        `list[int]`
            The list of node indices in the :c:`BoundaryElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.boundary_elements[0].nodes)
        [0, 1]
        """
        return self._nodes

    def insert_nodes(self, index, nodes):
        """Insert node indices to the :c:`BoundaryElement2D`.

        Parameters
        ----------
        index : int
            The index at which to insert the **nodes** into :a:`nodes`.
        nodes : list[int]
            The list of node indices to add to :a:`nodes`. Length must be a
            multiple of 2 and the maximum number of :a:`nodes` is 2.

        Note
        -----
        Before inserting the values in **nodes**, an attempt is made to
        cast to a flattened `numpy.ndarray` of `int`. The new number of nodes
        after insertion must be 0 or 2.

        Raises
        ------
        TypeError
            If **index** cannot be interpreted as `int`.
        ValueError
            If **nodes** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **nodes** cannot be cast to `int`, are already
            in :a:`nodes`, are negative, are >= :a:`mesh.num_nodes`, or
            the list of nodes to be added would result in :a:`nodes` having
            a length other than 0 or 2.

        Examples
        --------
        >>> # create a simple mesh
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.nodes.round(14))
        [[-0.    1.  ]
         [ 0.35  1.  ]
         [ 0.    0.  ]
         [ 0.35  0.  ]
         [ 1.    1.  ]
         [ 0.65  1.  ]
         [ 0.65  0.65]
         [ 1.    0.65]
         [ 1.   -0.  ]
         [ 0.65 -0.  ]
         [ 0.65  0.35]
         [ 1.    0.35]
         [ 0.    0.65]
         [ 0.    0.35]
         [ 0.35  0.65]
         [ 0.35  0.35]]

        >>> # create a new element
        >>> # note, this is normally not done explicitly, but is shown here
        >>> # for testing and documentation
        >>> e = vcfempy.meshgen.BoundaryElement2D(msh)
        >>> print(e.nodes)
        []
        >>> e.insert_nodes(0, [0, 1])
        >>> print(e.nodes)
        [0, 1]
        >>> print(np.round(e.length, 14))
        0.35

        >>> # insert no nodes in multiple ways
        >>> e.insert_nodes(0, None)
        >>> e.insert_nodes(0, [])
        >>> print(e.nodes)
        [0, 1]

        >>> # try to insert some invalid nodes
        >>> e = vcfempy.meshgen.BoundaryElement2D(msh)
        >>> e.insert_nodes(0, 'one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> e.insert_nodes(0, [0, 0])
        Traceback (most recent call last):
            ...
        ValueError: 0 is already a node
        >>> print(e.nodes)
        []
        >>> e.insert_nodes(0, [16, 17])
        Traceback (most recent call last):
            ...
        ValueError: node index 17 out of range
        >>> print(e.nodes)
        []
        >>> e.insert_nodes(0, [-1, -2])
        Traceback (most recent call last):
            ...
        ValueError: node index -2 out of range
        >>> print(e.nodes)
        []
        >>> e.insert_nodes(
        ...             0, [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        >>> print(e.nodes)
        []
        >>> e.insert_nodes('one', [0, 1])
        Traceback (most recent call last):
            ...
        TypeError: 'str' object cannot be interpreted as an integer
        >>> print(e.nodes)
        []
        >>> e.insert_nodes(0, [2])
        Traceback (most recent call last):
            ...
        ValueError: number of nodes in BoundaryElement2D can only be 0 or 2
        >>> print(e.nodes)
        []
        """
        if nodes is None:
            return
        nodes = np.array(nodes, dtype=int, ndmin=1)
        if len(nodes) == 0:
            return
        new_num_nodes = self.num_nodes + len(nodes)
        if new_num_nodes % 2 or new_num_nodes > 2:
            raise ValueError('number of nodes in BoundaryElement2D '
                             + 'can only be 0 or 2')
        self.invalidate_properties()
        old_nodes = list(self.nodes)
        nodes = np.flip(nodes.ravel())
        try:
            for n in nodes:
                if n in self.nodes:
                    raise ValueError(f'{n} is already a node')
                if n < 0 or n >= self.mesh.num_nodes:
                    raise ValueError(f'node index {n} out of range')
                self.nodes.insert(index, int(n))
        except ValueError:
            self._nodes = old_nodes
            raise

    def remove_nodes(self, remove_nodes):
        """Remove node indices from the :c:`BoundaryElement2D`.

        Parameters
        ----------
        remove_nodes : list[int]
            The list of nodes to remove from :a:`nodes`.

        Note
        -----
        Before removing the values in **remove_nodes**, an attempt will be
        made to cast it to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        ValueError
            If **remove_nodes** is not `array_like`, such as a jagged
            `list[list[int]]`.
            If any values in **remove_nodes** cannot be cast to `int` or
            are not in :a:`nodes`.
            If the number of **remove_nodes** would result in the length of
            :a:`nodes` being other than 0 or 2.

        Examples
        --------
        >>> # create a simple mesh, and remove nodes from an element
        >>> # this should not normally be done explicitly unless you know
        >>> # what you are doing
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.boundary_elements[0].nodes)
        [0, 1]
        >>> msh.boundary_elements[0].remove_nodes([0, 1])
        >>> print(msh.boundary_elements[0].nodes)
        []

        >>> # remove no nodes, in two different ways
        >>> msh.boundary_elements[0].insert_nodes(0, [0, 1])
        >>> msh.boundary_elements[0].remove_nodes(None)
        >>> msh.boundary_elements[0].remove_nodes([])
        >>> print(msh.boundary_elements[0].nodes)
        [0, 1]

        >>> # try to remove some invalid nodes
        >>> msh.boundary_elements[0].remove_nodes('one')
        Traceback (most recent call last):
            ...
        ValueError: invalid literal for int() with base 10: 'one'
        >>> msh.boundary_elements[0].remove_nodes([5, 0])
        Traceback (most recent call last):
            ...
        ValueError: list.remove(x): x not in list
        >>> print(msh.boundary_elements[0].nodes)
        [0, 1]
        >>> msh.boundary_elements[0].remove_nodes(4)
        Traceback (most recent call last):
            ...
        ValueError: number of nodes in BoundaryElement2D can only be 0 or 2
        >>> msh.boundary_elements[0].remove_nodes(
        ...                 [[1, 2], 3]) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: ...
        """
        if remove_nodes is None:
            return
        remove_nodes = np.array(remove_nodes, dtype=int, ndmin=1)
        if len(remove_nodes) == 0:
            return
        new_num_nodes = self.num_nodes - len(remove_nodes)
        if new_num_nodes % 2 or new_num_nodes < 0:
            raise ValueError('number of nodes in BoundaryElement2D '
                             + 'can only be 0 or 2')
        self.invalidate_properties()
        remove_nodes = remove_nodes.ravel()
        old_nodes = list(self.nodes)
        try:
            for rn in remove_nodes:
                self.nodes.remove(rn)
        except ValueError:
            self._nodes = old_nodes
            raise

    @property
    def neighbor(self):
        """The neighboring :c:`PolyElement2D` to the :c:`BoundaryElement2D`.

        Parameters
        ----------
        neighbor : :c:`PolyElement2D`
            The new neighboring :c:`PolyElement2D` to the
            :c:`BoundaryElement2D`.

        Returns
        -------
        ``None`` | :c:`PolyElement2D`
            The neighboring :c:`PolyElement2D` to the :c:`BoundaryElement2D`.
            If no neighbor has been assigned, returns ``None``.

        Note
        -----
        A :c:`BoundaryElement2D` can only have 0 or 1 neighbors.

        Raises
        ------
        TypeError
            If **neighbor** is not a :c:`PolyElement2D`.
        ValueError
            If **neighbor** does not have the same :a:`mesh` as the
            :c:`BoundaryElement2D`.

        Examples
        --------
        >>> # create a simple mesh
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()

        >>> # create a new element
        >>> # note, this is normally not done explicitly, but is shown here
        >>> # for testing and documentation
        >>> e = vcfempy.meshgen.BoundaryElement2D(msh)
        >>> print(e.neighbor)
        None
        >>> e.neighbor = msh.elements[0]
        >>> print(msh.elements.index(e.neighbor))
        0

        >>> # try to add some invalid neighbors
        >>> e.neighbor = 0
        Traceback (most recent call last):
            ...
        TypeError: neighbor must be a PolyElement2D
        >>> msh_new = vcfempy.meshgen.PolyMesh2D()
        >>> pe = vcfempy.meshgen.PolyElement2D(msh_new)
        >>> e.neighbor = pe
        Traceback (most recent call last):
            ...
        ValueError: neighbor must have same parent mesh
        """
        return self._neighbor

    @neighbor.setter
    def neighbor(self, neighbor):
        if not isinstance(neighbor, PolyElement2D):
            raise TypeError('neighbor must be a PolyElement2D')
        if self.mesh is not neighbor.mesh:
            raise ValueError('neighbor must have same parent mesh')
        self._neighbor = neighbor

    @property
    def length(self):
        """The length of the :c:`BoundaryElement2D`.

        Returns
        -------
        `float`
            The length of the :c:`BoundaryElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(np.round(msh.boundary_elements[0].length, 14))
        0.35
        """
        if self._length is None and self.num_nodes:
            self._length = shp.LineString(self.mesh.nodes[self.nodes]).length
        return self._length

    @property
    def centroid(self):
        """The centroid coordinates of the :c:`BoundaryElement2D`.

        Returns
        -------
        `numpy.ndarray`, shape = (2, )
            The coordinates of the centroid of the :c:`BoundaryElement2D`.

        Examples
        --------
        >>> # create a simple mesh and check the element properties
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.boundary_elements[0].centroid)
        [0.175 1.   ]
        """
        if self._centroid is None and self.num_nodes:
            c = shp.LineString(self.mesh.nodes[self.nodes]).centroid
            self._centroid = np.array([c.x, c.y], dtype=float)
        return self._centroid

    def invalidate_properties(self):
        """Resets cached value of computed attributes :a:`length` and
        :a:`centroid`.

        Note
        ----
        The :m:`invalidate_properties` method should be called whenever
        :a:`nodes` is changed. This is done by :m:`insert_nodes` and
        :m:`remove_nodes`, but needs to be done explicitly if making manual
        changes to :a:`nodes`.

        Examples
        --------
        >>> # create a simple mesh, check the element properties
        >>> # invalidate properties and check the values of (private) cache
        >>> # attributes
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> msh.mesh_scale = 0.4
        >>> msh.add_seed_points([0.5, 0.5])
        >>> msh.generate_mesh()
        >>> print(msh.boundary_elements[0]._length)
        None
        >>> print(msh.boundary_elements[0]._centroid)
        None
        >>> print(np.round(msh.boundary_elements[0].length, 14))
        0.35
        >>> print(msh.boundary_elements[0].centroid.round(14))
        [0.175 1.   ]
        >>> msh.boundary_elements[0].invalidate_properties()
        >>> print(msh.interface_elements[0]._length)
        None
        >>> print(msh.interface_elements[0]._centroid)
        None
        """
        self._length = None
        self._centroid = None

    def plot(self, ax=None, **kwargs):
        """Plot the :c:`BoundaryElement2D` using :m:`matplotlib.pyplot.plot`.

        Parameters
        ----------
        ax : None | :c:`matplotlib.axes.Axes`
            The axes to plot on. If not provided, will try to get one using
            :m:`matplotlib.pyplot.gca`.

        Other Parameters
        ----------------
        **kwargs : :c:`matplotlib.lines.Line2D` properties, optional
            Default values:
            `linewidth` = 2.0,
            `linestyle` = '--'.
            `color` = 'black'

        Returns
        -------
        :c:`matplotlib.axes.Axes`
            The axes that the :c:`BoundaryElement2D` was plotted on.

        Examples
        --------
        >>> # initialize a mesh and a material region, then generate a mesh
        >>> # and plot the elements, interface elements, and boundary
        >>> # elements
        >>> import matplotlib.pyplot as plt
        >>> import vcfempy.materials
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D('test mesh')
        >>> msh.add_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> msh.insert_boundary_vertices(0, [0, 1, 2, 3])
        >>> rock = vcfempy.materials.Material('rock', color='xkcd:clay')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, msh.boundary_vertices,
        ...                                       rock, 'rock region')
        >>> msh.mesh_scale = 0.2
        >>> msh.mesh_rand = 0.2
        >>> msh.generate_mesh()
        >>> fig = plt.figure()
        >>> for e in msh.elements:
        ...     ax = e.plot(edgecolor=None)
        ...     ax = e.plot_quad_points()
        >>> for e in msh.interface_elements:
        ...     ax = e.plot()
        >>> for e in msh.boundary_elements:
        ...     ax = e.plot()
        >>> xmin, xmax, ymin, ymax = ax.axis('equal')
        >>> xtext = ax.set_xlabel('x')
        >>> ytext = ax.set_ylabel('y')
        >>> ttext = ax.set_title('BoundaryElement2D Test Plot')
        >>> plt.savefig('BoundaryElement2D_test_plot.png')
        """
        if ax is None or not isinstance(ax, plt.Axes):
            ax = plt.gca()
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 2.0
        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = '--'
        if 'color' not in kwargs.keys():
            kwargs['color'] = 'black'
        ax.plot(self.mesh.nodes[self.nodes, 0],
                self.mesh.nodes[self.nodes, 1],
                **kwargs)
        return ax


def _get_unit_tangent_normal(v0, v1):
    tt = v1 - v0
    tt_len = np.linalg.norm(tt)
    tt /= tt_len
    nn = np.array([tt[1], -tt[0]])
    return tt, nn


def _reflect_point_across_edge(p, v, tt):
    vp = p - v
    pp = v + np.dot(vp, tt) * tt
    return (2 * pp - p)


def _get_edge_reflection_points(rp0, rp1, v, tt, d_scale, alpha_rand):
    rr = rp1 - rp0
    rr_len = np.linalg.norm(rr)
    rr /= rr_len
    num_points = int(np.round(rr_len / d_scale)) + 1
    if num_points <= 2:
        return np.empty((0, 2))
    dr = np.linspace(0., rr_len, num_points)[1:-1]
    dr += alpha_rand * d_scale * (2. * np.random.random(num_points - 2) - 1.)
    ref_points = []
    for ddrr in dr:
        rp = rp0 + ddrr * rr
        ref_points.append(rp)
        ref_points.append(_reflect_point_across_edge(rp, v, tt))
    return np.array(ref_points)
