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
import matplotlib.path as path
from scipy.spatial import Voronoi as Voronoi

import vcfempy.materials as mtl


class PolyMesh2D():
    """A class for 2D polygonal mesh generation.

    Parameters
    ----------
    vertices : array_like, optional
        Initial vertices to be added to :a:`vertices`. Passed to
        :m:`add_vertices`.
    boundary_vertices : int | list[int], optional
        Initial boundary vertex or list of boundary vertices to be added.
        Passed to :m:`insert_boundary_vertices`.
    material_regions : `list[list[int]]` | `list` of :c:`MaterialRegion2D`, \
*optional*
        Initial list(s) of material region(s) to be added. Passed to
        :m:`add_material_regions`.
    materials : :c:`vcfempy.materials.Material` | `list` of \
:c:`vcfempy.materials.Material`, *optional*
        Initial list of material types for the :a:`material_regions`. Passed
        to :m:`add_material_regions`.
    mesh_edges : list[int] | list[list[int]], optional
        Initial list(s) defining non-boundary edges to be preserved in the
        mesh generation. Passed to :m:`add_mesh_edges`.

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
    >>> import vcfempy.meshgen
    >>> msh = vcfempy.meshgen.PolyMesh2D()
    >>> print(msh.num_vertices)
    0

    >>> # add some vertices to the mesh
    >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
    >>> msh.add_vertices(new_verts)
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

    >>> # create a material and material region and add them to the mesh
    >>> import vcfempy.materials
    >>> m = vcfempy.materials.Material('rock')
    >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, bnd_verts, m)
    >>> msh.add_material_regions(mr)
    >>> print(msh.num_material_regions)
    1
    >>> print(msh.material_regions[0].vertices)
    [0, 1, 2, 3]
    >>> print(msh.material_regions[0].material.name)
    rock

    >>> # generate a simple mesh and check some mesh statistics
    >>> msh.generate_mesh((4, 4))
    >>> print(msh.num_nodes)
    34
    >>> print(msh.num_elements)
    16
    >>> import matplotlib.pyplot as plt
    >>> n, bins, _ = plt.hist(msh.num_nodes_per_element,
    ...                       bins=[k for k in range(3, 11)])
    >>> print(bins)
    [ 3  4  5  6  7  8  9 10]
    >>> print(n)
    [0. 4. 6. 6. 0. 0. 0.]

    >>> # plot the mesh and save an image
    >>> ax = plt.gca()
    >>> ax.clear()
    >>> ax = msh.plot_mesh(ax)
    >>> ax = msh.plot_boundaries(ax)
    >>> ax = msh.plot_mesh_boundaries(ax)
    >>> ax = msh.plot_vertices(ax)
    >>> ax = msh.plot_mesh_nodes(ax)
    >>> xmin, xmax, ymin, ymax = plt.axis('equal')
    >>> xlab_text = ax.set_xlabel('x', fontweight='bold')
    >>> ylab_text = ax.set_ylabel('y', fontweight='bold')
    >>> title_text = ax.set_title('Simple square mesh', fontweight='bold')
    >>> plt.savefig('PolyMesh2D_simple_mesh_example.png')
    """

    def __init__(self, vertices=None, boundary_vertices=None,
                 material_regions=None, materials=None, mesh_edges=None,
                 verbose_printing=False, high_order_quadrature=False):
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

        # initialize material regions
        self._material_regions = []
        self.add_material_regions(material_regions, materials)

        # initialize mesh edges
        self._mesh_edges = []
        self.add_mesh_edges(mesh_edges)

        # initialize flags for
        #      verbose printing
        #      high order quadrature in all elements
        self.verbose_printing = verbose_printing
        self.high_order_quadrature = high_order_quadrature

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
        Instead, modify it using the :m:`add_vertices` method.

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
        mutable. It is updated each time :a:`boundary_vertices` is changed,
        so the number of :a:`boundary_edges` should always be
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
        >>> print(msh.num_material_regions)
        0

        >>> # create some vertices and add them to the mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1.5, 0.5], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)

        >>> # create a new material and material region
        >>> # and add them to the mesh
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('rock')
        >>> mr = vcfempy.meshgen.MaterialRegion2D(msh, bnd_verts, m)
        >>> msh.add_material_regions(mr)
        >>> print(msh.num_material_regions)
        1
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
        mutable. Instead modify it using the :m:`add_material_regions`
        method.

        Examples
        --------
        >>> # initiaize a mesh, no initial properties provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.material_regions)
        []

        >>> # add some vertices to the mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> bnd_verts = [k for k, _ in enumerate(msh.vertices)]
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh.vertices)
        [[0. 0.]
         [0. 1.]
         [1. 1.]
         [1. 0.]]

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
        >>> msh.add_material_regions(mr_rock)
        >>> print(msh.num_material_regions)
        1
        >>> for mr in msh.material_regions:
        ...     print(mr.vertices)
        [0, 4, 5, 3]

        >>> # add two more material regions
        >>> # these are added by directly passing vertex indices
        >>> # to the add_material_regions method
        >>> msh.add_vertices([[0.5, 0.5], [0.5, 1]])
        >>> print(msh.vertices)
        [[0.  0. ]
         [0.  1. ]
         [1.  1. ]
         [1.  0. ]
         [0.  0.5]
         [1.  0.5]
         [0.5 0.5]
         [0.5 1. ]]
        >>> sand = vcfempy.materials.Material('sand')
        >>> clay = vcfempy.materials.Material('clay')
        >>> mr_sand_verts = [4, 1, 7, 6]
        >>> mr_clay_verts = [6, 7, 2, 5]
        >>> msh.add_material_regions([mr_sand_verts, mr_clay_verts],
        ...                          [sand, clay])
        >>> print(msh.num_material_regions)
        3
        >>> for mr in msh.material_regions:
        ...     print(mr.vertices)
        [0, 4, 5, 3]
        [4, 1, 7, 6]
        [6, 7, 2, 5]
        """
        return self._material_regions

    @property
    def num_mesh_edges(self):
        """Number of non-boundary edges to be preserved in mesh generation
        for the :c:`PolyMesh2D`.

        Returns
        -------
        `int`
            The number of :a:`mesh_edges` in the :c:`PolyMesh2D`.

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
        >>> msh.add_mesh_edges([5, 6])
        >>> print(msh.num_mesh_edges)
        1
        """
        return len(self.mesh_edges)

    @property
    def mesh_edges(self):
        """List of lists of vertex indices defining non-boundary edges to be
        preserved in mesh generation for the :c:`PolyMesh2D`.

        Returns
        -------
        `list[list[int]]`, shape = (:a:`num_mesh_edges`, 2)
            The list of index pairs defining non-boundary edges in the
            :c:`PolyMesh2D`.

        Note
        ----
        The list of :a:`mesh_edges` is not intended to be directly mutable.
        Instead, modify the list of :a:`mesh_edges` by using the
        :m:`add_mesh_edges` method.

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
        >>> msh.add_mesh_edges([4, 5])
        >>> print(msh.num_mesh_edges)
        1
        >>> print(msh.mesh_edges)
        [[4, 5]]
        >>> print(msh.vertices[msh.mesh_edges[0], :])
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
        >>> msh.add_mesh_edges([[6, 7], [8, 9]])
        >>> print(msh.num_mesh_edges)
        3
        >>> print(msh.mesh_edges)
        [[4, 5], [6, 7], [8, 9]]
        >>> for k, me in enumerate(msh.mesh_edges):
        ...     print(f'Mesh edge {k}')
        ...     print(msh.vertices[me, :])
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

        >>> # generate a very simple mesh
        >>> # note: num_elements == len(msh.points)
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_elements)
        4
        >>> print(msh.points)
        [[0.375 0.25 ]
         [0.875 0.25 ]
         [0.125 0.75 ]
         [0.625 0.75 ]]

        >>> # explicitly resetting the mesh clears the seed points
        >>> msh.mesh_valid = False
        >>> print(msh.num_elements)
        0
        >>> print(msh.points)
        []

        >>> # regenerate the mesh
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_elements)
        4
        >>> print(msh.points)
        [[0.375 0.25 ]
         [0.875 0.25 ]
         [0.125 0.75 ]
         [0.625 0.75 ]]

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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_nodes)
        10

        >>> # explicitly resetting the mesh clears the nodes
        >>> msh.mesh_valid = False
        >>> print(msh.num_nodes)
        0

        >>> # regenerate the mesh
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_nodes)
        10

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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.nodes)
        [[0.     0.    ]
         [0.     1.    ]
         [0.375  0.5625]
         [0.     0.375 ]
         [1.     1.    ]
         [0.375  1.    ]
         [0.625  0.4375]
         [1.     0.625 ]
         [1.     0.    ]
         [0.625  0.    ]]

        >>> # explicitly resetting the mesh clears the nodes
        >>> msh.mesh_valid = False
        >>> print(msh.nodes)
        []

        >>> # regenerate the mesh
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.nodes)
        [[0.     0.    ]
         [0.     1.    ]
         [0.375  0.5625]
         [0.     0.375 ]
         [1.     1.    ]
         [0.375  1.    ]
         [0.625  0.4375]
         [1.     0.625 ]
         [1.     0.    ]
         [0.625  0.    ]]

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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_elements)
        4

        >>> # explicitly resetting the mesh clears the elements
        >>> msh.mesh_valid = False
        >>> print(msh.num_elements)
        0

        >>> # regenerate the mesh
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_elements)
        4

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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.nodes)
        [[0.     0.    ]
         [0.     1.    ]
         [0.375  0.5625]
         [0.     0.375 ]
         [1.     1.    ]
         [0.375  1.    ]
         [0.625  0.4375]
         [1.     0.625 ]
         [1.     0.    ]
         [0.625  0.    ]]
        >>> for e in msh.elements:
        ...     print(e.nodes)
        [9, 0, 3, 2, 6]
        [8, 7, 6, 9]
        [2, 5, 1, 3]
        [6, 2, 5, 4, 7]

        >>> # explicitly resetting the mesh clears the elements
        >>> msh.mesh_valid = False
        >>> print(msh.elements)
        []

        >>> # regenerate the mesh
        >>> msh.generate_mesh((2, 2))
        >>> for e in msh.elements:
        ...     print(e.nodes)
        [9, 0, 3, 2, 6]
        [8, 7, 6, 9]
        [2, 5, 1, 3]
        [6, 2, 5, 4, 7]

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
        list[int]
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
        >>> msh.generate_mesh((2, 2))
        >>> for e in msh.elements:
        ...     print(e.nodes)
        [9, 0, 3, 2, 6]
        [8, 7, 6, 9]
        [2, 5, 1, 3]
        [6, 2, 5, 4, 7]
        >>> print(msh.num_nodes_per_element)
        [5, 4, 4, 5]

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
        >>> # initialize a mesh, no initial information provided
        >>> import vcfempy.meshgen
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh.element_materials)
        []

        >>> # add some vertices and boundary vertices to the mesh
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, [k for k, _
        ...                                  in enumerate(msh.vertices)])
        >>> # add some more vertices and create two materials and material
        >>> # regions and add them to the mesh
        >>> # the rock material region covers the bottom half of the mesh
        >>> # and the sand material region covers the top half
        >>> import vcfempy.materials
        >>> msh.add_vertices([[0, 0.5], [1, 0.5]])
        >>> rock = vcfempy.materials.Material('rock')
        >>> sand = vcfempy.materials.Material('sand')
        >>> msh.add_material_regions([[0, 4, 5, 3], [4, 1, 2, 5]],
        ...                          [rock, sand])
        >>> # still no materials assigned to elements
        >>> # because the mesh has not been generated
        >>> print(msh.element_materials)
        []

        >>> # generate a simple mesh
        >>> # now materials will be assigned to elements
        >>> msh.generate_mesh((2, 2))
        >>> for m in msh.element_materials:
        ...     print(m.name)
        rock
        rock
        sand
        sand

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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.element_areas)
        [-0.30078125  0.19921875  0.19921875 -0.30078125]

        >>> # notice that element areas sum to the area of the boundary
        >>> import numpy as np
        >>> print(np.sum(np.abs(msh.element_areas)))
        1.0
        >>> print(np.abs(vcfempy.meshgen.polygon_area(msh.vertices)))
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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.element_centroids)
        [[0.32251082 0.24323593]
         [0.82352941 0.26838235]
         [0.17647059 0.73161765]
         [0.67748918 0.75676407]]

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
        >>> msh.generate_mesh((2, 2))
        >>> for k, qp in enumerate(msh.element_quad_points):
        ...     print(f'Element {k} quad points, nq{k} = {len(qp)}')
        ...     print(qp)
        ...     print()
        Element 0 quad points, nq0 = 11
        [[ 0.22686688 -0.18242695]
         [-0.24188312 -0.18242695]
         [-0.24188312  0.09882305]
         [ 0.03936688  0.23944805]
         [ 0.22686688  0.14569805]
         [-0.00500541 -0.12161797]
         [-0.16125541 -0.02786797]
         [-0.06750541  0.11275703]
         [ 0.08874459  0.12838203]
         [ 0.15124459 -0.01224297]
         [ 0.          0.        ]]
        <BLANKLINE>
        Element 1 quad points, nq1 = 9
        [[ 0.13235294 -0.20128676]
         [ 0.13235294  0.26746324]
         [-0.14889706  0.12683824]
         [-0.14889706 -0.20128676]
         [ 0.08823529  0.02205882]
         [-0.00551471  0.13143382]
         [-0.09926471 -0.02481618]
         [-0.00551471 -0.13419118]
         [ 0.          0.        ]]
        <BLANKLINE>
        Element 2 quad points, nq2 = 9
        [[ 0.14889706 -0.12683824]
         [ 0.14889706  0.20128676]
         [-0.13235294  0.20128676]
         [-0.13235294 -0.26746324]
         [ 0.09926471  0.02481618]
         [ 0.00551471  0.13419118]
         [-0.08823529 -0.02205882]
         [ 0.00551471 -0.13143382]
         [ 0.          0.        ]]
        <BLANKLINE>
        Element 3 quad points, nq3 = 11
        [[-0.03936688 -0.23944805]
         [-0.22686688 -0.14569805]
         [-0.22686688  0.18242695]
         [ 0.24188312  0.18242695]
         [ 0.24188312 -0.09882305]
         [-0.08874459 -0.12838203]
         [-0.15124459  0.01224297]
         [ 0.00500541  0.12161797]
         [ 0.16125541  0.02786797]
         [ 0.06750541 -0.11275703]
         [ 0.          0.        ]]
        <BLANKLINE>

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
        >>> msh.generate_mesh((2, 2))
        >>> for k, qw in enumerate(msh.element_quad_weights):
        ...     print(f'Element {k} quad weights, nq{k} = {len(qw)}')
        ...     print(qw)
        ...     print()
        Element 0 quad weights, nq0 = 11
        [0.13700319 0.12979995 0.0987549  0.09296443 0.0915012  0.08627986
         0.08054059 0.07032208 0.07068559 0.07720864 0.06493957]
        <BLANKLINE>
        Element 1 quad weights, nq1 = 9
        [0.13313931 0.13313931 0.11431492 0.11431492 0.11160285 0.09836819
         0.10036614 0.09836819 0.09638618]
        <BLANKLINE>
        Element 2 quad weights, nq2 = 9
        [0.11431492 0.11431492 0.13313931 0.13313931 0.10036614 0.09836819
         0.11160285 0.09836819 0.09638618]
        <BLANKLINE>
        Element 3 quad weights, nq3 = 11
        [0.09296443 0.0915012  0.13700319 0.12979995 0.0987549  0.07068559
         0.07720864 0.08627986 0.08054059 0.07032208 0.06493957]
        <BLANKLINE>

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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_interface_elements)
        5

        >>> # explicitly resetting the mesh clears the interface elements
        >>> msh.mesh_valid = False
        >>> print(msh.num_interface_elements)
        0

        >>> # regenerate the mesh
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_interface_elements)
        5

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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_nodes)
        10
        >>> print(msh.nodes)
        [[0.     0.    ]
         [0.     1.    ]
         [0.375  0.5625]
         [0.     0.375 ]
         [1.     1.    ]
         [0.375  1.    ]
         [0.625  0.4375]
         [1.     0.625 ]
         [1.     0.    ]
         [0.625  0.    ]]
        >>> for e in msh.interface_elements:
        ...     print(e.nodes)
        [2, 5]
        [2, 3]
        [2, 6]
        [6, 7]
        [6, 9]
        >>> print(msh.num_elements)
        4
        >>> for e in msh.interface_elements:
        ...     print([msh.elements.index(n) for n in e.neighbors])
        [2, 3]
        [2, 0]
        [3, 0]
        [3, 1]
        [1, 0]

        >>> # explicitly resetting the mesh clears the interface elements
        >>> msh.mesh_valid = False
        >>> print(msh.interface_elements)
        []

        >>> # regenerate the mesh
        >>> msh.generate_mesh((2, 2))
        >>> for e in msh.interface_elements:
        ...     print(e.nodes)
        [2, 5]
        [2, 3]
        [2, 6]
        [6, 7]
        [6, 9]
        >>> for e in msh.interface_elements:
        ...     print([msh.elements.index(n) for n in e.neighbors])
        [2, 3]
        [2, 0]
        [3, 0]
        [3, 1]
        [1, 0]

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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_boundary_elements)
        8

        >>> # explicitly resetting the mesh clears the interface elements
        >>> msh.mesh_valid = False
        >>> print(msh.num_boundary_elements)
        0

        >>> # regenerate the mesh
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_boundary_elements)
        8

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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.num_nodes)
        10
        >>> print(msh.nodes)
        [[0.     0.    ]
         [0.     1.    ]
         [0.375  0.5625]
         [0.     0.375 ]
         [1.     1.    ]
         [0.375  1.    ]
         [0.625  0.4375]
         [1.     0.625 ]
         [1.     0.    ]
         [0.625  0.    ]]
        >>> for e in msh.boundary_elements:
        ...     print(e.nodes)
        [1, 3]
        [0, 3]
        [4, 5]
        [1, 5]
        [4, 7]
        [8, 9]
        [0, 9]
        [7, 8]
        >>> print(msh.num_elements)
        4
        >>> print([msh.elements.index(e.neighbor)
        ...        for e in msh.boundary_elements])
        [2, 0, 3, 2, 3, 1, 0, 1]

        >>> # explicitly resetting the mesh clears the boundary elements
        >>> msh.mesh_valid = False
        >>> print(msh.boundary_elements)
        []

        >>> # regenerate the mesh
        >>> msh.generate_mesh((2, 2))
        >>> for e in msh.boundary_elements:
        ...     print(e.nodes)
        [1, 3]
        [0, 3]
        [4, 5]
        [1, 5]
        [4, 7]
        [8, 9]
        [0, 9]
        [7, 8]
        >>> print([msh.elements.index(e.neighbor)
        ...        for e in msh.boundary_elements])
        [2, 0, 3, 2, 3, 1, 0, 1]

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
        >>> msh.generate_mesh((2, 2))
        >>> print(msh.mesh_valid)
        True
        >>> print(msh.num_elements)
        4

        >>> # explicitly resetting the mesh with a bool value
        >>> msh.mesh_valid = False
        >>> print(msh.mesh_valid)
        False
        >>> print(msh.num_elements)
        0

        >>> # regenerate the mesh
        >>> msh.generate_mesh((2, 2))
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
        >>> msh.generate_mesh((2, 2))
        >>> for k, qp in enumerate(msh.element_quad_points):
        ...     print(f'Element {k} quad points, nq{k} = {len(qp)}')
        ...     print(qp)
        ...     print() # doctest: +ELLIPSIS
        Element 0 quad points, nq0 = 11
        [[ 0.22686688 -0.18242695]
         [-0.24188312 -0.18242695]
        ...
         [ 0.15124459 -0.01224297]
         [ 0.          0.        ]]
        <BLANKLINE>
        Element 1 quad points, nq1 = 9
        [[ 0.13235294 -0.20128676]
         [ 0.13235294  0.26746324]
        ...
         [-0.00551471 -0.13419118]
         [ 0.          0.        ]]
        <BLANKLINE>
        Element 2 quad points, nq2 = 9
        [[ 0.14889706 -0.12683824]
         [ 0.14889706  0.20128676]
        ...
         [ 0.00551471 -0.13143382]
         [ 0.          0.        ]]
        <BLANKLINE>
        Element 3 quad points, nq3 = 11
        [[-0.03936688 -0.23944805]
         [-0.22686688 -0.14569805]
        ...
         [ 0.06750541 -0.11275703]
         [ 0.          0.        ]]
        <BLANKLINE>

        >>> # switch to high order quadrature
        >>> # no need to regenerate mesh, element quadrature will be reset
        >>> msh.high_order_quadrature = True
        >>> print(msh.high_order_quadrature)
        True
        >>> for k, qp in enumerate(msh.element_quad_points):
        ...     print(f'Element {k} quad points, nq{k} = {len(qp)}')
        ...     print(qp)
        ...     print() # doctest: +ELLIPSIS
        Element 0 quad points, nq0 = 21
        [[ 0.2571158  -0.20675054]
         [-0.2741342  -0.20675054]
        ...
         [ 0.10284632 -0.00832522]
         [ 0.          0.        ]]
        <BLANKLINE>
        Element 1 quad points, nq1 = 17
        [[ 0.15     -0.228125]
         [ 0.15      0.303125]
        ...
         [-0.00375  -0.09125 ]
         [ 0.        0.      ]]
        <BLANKLINE>
        Element 2 quad points, nq2 = 17
        [[ 0.16875  -0.14375 ]
         [ 0.16875   0.228125]
        ...
         [ 0.00375  -0.089375]
         [ 0.        0.      ]]
        <BLANKLINE>
        Element 3 quad points, nq3 = 21
        [[-0.0446158  -0.27137446]
         [-0.2571158  -0.16512446]
        ...
         [ 0.04590368 -0.07667478]
         [ 0.          0.        ]]
        <BLANKLINE>

        >>> # switch back to low order quadrature
        >>> # use a False-like string
        >>> msh.high_order_quadrature = 'off'
        >>> print(msh.high_order_quadrature)
        False
        >>> for k, qp in enumerate(msh.element_quad_points):
        ...     print(f'Element {k} quad points, nq{k} = {len(qp)}')
        ...     print(qp)
        ...     print() # doctest: +ELLIPSIS
        Element 0 quad points, nq0 = 11
        [[ 0.22686688 -0.18242695]
         [-0.24188312 -0.18242695]
        ...
         [ 0.15124459 -0.01224297]
         [ 0.          0.        ]]
        <BLANKLINE>
        Element 1 quad points, nq1 = 9
        [[ 0.13235294 -0.20128676]
         [ 0.13235294  0.26746324]
        ...
         [-0.00551471 -0.13419118]
         [ 0.          0.        ]]
        <BLANKLINE>
        Element 2 quad points, nq2 = 9
        [[ 0.14889706 -0.12683824]
         [ 0.14889706  0.20128676]
        ...
         [ 0.00551471 -0.13143382]
         [ 0.          0.        ]]
        <BLANKLINE>
        Element 3 quad points, nq3 = 11
        [[-0.03936688 -0.23944805]
         [-0.22686688 -0.14569805]
        ...
         [ 0.06750541 -0.11275703]
         [ 0.          0.        ]]
        <BLANKLINE>

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
        >>> msh = vcfempy.meshgen.PolyMesh2D()
        >>> print(msh)
        vcfempy.meshgen.PolyMesh2D
        Number of Vertices = 0
        Number of Boundary Vertices = 0
        Number of Material Regions = 0
        Number of Mesh Edges = 0
        Verbose Printing = False
        High Order Quadrature = False
        Mesh Generated = False
        <BLANKLINE>
        <BLANKLINE>

        >>> # add some vertices and boundary vertices to the mesh
        >>> # no mesh generated yet
        >>> new_verts = [[0, 0], [0, 1], [1, 1], [1, 0]]
        >>> bnd_verts = [k for k, _ in enumerate(new_verts)]
        >>> msh.add_vertices(new_verts)
        >>> msh.insert_boundary_vertices(0, bnd_verts)
        >>> print(msh)
        vcfempy.meshgen.PolyMesh2D
        Number of Vertices = 4
        Number of Boundary Vertices = 4
        Number of Material Regions = 0
        Number of Mesh Edges = 0
        Verbose Printing = False
        High Order Quadrature = False
        Mesh Generated = False
        <BLANKLINE>
        <BLANKLINE>

        >>> # set verbose printing flag
        >>> msh.verbose_printing = True
        >>> print(msh)
        vcfempy.meshgen.PolyMesh2D
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
        <BLANKLINE>
        <BLANKLINE>

        >>> # turn off verbose printing and add some vertices
        >>> # and two material regions
        >>> msh.verbose_printing = 'off'
        >>> msh.add_vertices([[0, 0.5], [1, 0.5]])
        >>> rock = vcfempy.materials.Material('rock')
        >>> sand = vcfempy.materials.Material('sand')
        >>> msh.add_material_regions([[0, 4, 5, 3], [4, 1, 2, 5]],
        ...                          [rock, sand])
        >>> print(msh)
        vcfempy.meshgen.PolyMesh2D
        Number of Vertices = 6
        Number of Boundary Vertices = 4
        Number of Material Regions = 2
        Number of Mesh Edges = 0
        Verbose Printing = False
        High Order Quadrature = False
        Mesh Generated = False
        <BLANKLINE>
        <BLANKLINE>

        >>> # turn verbose printing back on and generate the mesh
        >>> msh.verbose_printing = 'yes'
        >>> msh.generate_mesh((2, 2))
        >>> print(msh)
        vcfempy.meshgen.PolyMesh2D
        Number of Vertices = 6
        Number of Boundary Vertices = 4
        Number of Material Regions = 2
        Number of Mesh Edges = 0
        Verbose Printing = True
        High Order Quadrature = False
        Mesh Generated = True
        Number of Nodes = 10
        Number of Elements = 4
        Number of Interface Elements = 5
        Number of Boundary Elements = 8
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
        Material Region 0, rock
        [0, 4, 5, 3]
        <BLANKLINE>
        Material Region 1, sand
        [4, 1, 2, 5]
        <BLANKLINE>
        Nodes
        [[0.     0.    ]
         [0.     1.    ]
         [0.375  0.5625]
         [0.     0.375 ]
         [1.     1.    ]
         [0.375  1.    ]
         [0.625  0.4375]
         [1.     0.625 ]
         [1.     0.    ]
         [0.625  0.    ]]
        <BLANKLINE>
        Element Nodes, Areas, Points, Centroids, Materials
        [9, 0, 3, 2, 6], -0.30078125, [0.375 0.25 ], \
[0.32251082 0.24323593], rock
        [8, 7, 6, 9], 0.19921875, [0.875 0.25 ], \
[0.82352941 0.26838235], rock
        [2, 5, 1, 3], 0.19921875, [0.125 0.75 ], \
[0.17647059 0.73161765], sand
        [6, 2, 5, 4, 7], -0.30078125, [0.625 0.75 ], \
[0.67748918 0.75676407], sand
        <BLANKLINE>
        Interface Element Nodes and Neighbors
        [2, 5], [2, 3]
        [2, 3], [2, 0]
        [2, 6], [3, 0]
        [6, 7], [3, 1]
        [6, 9], [1, 0]
        <BLANKLINE>
        Boundary Element Nodes and Neighbors
        [1, 3], 2
        [0, 3], 0
        [4, 5], 3
        [1, 5], 2
        [4, 7], 3
        [8, 9], 1
        [0, 9], 0
        [7, 8], 1
        <BLANKLINE>
        <BLANKLINE>

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
        mesh_string = ('vcfempy.meshgen.PolyMesh2D\n'
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
                       + f'Mesh Generated = {self.mesh_valid}\n')

        # if mesh has been generated, print basic mesh information
        if self.mesh_valid:
            mesh_string += ('Number of Nodes = '
                            + f'{self.num_nodes}\n'
                            + 'Number of Elements = '
                            + f'{self.num_elements}\n'
                            + 'Number of Interface Elements = '
                            + f'{self.num_interface_elements}\n'
                            + 'Number of Boundary Elements = '
                            + f'{self.num_boundary_elements}\n\n')
        # otherwise, finish the string with an extra line break
        else:
            mesh_string += '\n'

        # check for verbose printing flag, return now if False
        if not self.verbose_printing:
            return mesh_string

        # otherwise, if verbose printing is True, continue printing
        # detailed mesh information

        if self.num_vertices:
            mesh_string += f'Vertices\n{self.vertices}\n\n'
        if self.num_boundary_vertices:
            mesh_string += f'Boundary Vertices\n{self.boundary_vertices}\n\n'
            mesh_string += f'Boundary Edges\n{self.boundary_edges}\n\n'
        if self.num_material_regions:
            for k, mr in enumerate(self.material_regions):
                mesh_string += f'Material Region {k}, {mr.material.name}\n'
                mesh_string += f'{mr.vertices}\n\n'
        if self.num_mesh_edges:
            mesh_string += 'Mesh Edges\n'
            for e in self.mesh_edges:
                mesh_string += '{e}\n'
            mesh_string += '\n'
        if self.num_nodes:
            mesh_string += f'Nodes\n{self.nodes}\n\n'
        if self.num_elements:
            mesh_string += ('Element Nodes, Areas, Points, Centroids, '
                            + 'Materials\n')
            for e, p in zip(self.elements, self.points):
                mesh_string += (f'{e.nodes}, {e.area}, {p}, {e.centroid}, '
                                + f'{e.material.name}\n')
            mesh_string += '\n'
        if self.num_interface_elements:
            mesh_string += 'Interface Element Nodes and Neighbors\n'
            for e in self.interface_elements:
                en = [self.elements.index(n) for n in e.neighbors]
                mesh_string += f'{e.nodes}, {en}\n'
            mesh_string += '\n'
        if self.num_boundary_elements:
            mesh_string += 'Boundary Element Nodes and Neighbors\n'
            for e in self.boundary_elements:
                en = self.elements.index(e.neighbor)
                mesh_string += f'{e.nodes}, {en}\n'
            mesh_string += '\n'
        return mesh_string

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

    def insert_boundary_vertices(self, index, boundary_vertices):
        """Insert one or more boundary vertex indices to the :c:`PolyMesh2D`.

        Parameters
        ----------
        index : int
            The index at which to insert the **boundary_vertices** into
            :m:`vertices`.
        boundary_vertices : int | list[int]
            The list of vertex indices to add to :m:`vertices`.

        Note
        -----
        Before inserting the values in **boundary_vertices**, it will be
        cast to a flattened `numpy.ndarray` of `int`.

        Raises
        ------
        TypeError
            If **index** cannot be cast to `int`.
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
        Before removing the values in **remove_vertices**, it will be cast to
        a flattened `numpy.ndarray` of `int`.

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

    def add_material_regions(self, material_regions, materials=None):
        """Add material regions to the :c:`PolyMesh2D`.

        Parameters
        ----------
        material_regions : `list` of [:c:`MaterialRegion2D` | `list[int]`]
            List of material regions to add to the :c:`PolyMesh2D`. If
            provided as `list[list[int]]` then each `list[int]` is passed to
            :c:`MaterialRegion2D` along with the corresponding material from
            **materials**.
        materials : `list` of :c:`vcfempy.materials.Material`, optional
            Materials corresponding to each material region to be added. This
            is ignored if **material_regions** is provided as a `list` of
            :c:`MaterialRegion2D`.

        Raises
        ------

        Examples
        --------
        """
        if material_regions is None:
            return
        if isinstance(material_regions, MaterialRegion2D):
            if material_regions in self.material_regions:
                raise ValueError('material region already in list')
            self.material_regions.append(material_regions)
            return
        all_int = True
        for k, mr in enumerate(material_regions):
            if isinstance(mr, MaterialRegion2D):
                all_int = False
                if mr in self.material_regions:
                    raise ValueError(f'material region {k} already in list')
                self.material_regions.append(mr)
            elif isinstance(mr, list):
                all_int = False
                if isinstance(materials, list):
                    self.material_regions.append(
                            MaterialRegion2D(self, mr, materials[k]))
                else:
                    self.material_regions.append(
                            MaterialRegion2D(self, mr, materials))
            elif not isinstance(mr, (int, np.integer)):
                raise ValueError(f'material region {k} has invalid type')
            elif not all_int:
                raise ValueError(f'material region {k} provided as int')
        if all_int:
            self.material_regions.append(
                   MaterialRegion2D(self, material_regions, materials))
        self.mesh_valid = False

    def add_mesh_edges(self, mesh_edges):
        """ Add mesh edges to PolyMesh2D.

        Parameters
        ----------
        mesh_edges : list[int] | list[list[int]]
            Lists of vertex indices defining edges to be maintained
            in mesh generation

        Returns
        -------
        None

        Raises
        ------
        TypeError
            type(mesh_edges) not in [NoneType, list]
            type(mesh_edges[k]) not in [int, numpy.int32, list]
            if mesh_edges is list of list of ints:
                type(mesh_edges[k][j]) not in [int, numpy.int32]
        ValueError
            if mesh_edges is list of ints:
                len(mesh_edges) != 2
                mesh_edges[k] >= self.num_vertices
            if mesh_edges is list of list of ints:
                len(mesh_edges[k]) != 2
                mesh_edges[k][j] >= self.num_vertices

        Examples
        --------
        """
        # basic type check of mesh_edges
        if type(mesh_edges) not in [type(None), list]:
            raise TypeError('type(mesh_edges) not in [NoneType, list]')
        # catch null case where mesh_edges is None or an empty list
        if mesh_edges is None or len(mesh_edges) == 0:
            pass
        # mesh_edges is a list[int]
        # Note: if here, we know that mesh_edges is a non-empty list
        elif type(mesh_edges[0]) in [int, np.int32]:
            # check that mesh_edges has the right length
            if len(mesh_edges) != 2:
                raise ValueError('mesh_edges given as list[int], '
                                 + 'but len(mesh_edges) != 2')
            # check that all items in mesh_edges are ints
            # and values are < self.num_vertices
            for v in mesh_edges:
                if type(v) not in [int, np.int32]:
                    raise TypeError('mesh_edges given as list[int], '
                                    + 'but type of contents not all '
                                    + 'in [int, numpy.int32]')
                if v >= self.num_vertices:
                    raise ValueError('mesh_edges given as list[int], '
                                     + 'but contains values >= num_vertices')
            # if here, mesh_edges is a valid list[int]
            # append it to the list of mesh edges
            # and invalidate the mesh
            self.mesh_edges.append([int(k) for k in mesh_edges])
            self.mesh_valid = False
        # mesh_edges is a list[list[int]]
        else:
            # check that all mesh edges are lists of len == 2
            # and contain ints < self.num_vertices
            for edge in mesh_edges:
                if type(edge) is not list:
                    raise TypeError('mesh_edges given as list[list[int]], '
                                    + 'but type of some contents is not list')
                if len(edge) != 2:
                    raise ValueError('mesh_edges given as list[list[int]], '
                                     + 'but some edges have len != 2')
                for v in edge:
                    if type(v) not in [int, np.int32]:
                        raise TypeError('mesh_edges given as list[list[int]], '
                                        + 'but type of some vertices not in '
                                        + '[int, numpy.int32]')
                    if v >= self.num_vertices:
                        raise ValueError('mesh_edges given as list[list[int]],'
                                         + ' but some vertices '
                                         + '>= num_vertices')
            # if here, mesh_edges is a valid list of list of ints
            # append each list to the list of material regions
            # and invalidate the mesh
            for edge in mesh_edges:
                self.mesh_edges.append([int(k) for k in edge])
            self.mesh_valid = False

    def generate_mesh(self, grid_size=[10, 10], alpha_rand=0.0,
                      verbose_plot=False):
        """ Generate polygonal mesh. """
        # generate seed points within boundary

        # set size of grid and degree of randomness
        # total number of points is nx*ny
        nx = grid_size[0]
        ny = grid_size[1]

        # get size parameters for grid
        Lx = np.max(self.vertices[self.boundary_vertices, 0]) \
            - np.min(self.vertices[self.boundary_vertices, 0])
        Ly = np.max(self._vertices[self.boundary_vertices, 1]) \
            - np.min(self.vertices[self.boundary_vertices, 1])
        dx = Lx/nx
        dy = Ly/ny
        d_scale = 0.5 * np.linalg.norm([dx, dy])

        # generate regular grid
        xc = np.linspace(np.min(self.vertices[self.boundary_vertices, 0])
                         + dx/2,
                         np.max(self.vertices[self.boundary_vertices, 0])
                         - dx/2, nx)
        yc = np.linspace(np.min(self.vertices[self.boundary_vertices, 1])
                         + dy/2,
                         np.max(self.vertices[self.boundary_vertices, 1])
                         - dy/2, ny)
        xc, yc = np.meshgrid(xc, yc)

        # shift points for hexagonal grid
        for k in range(xc.shape[0]):
            if k % 2:
                xc[k, :] -= 0.25*dx
            else:
                xc[k, :] += 0.25*dx

        # reshape grid into list of points
        self._points = np.vstack([xc.ravel(), yc.ravel()]).T

        if verbose_plot:
            plt.figure()
            ax = self.plot_boundaries()
            ax.plot(self.points[:, 0], self.points[:, 1], 'or')
            plt.savefig('mesh0.png')

        # randomly shift seed points
        xc_shift = alpha_rand*dx*(2*np.random.random([xc.size, 1]) - 1)
        yc_shift = alpha_rand*dy*(2*np.random.random([yc.size, 1]) - 1)
        self.points[:, 0] += xc_shift[:, 0]
        self.points[:, 1] += yc_shift[:, 0]

        # eliminate points that are outside the boundaries
        bpath = path.Path(self.vertices[self.boundary_vertices])
        in_bnd = bpath.contains_points(self.points)
        self._points = self.points[in_bnd]

        if verbose_plot:
            ax.clear()
            self.plot_boundaries(ax)
            ax.plot(self.points[:, 0], self.points[:, 1], 'or')
            plt.savefig('mesh1.png')

        # remove existing points near mesh edges
        # and add reflected points along mesh edges to capture them
        # in the mesh
        de_min = -d_scale
        for edge in self.mesh_edges:
            # get vertices
            e0 = self.vertices[edge[0]]
            e1 = self.vertices[edge[1]]
            ee = e1 - e0
            ee_len = np.linalg.norm(ee)
            de_max = ee_len + d_scale

            # find points near the edge for deletion
            keep_points = np.bool_(np.ones(len(self.points)))
            for j, p in enumerate(self.points):
                # find projection of the point onto the edge
                ep = p-e0
                pp = e0 + (np.dot(ep, ee) / np.dot(ee, ee)) * ee
                # check if point is close to the edge
                # and within the length of the edge
                # +/- d_scale
                d = np.linalg.norm(p-pp)
                de = np.sign(np.dot(pp-e0, ee)) * np.linalg.norm(pp-e0)
                if d < 0.5*d_scale and de >= de_min and de <= de_max:
                    keep_points[j] = False
            # delete points near the edge
            self._points = self.points[keep_points]

            # get unit vector in direction of edge and point step size
            # Note: Add 1 to number of points because 2 points will be added
            #       a half step off each end of the mesh edge to preserve the
            #       vertices at the ends
            ee_hat = ee / ee_len
            nn_hat = np.array([ee_hat[1], -ee_hat[0]])
            num_points = int(np.round(ee_len / d_scale)) + 1
            de = (ee_len + d_scale) / num_points

            # make list of points to add along edge
            # and add them to the overall point list
            new_points = []
            dp_list = np.linspace(-0.5*de, ee_len+0.5*de, num_points)
            for dp in dp_list:
                # add points on both sides of the edge
                new_points.append(e0 + dp*ee_hat + 0.25*d_scale*nn_hat)
                new_points.append(e0 + dp*ee_hat - 0.25*d_scale*nn_hat)
            self._points = np.vstack([self.points, new_points])

        # eliminate points that are outside the boundaries
        in_bnd = bpath.contains_points(self.points)
        self._points = self.points[in_bnd]

        if verbose_plot:
            ax.clear()
            self.plot_boundaries(ax)
            ax.plot(self.points[:, 0], self.points[:, 1], 'or')
            plt.savefig('mesh2.png')

        # add points to ensure boundary vertices are
        # captured in the mesh
        # Note: the added points differ depending on
        #       whether the vertex is convex, concave, or straight
        for k, edge in enumerate(self.boundary_edges):
            # get previous edge
            prv_edge = self.boundary_edges[k-1]

            # get boundary vertices
            bm1 = self.vertices[prv_edge[0]]
            b0 = self.vertices[edge[0]]
            bp1 = self.vertices[edge[1]]

            # get unit vectors in direction of adjacent edges
            bbf = bp1-b0
            bbr = bm1-b0
            d_bbf = np.linalg.norm(bbf)
            d_bbr = np.linalg.norm(bbr)
            bbf = bbf / d_bbf
            bbr = bbr / d_bbr

            # get unit vectors in direction normal to
            # perpendicular bisector of the vertex
            # Note: at convex or straight vertex, pp_hat is inward pointing
            #       at concave vertex, pp_hat is outward pointing
            pp_hat = bbf+bbr

            # check length of pp_hat, if non-zero normalize
            if np.linalg.norm(pp_hat) > 1.e-8:
                pp_hat = pp_hat / np.linalg.norm(pp_hat)
            # if length of pp_hat is zero, edge is straight
            # make pp_hat inward pointing normal
            else:
                pp_hat = np.array([bbf[1], -bbf[0]])
            # get tangential unit vector, normal to pp_hat
            vv_hat = np.array([pp_hat[1], -pp_hat[0]])

            # check for straight edge
            bbr_bbf_crs = np.cross(bbr, bbf)
            if np.abs(bbr_bbf_crs) < 1.e-8:
                # get local scale, in case adjacent edges are short
                d_scale_loc = np.min([d_scale, d_bbf, d_bbr])

                # delete points near vertex b0
                keep_points = np.bool_(np.ones(len(self.points)))
                for j, p in enumerate(self.points):
                    if np.linalg.norm(p-b0) < d_scale_loc:
                        keep_points[j] = False
                self._points = self.points[keep_points]

                # create two new points near concave vertex
                new_points = [b0 + d_scale_loc*(0.4*pp_hat + 0.8*vv_hat),
                              b0 + d_scale_loc*(0.4*pp_hat - 0.8*vv_hat)]
                self._points = np.vstack([self.points, new_points])

            # check for concave vertex
            elif bbr_bbf_crs < 0:
                # get local scale, in case adjacent edges are short
                d_scale_loc = np.min([d_scale, d_bbf, d_bbr])

                # delete points near vertex b0
                keep_points = np.bool_(np.ones(len(self.points)))
                for j, p in enumerate(self.points):
                    if np.linalg.norm(p-b0) < d_scale_loc:
                        keep_points[j] = False
                self._points = self.points[keep_points]

                # create two new points near concave vertex
                new_points = [b0 + 0.8*d_scale_loc*vv_hat,
                              b0 - 0.8*d_scale_loc*vv_hat]
                self._points = np.vstack([self.points, new_points])

            # otherwise, it is a convex vertex
            # check if adjacent edges are short
            elif d_bbf < d_scale or d_bbr < d_scale:
                # get local scale, in case adjacent edges are short
                d_scale_loc = np.min([d_scale, d_bbf, d_bbr])

                # delete points near vertex b0
                keep_points = np.bool_(np.ones(len(self.points)))
                for j, p in enumerate(self.points):
                    if np.linalg.norm(p-b0) < d_scale_loc:
                        keep_points[j] = False
                self._points = self.points[keep_points]

                # create new point near convex vertex
                # adjacent to a short boundary edge
                new_points = [b0 + 0.8*d_scale_loc*pp_hat]
                self._points = np.vstack([self.points, new_points])

        if verbose_plot:
            ax.clear()
            self.plot_boundaries(ax)
            ax.plot(self.points[:, 0], self.points[:, 1], 'or')
            plt.savefig('mesh3.png')

        # eliminate points that are outside the boundaries
        in_bnd = bpath.contains_points(self.points)
        self._points = self.points[in_bnd]

        if verbose_plot:
            ax.clear()
            self.plot_boundaries(ax)
            ax.plot(self.points[:, 0], self.points[:, 1], 'or')
            plt.savefig('mesh4.png')

        # reflect seed points about boundaries
        # this ensures a voronoi diagram with ridges along each boundary
        dmax = np.min([4.0*d_scale, Lx, Ly])
        reflected_points = []
        for p in self.points:
            for k, edge in enumerate(self.boundary_edges):
                # get previous and next edges
                prv_edge = self.boundary_edges[k-1]
                nxt_edge = self.boundary_edges[(k+1)
                                               % self.num_boundary_vertices]

                # get boundary vertices
                bm1 = self.vertices[prv_edge[0]]
                b0 = self.vertices[edge[0]]
                b1 = self.vertices[edge[1]]
                b2 = self.vertices[nxt_edge[1]]

                # set flags for convex vertices
                bbr0 = (bm1-b0) / np.linalg.norm(bm1-b0)
                bbf0 = (b1-b0) / np.linalg.norm(b1-b0)
                is_cvx0 = (np.cross(bbr0, bbf0) > 0)
                bbr1 = (b0-b1) / np.linalg.norm(b0-b1)
                bbf1 = (b2-b1) / np.linalg.norm(b2-b1)
                is_cvx1 = (np.cross(bbr1, bbf1) > 0)

                # project point onto boundary
                # bp = b0 + |a|cos(theta)*bhat
                #    = b0 + |a|cos(theta)*b / |b|
                #    = b0 + |a||b|cos(theta)*b / (|b||b|)
                #    = b0 + (a.b / b.b)*b
                # bhat = b / |b|
                # a.b = |a||b|cos(theta)
                # b.b = |b||b|
                bb = b1-b0
                bp = p-b0
                pp = b0 + (np.dot(bp, bb) / np.dot(bb, bb)) * bb
                dp = pp-p
                d = np.linalg.norm(dp)

                # get outward normal of current edge
                nhat = np.array([-bb[1], bb[0]]) / np.linalg.norm(bb)

                # check distance to boundary, and direction of dp
                # only reflect points within dmax of boundary segment
                # and where dp points outward
                if d < dmax and np.dot(dp, nhat) > 0:
                    # check whether vertices are convex
                    # Note: always reflect if vertices are convex
                    #       but at concave vertices only reflect if
                    #       the projected point pp is within the segment
                    db = pp-b0
                    db = np.sign(np.dot(db, bb)) * np.linalg.norm(db) \
                        / np.linalg.norm(bb)
                    if (is_cvx0 or db >= 0.0) and (is_cvx1 or db <= 1.0):
                        reflected_points.append(p + 2*dp)
        # convert reflected points list to array
        reflected_points = np.array(reflected_points)

        if verbose_plot:
            ax.clear()
            self.plot_boundaries(ax)
            ax.plot(self.points[:, 0], self.points[:, 1], 'or')
            ax.plot(reflected_points[:, 0], reflected_points[:, 1], 'xr')
            plt.savefig('mesh5.png')

        # create Voronoi diagram of seed points
        all_points = np.vstack([self.points, reflected_points])
        vor = Voronoi(all_points)

        # get list of Voronoi regions inside the boundary
        npoint = len(self.points)
        point_region = vor.point_region[:npoint]

        # compile list of elements to keep
        # Note: this is a temporary variable,
        #       element objects will be created later
        element_nodes = []
        for k in point_region:
            element_nodes.append(vor.regions[k])

        # compile list of vertices to keep
        nodes_to_keep = set()
        for e in element_nodes:
            for k in e:
                nodes_to_keep.add(k)
        nodes_to_keep = list(nodes_to_keep)

        # obtain vertices
        nodes = []
        for k in nodes_to_keep:
            nodes.append(vor.vertices[k])
        self._nodes = np.array(nodes)

        # obtain ridge information to keep
        # Note: these are lists indicating neighbouring elements
        #       and the edges between elements
        element_neighbors = []
        element_edges = []
        for rp, rv in zip(vor.ridge_points, vor.ridge_vertices):

            # check if ridge contains at least one point inside the boundary
            if rp[0] < npoint or rp[1] < npoint:

                # save the ridge
                # if either ridge point was outside the boundary,
                # change it to -1
                element_neighbors.append([rp[0]
                                          if rp[0] < npoint else -1,
                                          rp[1]
                                          if rp[1] < npoint else -1])
                # save the ridge vertices
                element_edges.append(rv)

        # convert node indices to reduced set of those kept in/on boundary
        node_dict = {n: k for k, n in enumerate(nodes_to_keep)}
        for k, e in enumerate(element_nodes):
            for j, v in enumerate(element_nodes[k]):
                element_nodes[k][j] = node_dict[element_nodes[k][j]]
        for k, e in enumerate(element_edges):
            for j, v in enumerate(element_edges[k]):
                element_edges[k][j] = node_dict[element_edges[k][j]]

        # determine material type of each element
        m0 = mtl.Material('NULL')
        element_materials = [m0 for k, _ in enumerate(element_nodes)]
        element_materials = np.array(element_materials)
        for mr in self.material_regions:
            bpath = path.Path(self.vertices[mr.vertices, :])
            in_bnd = bpath.contains_points(self.points)
            element_materials[in_bnd] = mr.material

        # create list of elements
        self._elements = []
        for e, m in zip(element_nodes, element_materials):
            # create a new element and add it to the list of elements
            # Note: here, the first argument self initializes the element
            #       with a reference to the current mesh as its parent mesh
            self.elements.append(PolyElement2D(self, e, m))

        # create lists of interface and boundary elements
        self._interface_elements = []
        self._boundary_elements = []
        for ee, en in zip(element_edges, element_neighbors):
            # check for boundary element
            if en[0] == -1 or en[1] == -1:
                neighbor = (self.elements[en[1]] if en[0] == -1
                            else self.elements[en[0]])
                self.boundary_elements.append(
                        BoundaryElement2D(self, ee, neighbor))
            # otherwise, we have an interface element
            # assign material type from first neighbor
            else:
                neighbors = [self.elements[n] for n in en]
                material = neighbors[0].material
                self.interface_elements.append(
                        InterfaceElement2D(self, material, ee, neighbors))

        # set mesh valid
        # Note: the setter will perform checks for mesh validity
        self.mesh_valid = True

        if verbose_plot:
            ax.clear()
            self.plot_mesh(ax)
            ax.plot(self.points[:, 0], self.points[:, 1], 'or')
            plt.savefig('mesh6.png')

    def plot_boundaries(self, ax=None, line_type='-k'):
        """ Plot out PolyMesh2D boundaries. """
        if ax is None:
            ax = plt.gca()
        for edge in self.boundary_edges:
            ax.plot(self.vertices[edge, 0], self.vertices[edge, 1], line_type)
        return ax

    def plot_vertices(self, ax=None, line_type='sk', markersize=5.0):
        """ Plot out PolyMesh2D vertices. """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.vertices[:, 0],
                self.vertices[:, 1],
                line_type, markersize=markersize)
        return ax

    def plot_mesh_edges(self, ax=None, line_type='-k'):
        """ Plot out PolyMesh2D mesh edges. """
        if ax is None:
            ax = plt.gca()
        for edge in self.mesh_edges:
            ax.plot(self.vertices[edge, 0],
                    self.vertices[edge, 1],
                    line_type)
        return ax

    def plot_material_regions(self, ax=None, line_type='-k', fill=True):
        """ Plot out PolyMesh2D material regions. """
        if ax is None:
            ax = plt.gca()
        for mr in self.material_regions:
            mr.plot(ax, line_type, fill)
        return ax

    def plot_mesh(self, ax=None, line_type=':k', fill=True):
        """ Plot out PolyMesh2D elements. """
        if ax is None:
            ax = plt.gca()
        for e in self.elements:
            e.plot(ax, line_type, fill)
        for e in self.interface_elements:
            ax.plot(self.nodes[e.nodes, 0],
                    self.nodes[e.nodes, 1],
                    line_type)
        return ax

    def plot_mesh_boundaries(self, ax=None, line_type='--b'):
        """ Plot out PolyMesh2D element edges that are on the boundaries. """
        if ax is None:
            ax = plt.gca()
        for e in self.boundary_elements:
            ax.plot(self.nodes[e.nodes, 0],
                    self.nodes[e.nodes, 1],
                    line_type)
        return ax

    def plot_mesh_nodes(self, ax=None, line_type='ok', markersize=2.0):
        """ Plot out PolyMesh2D nodes. """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.nodes[:, 0],
                self.nodes[:, 1],
                line_type, markersize=markersize)
        return ax

    def plot_quadrature_points(self, ax=None, line_type='+k', markersize=1.5):
        """ Plot out PolyMesh2D quadrature points. """
        if ax is None:
            ax = plt.gca()
        for e in self.elements:
            e.plot_quadrature_points(ax, line_type, markersize)
        return ax


class MaterialRegion2D():
    """ A class for defining material regions and their attributes. """

    def __init__(self, mesh, vertices=None, material=None):
        self.mesh = mesh

        self._vertices = []
        self.insert_vertices(0, vertices)

        self.material = material

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def vertices(self):
        return self._vertices

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        if type(mesh) not in [type(None), PolyMesh2D]:
            raise TypeError('type(mesh) not in [NoneType, '
                            + 'vcfempy.meshgen.PolyMesh2D]')
        self._mesh = mesh

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        if type(material) not in [type(None), mtl.Material]:
            raise TypeError('type(material) not in [NoneType, '
                            + 'vcfempy.materials.Material]')
        # change the material type of the material region
        # invalidate parent mesh since existing mesh would have
        # incorrect material assigned to elements
        self._material = material
        self.mesh.mesh_valid = False

    def insert_vertices(self, i, vertices):
        # basic type check of vertices
        if type(vertices) not in [type(None), int, np.int32, list]:
            raise TypeError('type(vertices) not in [NoneType, int, '
                            + 'numpy.int32, list]')

        # catch case of single vertex
        if type(vertices) in [int, np.int32]:
            vertices = [vertices]

        # if vertices given as None or empty list, return early
        # Note: if here, we know that vertices is either None or a list
        #       (i.e. not int) because of earlier type check and first if block
        # in this case, we can skip re-processing the mesh since it
        # is still valid
        elif vertices is None or len(vertices) == 0:
            return

        # vertices is a non-empty list
        # Note: we know this because of earlier type check on vertices

        # check contents of vertices
        for v in vertices:
            # check type is integer
            if type(v) not in [int, np.int32]:
                raise TypeError('type of vertices contents not in [int, '
                                + 'numpy.int32]')
            # check value of vertex is less than number of vertices
            if v >= self.mesh.num_vertices:
                raise ValueError('vertices values must all be less than '
                                 + 'number of vertices in the parent mesh')

        # insert vertices
        # Note: if here, we know that vertices is a valid list of ints
        # reset the mesh
        vertices.reverse()
        for k in vertices:
            self.vertices.insert(i, int(k))
        self.mesh.mesh_valid = False

    def plot(self, ax=None, fill=True, line_type='-k'):
        if ax is None:
            ax = plt.gca()
        if fill:
            ax.fill(self.mesh.vertices[self.vertices, 0],
                    self.mesh.vertices[self.vertices, 1],
                    color=self.material.color)
        vlist = [self.vertices[j % self.num_vertices]
                 for j in range(self.num_vertices+1)]
        ax.plot(self.mesh.vertices[vlist, 0],
                self.mesh.vertices[vlist, 1],
                line_type)


class PolyElement2D():
    """
    A class for polygonal element geometry and quadrature generation

    Parameters
    ----------
    mesh : vcfempy.meshgen.PolyMesh2D
        The parent mesh
    nodes : None or list of int, optional
        The list of node indices from the parent mesh
        Can be in CW or CCW order
    material : None or vcfempy.materials.Material, optional
        The material type assigned to the element

    Examples
    --------
    """

    def __init__(self, mesh, nodes=None, material=None):
        # initialize parent mesh
        self.mesh = mesh

        # initialize nodes
        self._nodes = []
        self.insert_nodes(0, nodes)

        # initialize material
        self.material = material

        # initialize geometry and quadrature attributes
        self.invalidate_properties()

    @property
    def num_nodes(self):
        """Number of nodes in the element

        Returns
        -------
        int
            The number of nodes in the element

        Examples
        --------
        """
        return len(self.nodes)

    @property
    def nodes(self):
        """List of element nodes

        Returns
        -------
        list[int]
            The list of node indices in the element

        Examples
        --------
        """
        return self._nodes

    @property
    def mesh(self):
        """Parent mesh

        Parameters
        ----------
        mesh : None | PolyMesh2D
            The parent mesh to assign to the element

        Returns
        -------
        None | PolyMesh2D
            The parent mesh assigned to the element

        Raises
        ------
        TypeError
            type(mesh) not in [NoneType, PolyMesh2D]

        Examples
        --------
        """
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        # basic type check of mesh
        if type(mesh) not in [type(None), PolyMesh2D]:
            raise TypeError('type(mesh) not in [NoneType, '
                            + 'vcfempy.meshgen.PolyMesh2D]')
        self._mesh = mesh

    @property
    def material(self):
        """Material type of the PolyElement2D

        Parameters
        ----------
        material : None | vcfempy.materials.Material
            The material to assign to the element

        Returns
        -------
        None | vcfempy.materials.Material
            The material assigned to the element

        Raises
        ------
        TypeError
            If type(material) not in [NoneType, vcfempy.materials.Material]

        Examples
        --------
        """
        return self._material

    @material.setter
    def material(self, material):
        # basic type check of material
        if type(material) not in [type(None), mtl.Material]:
            raise TypeError('type(material) not in [NoneType, '
                            + 'vcfempy.materials.Material]')
        self._material = material

    @property
    def area(self):
        """Element area

        Returns
        -------
        float
            The element area
            Positive if nodes in CCW order, negative if nodes in CW order

        Examples
        --------
        """
        # check if area has not been calculated
        # if not, calculate it
        if self._area is None:
            self._area = polygon_area(self.mesh.nodes[self.nodes])
        return self._area

    @property
    def centroid(self):
        """Element centroid coordinates

        Returns
        -------
        numpy.array, shape = (2, )
            The coordinates of the element centroid

        Examples
        --------
        """
        # check if centroid has not been calculated
        # if not, calculate it
        if self._centroid is None:
            self._centroid = polygon_centroid(self.mesh.nodes[self.nodes],
                                              self.area)[0]
        return self._centroid

    @property
    def quad_points(self):
        """Element quadrature point coordinates

        Returns
        -------
        numpy.ndarray, shape = (num_quad_points, 2)
            The coordinates of the element quadrature points

        Examples
        --------
        """
        # check if quadrature has not been generated
        # if not, generate it
        if self._quad_points is None:
            self.generate_quadrature()
        return self._quad_points

    @property
    def quad_weights(self):
        """Element quadrature point weights

        Returns
        -------
        numpy.ndarray, shape = (num_quad_points, )
            The weights of the element quadrature points
            Should sum to 1.0

        Examples
        --------
        """
        # check if quadrature has not been generated
        # if not, generate it
        if self._quad_weights is None:
            self.generate_quadrature()
        return self._quad_weights

    @property
    def quad_integrals(self):
        """Element quadrature basis function integrals

        Returns
        -------
        numpy.ndarray, size = (num_quad_basis_functions, )
            The values of the element quadrature basis function integrals
            Computed in local coordinates with origin at centroid
            Basis functions depends on number of nodes and
            mesh.high_order_quadrature
            3 nodes: 1
            up to 5 nodes: ..., x**2, x*y, y**2
            up to 7 nodes: ..., x**4, x**3 * y, ... y**4
            mesh.high_order_quadrature or up to 10 nodes:
            ..., x**6, x**5 * y, ... y**6

        Examples
        --------
        """
        # check if quadrature has not been generated
        # if not, generate it
        if self._quad_integrals is None:
            self.generate_quadrature()
        return self._quad_integrals

    def insert_nodes(self, i, nodes):
        """Insert one or more nodes at index i
        nodes can be int or list of ints
        """
        # basic type check of nodes
        if type(nodes) not in [type(None), int, np.int32, list]:
            raise TypeError('type(nodes) not in [NoneType, int, '
                            + 'numpy.int32, list]')

        # catch case of single node
        if type(nodes) in [int, np.int32]:
            nodes = [nodes]
        # if nodes given as None or empty list, return early
        # Note: if here, we know that nodes is either None or a list
        #       (i.e. not int) because of earlier type check and first if block
        # in this case, we can skip re-processing the mesh since it is
        # still valid
        elif nodes is None or len(nodes) == 0:
            return

        # nodes is a non-empty list
        # Note: we know this because of earlier type check on nodes
        # check contents of nodes
        for v in nodes:
            # check type is integer
            if type(v) not in [int, np.int32]:
                raise TypeError('type of nodes contents not in [int, '
                                + 'numpy.int32]')
            # check value of node is less than number of nodes in parent mesh
            if v >= self.mesh.num_nodes:
                raise ValueError('nodes values must all be less than number '
                                 + 'of nodes in the parent mesh')

        # insert nodes
        # Note: if here, we know that nodes is a valid list of ints
        # nodes were added, so reset element properties
        nodes.reverse()
        for k in nodes:
            self.nodes.insert(i, int(k))
        self.invalidate_properties()

    def invalidate_properties(self):
        """Resets computed element properties
        Should be called whenever nodes is changed
        """
        self._area = None
        self._centroid = None
        self._quad_points = None
        self._quad_weights = None
        self._quad_integrals = None

    def generate_quadrature(self):
        """Generate quadrature points and weights for a PolyElement2D
        Determines correct quadcon method to call depending on
        num_nodes and mesh.high_order_quadrature
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
        # only require linear integration over a triangle
        # one integration point is sufficient
        self._quad_points = np.zeros((1, 2))
        self._quad_weights = np.array([1.])
        self._quad_integrals = np.array([self.area])

    def _quadcon5(self):
        vertices = self.mesh.nodes[self.nodes] - self.centroid
        cent = np.zeros(2)
        area = self.area

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

        # loop over vertices
        n = len(vertices)
        for k, v0 in enumerate(vertices):
            # form triangle with 2 vertices + centroid
            v1 = vertices[(k+1) % n]
            x = np.vstack([cent, v0, v1])

            # triangle area
            detJ = 0.5 * np.abs((x[1, 0]-x[0, 0]) * (x[2, 1]-x[0, 1])
                                - (x[2, 0]-x[0, 0]) * (x[1, 1]-x[0, 1]))

            # perform Gaussian integration over triangle
            for wj, Nj in zip(w, N):
                xj = Nj @ x
                phi += detJ * wj * np.array([1.,
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
        # point weights
        # Note: if nq > nphi, this is a least squares solution
        wq = np.linalg.lstsq(PHI, phi, rcond=None)[0]
        wq /= np.abs(area)

        # set element quadrature (private) attributes
        self._quad_points = xq
        self._quad_weights = wq
        self._quad_integrals = phi

    def _quadcon7(self):
        vertices = self.mesh.nodes[self.nodes] - self.centroid
        cent = np.zeros(2)
        area = self.area

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

        # loop over vertices
        n = len(vertices)
        for k, v0 in enumerate(vertices):
            # form triangle with 2 vertices + centroid
            v1 = vertices[(k+1) % n]
            x = np.vstack([cent, v0, v1])

            # triangle area
            detJ = 0.5 * np.abs((x[1, 0]-x[0, 0]) * (x[2, 1]-x[0, 1])
                                - (x[2, 0]-x[0, 0]) * (x[1, 1]-x[0, 1]))

            # perform Gaussian integration over triangle
            for wj, Nj in zip(w, N):
                xj = Nj @ x
                phi += detJ * wj * np.array([1.,
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
        # point weights
        # Note: if nq > nphi, this is a least squares solution
        wq = np.linalg.lstsq(PHI, phi, rcond=None)[0]
        wq /= np.abs(area)

        # set element quadrature (private) attributes
        self._quad_points = xq
        self._quad_weights = wq
        self._quad_integrals = phi

    def _quadcon10(self):
        vertices = self.mesh.nodes[self.nodes] - self.centroid
        cent = np.zeros(2)
        area = self.area

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

        # loop over vertices
        n = len(vertices)
        for k, v0 in enumerate(vertices):
            # form triangle with 2 vertices + centroid
            v1 = vertices[(k+1) % n]
            x = np.vstack([cent, v0, v1])

            # triangle area
            detJ = 0.5 * np.abs((x[1, 0]-x[0, 0]) * (x[2, 1]-x[0, 1])
                                - (x[2, 0]-x[0, 0]) * (x[1, 1]-x[0, 1]))

            # perform Gaussian integration over triangle
            for wj, Nj in zip(w, N):
                xj = Nj @ x
                phi += detJ * wj * np.array([1.,
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
        # point weights
        # Note: if nq > nphi, this is a least squares solution
        wq = np.linalg.lstsq(PHI, phi, rcond=None)[0]
        wq /= np.abs(area)

        # set element quadrature (private) attributes
        self._quad_points = xq
        self._quad_weights = wq
        self._quad_integrals = phi

    def plot(self, ax=None, line_type=':k', fill=True, borders=False):
        """Plots the element
        Can provide a matplotlib.pyplot.axis or if None will
        use matplotlib.pyplot.gca()
        If fill, will fill the area with material.color
        If borders, will plot element borders with line_type
        """
        if ax is None:
            ax = plt.gca()
        if fill:
            ax.fill(self.mesh.nodes[self.nodes, 0],
                    self.mesh.nodes[self.nodes, 1],
                    color=self.material.color)
        if borders:
            vlist = [self.nodes[j % self.num_nodes]
                     for j in range(self.num_nodes+1)]
            ax.plot(self.mesh.nodes[vlist, 0],
                    self.mesh.nodes[vlist, 1],
                    line_type)
        return ax

    def plot_quadrature_points(self, ax=None, line_type='+k', markersize=1.5):
        """Plots element quadrature points
        Can provide a matplotlib.pyplot.axis or if None will
        use matplotlib.pyplot.gca()
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.quad_points[:, 0] + self.centroid[0],
                self.quad_points[:, 1] + self.centroid[1],
                line_type, markersize=markersize)
        return ax


class InterfaceElement2D():
    """A class for interfaces between neighboring :c:`PolyElement2D` elements
    in a :c:`PolyMesh2D`.

    Parameters
    ----------
    mesh : :c:`PolyMesh2D`
        The parent mesh
    material : :c:`vcfempy.materials.Material`, optional
        The material type
    nodes : list[int], optional
        Initial list of node indices from the parent mesh that are contained
        in the :c:`InterfaceElement2D`
    neighbors: `list` of :c:`PolyElement2D`, optional
        List of neighboring :c:`PolyElement2D` from the parent mesh
    width : float, optional
        The element width in the direction normal to the length of the
        :c:`InterfaceElement2D`

    Examples
    --------
    """

    def __init__(self, mesh, material=None, nodes=None, neighbors=None,
                 width=None):
        self._mesh = None
        self.mesh = mesh

        self._material = None
        self.material = material

        self._width = 0.
        self.width = width

        self._nodes = []
        self.insert_nodes(0, nodes)

        self._neighbors = []
        self.add_neighbors(neighbors)

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        if type(mesh) not in [type(None), PolyMesh2D]:
            raise TypeError('type(mesh) not in [NoneType, '
                            + 'vcfempy.meshgen.PolyMesh2D]')
        self._mesh = mesh

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        if type(material) not in [type(None), mtl.Material]:
            raise TypeError('type(material) not in [NoneType, '
                            + 'vcfempy.materials.Material]')
        self._material = material

    @property
    def width(self, val):
        return self._width

    @width.setter
    def width(self, val):
        # try to cast width to float
        # this will raise a ValueError if val is not float-like
        if val is None:
            self._width = 0.
        else:
            self._width = float(val)

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def nodes(self):
        return self._nodes

    def insert_nodes(self, index, nodes=None):
        """Insert one or more nodes at index i
        nodes can be int or list of ints
        """
        # basic type check of nodes
        # nodes can only be None or a list because an InterfaceElement2D can
        # only have 0, 2, or 4 nodes
        if type(nodes) not in [type(None), list]:
            raise TypeError('type(nodes) not in [NoneType, list]')

        # if nodes given as None or empty list, return early
        if nodes is None or len(nodes) == 0:
            return

        # nodes is a non-empty list
        # Note: we know this because of earlier type check on nodes

        # catch incorrect number of nodes
        # interfaces can only have 0, 2, or 4 nodes
        new_num_nodes = self.num_nodes + len(nodes)
        if new_num_nodes % 2 == 1 or new_num_nodes > 4:
            raise ValueError('InterfaceElement2D can only have 0, 2, or 4 '
                             + 'nodes')

        # check contents of nodes
        for v in nodes:
            # check type is integer
            if type(v) not in [int, np.int32]:
                raise TypeError('type of nodes contents not in [int, '
                                + 'numpy.int32]')
            # check value of node is less than number of nodes in parent mesh
            if v >= self.mesh.num_nodes:
                raise ValueError('nodes values must all be less than number '
                                 + 'of nodes in the parent mesh')

        # insert nodes
        # Note: if here, we know that nodes is a valid list of ints
        # nodes were added, so reset element properties
        nodes.reverse()
        for n in nodes:
            self.nodes.insert(index, int(n))
        self.invalidate_properties()

    @property
    def num_neighbors(self):
        return len(self.neighbors)

    @property
    def neighbors(self):
        return self._neighbors

    def add_neighbors(self, neighbors):
        # basic type check of neighbors
        # Note: neighbors must be either None or a list because there can
        #       only be 0 or 2 neighbors
        if type(neighbors) not in [type(None), list]:
            raise ValueError('type of neighbors not in [NoneType, list]')

        # check for early return if neighbors is empty
        if neighbors is None or len(neighbors) == 0:
            return

        # if here, we know that neighbors is a non-empty list

        # check for correct length of neighbors
        # the new number must be 2 because there can only be 0 or 2
        # neighbors and we know neighbors is not empty
        if self.num_neighbors + len(neighbors) != 2:
            raise ValueError('InterfaceElement2D can only have 0 or 2 '
                             + 'neighbors')

        # check contents of neighbors
        # they must be PolyElement2D elements with the same parent mesh
        for n in neighbors:
            if type(n) is not PolyElement2D:
                raise TypeError('type of all neighbors must be '
                                + 'vcfempy.meshgen.PolyElement2D')
            if n.mesh is not self.mesh:
                raise ValueError('vcfempy.meshgen.InterfaceElement2D must '
                                 + 'have same parent mesh as its neighbors')

        # passed all checks, add neighbors to the interface element
        for n in neighbors:
            self.neighbors.append(n)

    def invalidate_properties(self):
        self._length = None

    @property
    def length(self):
        if self._length is None and self.num_nodes >= 2:
            n0 = self.mesh.nodes[self.nodes[0]]
            n1 = self.mesh.nodes[self.nodes[1]]
            self._length = np.linalg.norm(n1-n0)
        return self._length


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
        self._mesh = None
        self.mesh = mesh

        self._nodes = []
        self.insert_nodes(0, nodes)

        self.neighbor = neighbor

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        if type(mesh) not in [type(None), PolyMesh2D]:
            raise TypeError('type(mesh) not in [NoneType, '
                            + 'vcfempy.meshgen.PolyMesh2D]')
        self._mesh = mesh

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def nodes(self):
        return self._nodes

    def insert_nodes(self, index, nodes=None):
        """Insert one or more nodes at index i
        nodes can be int or list of ints
        """
        # basic type check of nodes
        # nodes can only be None or a list because a BoundaryElement2D
        # can only have 0 or 2 nodes
        if type(nodes) not in [type(None), list]:
            raise TypeError('type(nodes) not in [NoneType, list]')

        # if nodes given as None or empty list, return early
        if nodes is None or len(nodes) == 0:
            return

        # nodes is a non-empty list
        # catch incorrect number of nodes
        # boundary elements can only have 0 or 2 nodes
        if self.num_nodes + len(nodes) != 2:
            raise ValueError('vcfempy.BoundaryElement2D can only have '
                             + '0 or 2 nodes')

        # check contents of nodes
        for n in nodes:
            # check type is integer
            if type(n) not in [int, np.int32]:
                raise TypeError('type of nodes contents not in [int, '
                                + 'numpy.int32]')
            # check value of node is less than number of nodes in parent mesh
            if n >= self.mesh.num_nodes:
                raise ValueError('nodes values must all be less than number '
                                 + 'of nodes in the parent mesh')

        # insert nodes
        # Note: if here, we know that nodes is a valid list of ints
        # nodes were added, so reset element properties
        nodes.reverse()
        for n in nodes:
            self.nodes.insert(index, int(n))
        self.invalidate_properties()

    @property
    def neighbor(self):
        return self._neighbor

    @neighbor.setter
    def neighbor(self, n):
        # basic type check of neighbor
        if type(n) not in [type(None), PolyElement2D]:
            raise TypeError('type(n) not in [NoneType, PolyElement2D]')
        # check for matching parent mesh
        if n is not None and self.mesh is not n.mesh:
            raise ValueError('neighbor must have same parent mesh')
        self._neighbor = n

    def invalidate_properties(self):
        self._length = None

    @property
    def length(self):
        if self._length is None and self.num_nodes == 2:
            n0 = self.mesh.nodes[self.nodes[0]]
            n1 = self.mesh.nodes[self.nodes[1]]
            self._length = np.linalg.norm(n1-n0)
        return self._length


def polygon_area(x):
    n = len(x)
    area = 0.
    for k, v0 in enumerate(x):
        vm1 = x[k-1]
        vp1 = x[(k+1) % n]
        area += v0[0] * (vp1[1] - vm1[1])
    return 0.5*area


def polygon_centroid(x, area=None):
    if area is None:
        area = polygon_area(x)
    n = len(x)
    cent = np.zeros(2)
    for k, v0 in enumerate(x):
        v1 = x[(k+1) % n]
        d = v0[0]*v1[1] - v0[1]*v1[0]
        cent += (v0+v1) * d
    cent /= (6. * area)
    return cent, area
