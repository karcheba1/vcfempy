"""A module containing attributes, functions, classes and methods 
for meshes in the Voronoi Cell Finite Element Method (VCFEM).

Attributes
----------
"""

# code version
ver = '0.0.1'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
from scipy.spatial import Voronoi as Voronoi

import vcfempy.materials as mtl

class PolyMesh2D():
    """A class for 2D polygonal mesh generation.
    
    Properties
    ----------
    vertices : ndarray, immutable, shape (nvertices, 2)
        The list of (x,y) coordinates of vertices defining the mesh
    boundary_vertices : list of ints, immutable
        The clockwise list of vertices defining the analysis boundary polygon
    boundary_edges : list of list of ints, immutable, shape (nboundaryedges, 2)
        The list of edges defining the vertices in each boundary edge
        
    Private Attributes
    ------------------
    _vertices : ndarray, shape (nvertices, 2)
        The list of (x,y) coordinates of vertices defining the mesh
    _boundary_vertices : list of ints
        The clockwise list of vertices defining the analysis boundary polygon
    _boundary_edges : list of list of ints, shape (nboundaryedges, 2)
        The list of edges defining the vertices in each boundary edge
        
    Examples
    --------
    """
    
    def __init__(self, \
                 vertices = None, boundary_vertices = None, \
                 material_regions = None, materials = None, \
                 mesh_edges = None):
        """Create a new PolyMesh2D object.
        
        Parameters
        ----------
        vertices : list of lists of (ints or floats) | ndarray of (ints or floats), optional, shape (nvertices, 2)
            Initial vertices to be added to the PolyMesh2D
        boundary_vertices : int | list of ints, optional
            Initial list of boundary vertices to be added
        material_regions : list of ints | list of list of ints, optional
            Initial list(s) of material region vertices to be added
        materials : list of vcfempy.materials.Material
            Initial list of material types, optional, len(materials) == len(material_regions)
        mesh_edges : list of ints | list of list of ints, optional
            Initial list(s) defining non-boundary edges to be preserved in the mesh generation
        
        Returns
        -------
        PolyMesh2D
            A PolyMesh2D object
        
        Raises
        ------
        None
        
        Examples
        --------
        >>> PolyMesh2D(boundary_vertices = 0)
        Traceback (most recent call last):
        ...
        ValueError: boundary_vertices values must all be less than number of vertices
        """ 
        
        # initialize flags for 
        #      verbose printing
        #      high order quadrature in all elements
        self.set_verbose_printing(False)
        self.set_high_order_quadrature(False)
        
        # initialize vertices
        self._vertices = None
        self.add_vertices(vertices)
        
        # initialize boundary vertices and edges
        self._boundary_vertices = []
        self.insert_boundary_vertices(0, boundary_vertices)
            
        # initialize boundary edges and mesh properties
        # Note: Although inserting boundary vertices sometimes does this
        #       this is still necessary in case boundary_vertices is None or an empty list
        self.generate_boundary_edges()
        self.invalidate_mesh()
        
        # initialize material regions
        self._material_regions = []
        self.add_material_regions(material_regions, materials)
        
        # initialize mesh edges
        self._mesh_edges = []
        self.add_mesh_edges(mesh_edges)
        
                
    @property
    def num_vertices(self):
        """ Getter for number of vertices in PolyMesh2D. """
        if self._vertices is None:
            return 0
        else:
            return self._vertices.shape[0]
        
    @property
    def num_boundary_vertices(self):
        """ Getter for number of boundary vertices in PolyMesh2D. """
        return len(self.boundary_vertices)
        
    @property
    def num_boundary_edges(self):
        """ Getter for number of boundary edges in PolyMesh2D. """
        return len(self.boundary_edges)
        
    @property
    def num_material_regions(self):
        """ Getter for number of material regions in PolyMesh2D. """
        return len(self.material_regions)
    
    @property
    def num_mesh_edges(self):
        """ Getter for number of mesh edges in PolyMesh2D. """
        return len(self.mesh_edges)
        
    @property
    def num_nodes(self):
        """ Getter for number of nodes in PolyMesh2D. """
        if self._nodes is None:
            return 0
        else:
            return self._nodes.shape[0]
        
    @property
    def num_elements(self):
        """ Getter for number of elements in PolyMesh2D. """
        return len(self.elements)
        
    @property
    def num_element_edges(self):
        """ Getter for number of element edges in PolyMesh2D. """
        return len(self.element_edges)
    
    @property
    def num_nodes_per_element(self):
        """ Getter for number of nodes per element in PolyMesh2D. """
        return self._num_nodes_per_element
        
    @property
    def num_points(self):
        """ Getter for number of element seed points in PolyMesh2D. """
        if self._points is None:
            return 0
        else:
            return self._points.shape[0]
                                    
    @property
    def vertices(self):
        """ Getter for vertices in PolyMesh2D. """
        return self._vertices
    
    @property
    def boundary_vertices(self):
        """ Getter for list of boundary_vertices in PolyMesh2D. """
        return self._boundary_vertices
    
    @property
    def boundary_edges(self):
        """ Getter for list of boundary_edges in PolyMesh2D. """
        return self._boundary_edges
    
    @property
    def material_regions(self):
        """ Getter for list of material_regions in PolyMesh2D. """
        return self._material_regions
    
    @property
    def mesh_edges(self):
        """ Getter for list of mesh edges in PolyMesh2D. """
        return self._mesh_edges
    
    @property
    def mesh_valid(self):
        """ Getter for mesh valid flag for PolyMesh2D. """
        return self._mesh_valid
    
    @property
    def nodes(self):
        """ Getter for nodes in PolyMesh2D. """
        return self._nodes
    
    @property
    def points(self):
        """ Getter for element seed points in PolyMesh2D. """
        return self._points
    
    @property
    def elements(self):
        """ Getter for elements in PolyMesh2D. """
        return self._elements
    
    @property
    def element_neighbors(self):
        """ Getter for list of element neighbors in PolyMesh2D. """
        return self._element_neighbors
    
    @property
    def element_edges(self):
        """ Getter for list of element edges in PolyMesh2D. """
        return self._element_edges
    
    @property
    def element_materials(self):
        """ Getter for list of element materials in PolyMesh2D. """
        return self._element_materials
    
    @property
    def element_areas(self):
        """ Getter for list of element areas in PolyMesh2D. """
        return self._element_areas
    
    @property
    def element_centroids(self):
        """ Getter for list of element centroids in PolyMesh2D. """
        return self._element_centroids
    
    @property
    def element_quad_points(self):
        """ Getter for list of element quadrature points in PolyMesh2D. """
        return self._element_quad_points
    
    @property
    def element_quad_weights(self):
        """ Getter for list of element quadrature weights in PolyMesh2D. """
        return self._element_quad_weights
    
    @property
    def high_order_quadrature(self):
        """ Getter for high order quadrature flag. """
        return self._high_order_quadrature
    
    def set_high_order_quadrature(self, flag = True):
        """ Setter for high order quadrature flag. """
        if type(flag) in [bool, np.bool_]:
            self._high_order_quadrature = bool(flag)
        else:
            raise TypeError('type(flag) not in [bool, numpy.bool_]')
    
    @property
    def verbose_printing(self):
        """ Getter for verbose printing flag. """
        return self._verbose_printing
    
    def set_verbose_printing(self, flag = True):
        """ Setter for verbose printing flag. """
        if type(flag) in [bool, np.bool_]:
            self._verbose_printing = bool(flag)
        else:
            raise TypeError('type(flag) not in [bool, numpy.bool_]')
        
        
    def __str__(self):
        """Print out detailed information about the PolyMesh2D.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            A string representation of the PolyMesh2D.
        
        Raises
        ------
        None
        
        Examples
        --------
        >>> print(PolyMesh2D())
        vcfempy.meshgen.PolyMesh2D
        Version = 0.0.1
        Number of Vertices = 0
        Number of Boundary Vertices = 0
        Number of Boundary Edges = 0
        Number of Nodes = 0
        Number of Points = 0
        Number of Elements = 0
        Number of Element Edges = 0
        <BLANKLINE>
        <BLANKLINE>
        
        >>> print(PolyMesh2D([[0,0], [0,1], [1,1], [1,0]]))
        vcfempy.meshgen.PolyMesh2D
        Version = 0.0.1
        Number of Vertices = 4
        Number of Boundary Vertices = 0
        Number of Boundary Edges = 0
        Number of Nodes = 0
        Number of Points = 0
        Number of Elements = 0
        Number of Element Edges = 0
        <BLANKLINE>
        <BLANKLINE>
        
        >>> m = PolyMesh2D([[0,0], [0,1], [1,1], [1,0]])
        >>> m.set_verbose_printing(True)
        >>> print(m)
        vcfempy.meshgen.PolyMesh2D
        Version = 0.0.1
        Number of Vertices = 4
        Number of Boundary Vertices = 0
        Number of Boundary Edges = 0
        Number of Nodes = 0
        Number of Points = 0
        Number of Elements = 0
        Number of Element Edges = 0
        <BLANKLINE>
        vertices
        [[0 0]
         [0 1]
         [1 1]
         [1 0]]
        <BLANKLINE>
        <BLANKLINE>
        
        """
        
        # print header indicating type Mesh and basic information
        mesh_string = 'vcfempy.meshgen.PolyMesh2D\n'
        mesh_string += 'Version = {!s}\n'.format(ver)
        mesh_string += 'Number of Vertices = {!s}\n'.format(self.num_vertices)
        mesh_string += 'Number of Boundary Vertices = {!s}\n'.format(self.num_boundary_vertices)
        mesh_string += 'Number of Boundary Edges = {!s}\n'.format(self.num_boundary_edges)
        mesh_string += 'Number of Nodes = {!s}\n'.format(self.num_nodes)
        mesh_string += 'Number of Points = {!s}\n'.format(self.num_points)
        mesh_string += 'Number of Elements = {!s}\n'.format(self.num_elements)
        mesh_string += 'Number of Element Edges = {!s}\n\n'.format(self.num_element_edges)
        
        if not self.verbose_printing:
            return mesh_string
        
        # print vertices
        if self.num_vertices:
            mesh_string += 'vertices\n'
            mesh_string += '{!s}\n\n'.format(self.vertices)
        
        # print boundary_vertices
        if len(self.boundary_vertices):
            mesh_string += 'boundary_vertices\n'
            mesh_string += '{!s}\n\n'.format(self.boundary_vertices)
        
        # print boundary_edges
        if len(self.boundary_edges):
            mesh_string += 'boundary_edges\n'
            mesh_string += '{!s}\n\n'.format(self.boundary_edges)
            
        # print nodes
        if self.num_nodes:
            mesh_string += 'nodes\n'
            mesh_string += '{!s}\n\n'.format(self.nodes)
            
        # print points
        if self.num_points:
            mesh_string += 'points\n'
            mesh_string += '{!s}\n\n'.format(self.points)
            
        # print elements
        if self.num_elements:
            mesh_string += 'elements\n'
            mesh_string += '{!s}\n\n'.format(self.elements)
            
        # print element edges
        if self.num_elements:
            mesh_string += 'element edges\n'
            mesh_string += '{!s}\n\n'.format(self.element_edges)
        
        return mesh_string
    
    
    def add_vertices(self, vertices):
        """Add vertices to PolyMesh2D.
        
        Parameters
        ----------
        vertices : list of (int | float) | list of lists of (int | float) | numpy.ndarray, shape = (*,2)
        
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            type(vertices) not in [NoneType, list, numpy.ndarray]
            if type(vertices[0]) in [int, float, numpy.int32, numpy.float64]:
                type(vertices[k]) not in [int, float, numpy.int32, numpy.float64]
            if type(vertices[0]) in [list, numpy.ndarray]:
                type(vertices[k]) not in [list, numpy.ndarray]
            if type(vertices[k]) in [list, numpy.ndarray]:
                type(vertices[k][j]) not in [int, float, numpy.int32, numpy.float64]
        ValueError
            if type(vertices[0]) in [int, float, numpy.int32, numpy.float64]:
                len(vertices) != 2
            if type(vertices[k]) in [list, numpy.ndarray]:
                len(vertices[k]) != 2
            if type(vertices) is numpy.ndarray:
                len(vertices.shape) != 2
                
        
        Examples
        --------
        
        """
        
        # basic type check of vertices
        if type(vertices) not in [type(None), list, np.ndarray]:
            raise TypeError('type(vertices) not in [NoneType, list, numpy.ndarray]')
                
        # catch case no vertices given
        # either as None or empty list or empty numpy.ndarray
        # in all cases, do nothing
        if vertices is None \
            or (type(vertices) is list and len(vertices) == 0) \
            or (type(vertices) is np.ndarray and vertices.size == 0):
                
            return
            
        # vertices given as a list
        # Note: if here, we know that len(vertices) > 0
        elif type(vertices) is list:
            
            # catch case of single vertex given as list of numerics
            if type(vertices[0]) in [int, float, np.int32, np.float64]:
                
                # check for correct dimensions
                if len(vertices) != 2:
                    raise ValueError('type(vertices) is list of numeric, but len(vertices) != 2')
                    
                # check that all values are numeric
                for v in vertices:
                    if type(v) not in [int, float, np.int32, np.float64]:
                        err_str = 'type(vertices) is list with len == 2, vertices[0] is numeric, '
                        err_str += 'but type of other values in vertices not in [int, float, numpy.int32, numpy.float64]'
                        raise TypeError(err_str)
                        
                # if here, we know that we have a valid single vertex to add
                if self.vertices is None:
                    self._vertices = np.array([vertices])
                else:
                    self._vertices = np.vstack([self.vertices, vertices])
            
            # otherwise, vertices should be list of list (or list of numpy.ndarray) of numeric 
            # where each sub-list or sub-array has len == 2
            else:
            
                # check that all vertex lists have the right type and size
                for v in vertices:
                    
                    # check type of vertex
                    if type(v) not in [list, np.ndarray]:
                        raise TypeError('vertices given as list of lists, but type of all contents not in [list, numpy.ndarray]')

                    # check that each vertex contains two components
                    if len(v) != 2:
                        raise ValueError('vertices given as list of lists, but not all vertices have shape (*, 2)')

                    # check that the values in the vertex list are numeric
                    for x in v:
                        if type(x) not in [int, float, np.int32, np.float64]:
                            err_str = 'vertices given as list of lists with, but type of contents '
                            err_str += 'not in [int, float, numpy.int32, numpy.float64]'
                            raise TypeError(err_str)

                # if here, we know that we have a valid list of vertices to add
                if self.vertices is None:
                    self._vertices = np.array(vertices)
                else:
                    self._vertices = np.vstack([self.vertices, vertices])
        
        # vertices given as a numpy.ndarray
        # Note 1: we know this because of the earlier type check of vertices
        # Note 2: if here, we know that len(vertices) > 0
        else:
            
            # check shape of vertices
            # should have one or two dimensions
            if len(vertices.shape) > 2 \
                or (len(vertices.shape) == 1 and vertices.shape[0] != 2) \
                or (len(vertices.shape) == 2 and vertices.shape[1] != 2):
                raise ValueError('vertices given as numpy.ndarray, but shape is not (*, 2)')
                
            # check type of vertices array contents
            # Note: can just check vertices[0,0] since we know type(vertices) is numpy.ndarray
            #       which has uniform type
            if type(vertices.flatten()[0]) not in [np.int32, np.float64]:
                raise TypeError('vertices given as numpy.ndarray, but type of contents not in [numpy.int32, numpy.float64]')
            
            # if here, we know we have a valid numpy.ndarray of vertices to add
            if self.vertices is None:
                if len(vertices.shape) == 1:
                    self._vertices = np.array([vertices])
                else:
                    self._vertices = np.array(vertices)
            else:
                self._vertices = np.vstack([self.vertices, vertices])
                
        
    def insert_boundary_vertices(self, i, boundary_vertices):
        
        # basic type check of boundary_vertices
        if type(boundary_vertices) not in [type(None), int, np.int32, list]:
            raise TypeError('type(boundary_vertices) not in [NoneType, int, numpy.int32, list]')
            
        # catch case of single boundary vertex
        if type(boundary_vertices) in [int, np.int32]:
            
            # check value of vertex is less than number of vertices
            if boundary_vertices >= self.num_vertices:
                raise ValueError('boundary_vertices values must all be less than number of vertices')
                
            # add the vertex
            self.boundary_vertices.insert(i, int(boundary_vertices))
            
            # since a valid vertex was added
            # generate boundary edges and reset the mesh
            self.generate_boundary_edges()
            self.invalidate_mesh()
            
        # if no boundary vertices given, do nothing
        # Note: if here, we know that boundary_vertices is either None or a list
        #       (i.e. not int) because of earlier type check and first if block
        # in this case, we can also skip re-processing boundary edges and
        # the mesh, if present, is still valid
        elif boundary_vertices is None or len(boundary_vertices) == 0:
            pass
            
        # boundary_vertices is a non-empty list
        # Note: we know this because of earlier type check on boundary_vertices
        else:
            
            # check contents of boundary_vertices
            for v in boundary_vertices:
                
                # check type is integer
                if type(v) not in [int, np.int32]:
                    raise TypeError('type of boundary_vertices contents not in [int, numpy.int32]')
                    
                # check value of vertex is less than number of vertices
                if v >= self.num_vertices:
                    raise ValueError('boundary_vertices values must all be less than number of vertices')
                    
            # insert boundary vertices
            # Note: if here, we know that boundary_vertices is a valid list of ints
            for k in range(-1, -len(boundary_vertices)-1, -1):
                self.boundary_vertices.insert(i, int(boundary_vertices[k]))
                
            # since valid boundary vertices were added
            # generate boundary edges and reset the mesh
            self.generate_boundary_edges()
            self.invalidate_mesh()
    
        
    def remove_boundary_vertices(self, remove_vertices):
        if type(remove_vertices) is int:
            self.boundary_vertices.remove(remove_vertices)
        elif type(remove_vertices) is list:
            for v in remove_vertices:
                self.boundary_vertices.remove(v)
        else:
            raise TypeError('type(remove_vertices) not in [int, list]')
        self.invalidate_mesh()
        self.generate_boundary_edges()
        
    def pop_boundary_vertex(self, pop_index):
        ind = self.boundary_vertices.pop(pop_index)
        self.invalidate_mesh()
        self.generate_boundary_edges()
        return ind
    
    def generate_boundary_edges(self):
        self._boundary_edges = [[self.boundary_vertices[k], self.boundary_vertices[(k+1) % self.num_boundary_vertices]] \
                                        for k in range(self.num_boundary_vertices)]
        
    def add_material_regions(self, material_regions, materials = None):
        """ Add material regions to PolyMesh2D.
        
        Parameters
        ----------
        material_regions : list of int | list of lists of int | list of MaterialRegion2D
            Lists of vertex indices defining clockwise boundary path of each new material region
            or list of object references to defined MaterialRegion2D
        materials : Material | list of Materials
            Materials corresponding to each material region to be added
        
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            type(material_regions) not in [NoneType, list]
            type(material_regions[k]) not in [int, numpy.int32, list]
            if material_regions is list of ints:
                type(materials) is not vcfempy_material.Material
            if material_regions is list of lists of ints:
                type(materials) is not list
            if type(materials) is list (and valid):
                type(materials[k]) is not vcfempy_material.Material
        ValueError
            if material_regions is list of ints:
                material_regions[k] >= self.num_vertices
            if material_regions is list of list of ints:
                material_regions[k][j] >= self.num_vertices
                len(materials) != len(material_regions)
        
        Examples
        --------
        
        """
        
        # basic type check of material_regions
        if type(material_regions) not in [type(None), list, MaterialRegion2D]:
            raise TypeError('type(material_regions) not in [NoneType, list, vcfempy.meshgen.MaterialRegion2D]')

        # basic type check of materials
        if type(materials) not in [type(None), list, mtl.Material]:
            raise TypeError('type(materials) not in [NoneType, list, vcfempy.materials.Material]')
            
        # catch null case where there is nothing to add
        # ignore materials in this case
        # and return early, doing nothing
        if material_regions is None or (type(material_regions) is list and len(material_regions) == 0):
            return

        # catch case that a single MaterialRegion2D is given
        # if so, make material_regions a list of MaterialRegion2D
        elif type(material_regions) is MaterialRegion2D:
            material_regions = [material_regions]
        
        # material_regions is a list of int
        # Note: if here, we know that material_regions is a non-empty list
        # in this case, try to create a MaterialRegion2D and then
        # redefine material_regions as a list of MaterialRegion2D
        elif type(material_regions[0]) in [int, np.int32]:
            
            # check that materials has the right type
            if type(materials) not in [type(None), mtl.Material]:
                raise TypeError('material_regions given as list of ints, but type(materials) not in [NoneType, vcfempy.materials.Material]')

            # try to create list of new MaterialRegion2D from provided information
            # Note: here self is passed to MaterialRegion2D to set a reference to this
            #       PolyMesh2D as the parent mesh of the new MaterialRegion2D
            material_regions = [MaterialRegion2D(self, material_regions, materials)]

        # if here, we know that type(material_regions) is list
        # either because it was provided as such, or it was constructed
        # by previous if-elif block

        # if only a single material is given,
        # redefine materials as a list of vcfempy.materials.Material
        # with the same length as material_regions
        if type(materials) in [type(None), mtl.Material]:
            materials = [materials]*len(material_regions)

        # now we know that type(materials) is list
        # double check that both lists have the same length
        if len(material_regions) != len(materials):
            raise ValueError('material_regions and materials must have same length')

        # loop over material_regions, adding them to the mesh
        for mr, m in zip(material_regions, materials):

            # if current material region is a MaterialRegion2D
            # append it directly to the list of material regions
            if type(mr) is MaterialRegion2D:

                # assign m to mr, if m is not None
                # Note: this behaviour avoids overwriting
                #       non-None materials if already assigned to mr
                if m is not None:
                    mr.set_material(m)

                self.material_regions.append(mr)

            # if current material region is a list
            # try to create a new MaterialRegion2D
            # and then append it to the list of material regions
            elif type(mr) is list:

                self.material_regions.append(MaterialRegion2D(self, mr, m))

            # otherwise, mr has invalid type
            else:
                raise TypeError('material_regions given as list of list of ints or list of MaterialRegion2D, but type of some material_regions invalid')

            
        # new material regions were added
        # invalidate the mesh
        self.invalidate_mesh()
        
            
    def add_mesh_edges(self, mesh_edges):
        """ Add mesh edges to PolyMesh2D.
        
        Parameters
        ----------
        mesh_edges : list of int | list of lists of int
            Lists of vertex indices defining edges to be maintained in mesh generation
        
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
        
        # mesh_edges is a list of int
        # Note: if here, we know that mesh_edges is a non-empty list
        elif type(mesh_edges[0]) in [int, np.int32]:
            
            # check that mesh_edges has the right length
            if len(mesh_edges) != 2:
                raise ValueError('mesh_edges given as list of ints, but len(mesh_edges) != 2')
            
            # check that all items in mesh_edges are ints
            # and values are < self.num_vertices
            for v in mesh_edges:
                if type(v) not in [int, np.int32]:
                    raise TypeError('mesh_edges given as list of ints, but type of contents not all in [int, numpy.int32]')
                if v >= self.num_vertices:
                    raise ValueError('mesh_edges given as list of ints, but contains values >= num_vertices')
                    
            # if here, mesh_edges is a valid list of ints
            # append it to the list of mesh edges
            self.mesh_edges.append([int(k) for k in mesh_edges])
            
            # new mesh edge was added
            # invalidate the mesh
            self.invalidate_mesh()
            
        # mesh_edges is a list of list of ints
        else:
            
            # check that all mesh edges are lists of len == 2
            # and contain ints < self.num_vertices
            for edge in mesh_edges:
                if type(edge) is not list:
                    raise TypeError('mesh_edges given as list of lists of ints, but type of some contents is not list')
                if len(edge) != 2:
                    raise ValueError('mesh_edges given as list of lists of ints, but some edges have len != 2')
                for v in edge:
                    if type(v) not in [int, np.int32]:
                        raise TypeError('mesh_edges given as list of lists of ints, but type of some vertices not in [int, numpy.int32]')
                    if v >= self.num_vertices:
                        raise ValueError('mesh_edges given as list of lists of ints, but some vertices >= num_vertices')
                        
            # if here, mesh_edges is a valid list of list of ints
            # append each list to the list of material regions
            for edge in mesh_edges:
                self.mesh_edges.append([int(k) for k in edge])
                
            # new mesh edges were added
            # invalidate the mesh
            self.invalidate_mesh()
        
    
    def invalidate_mesh(self):
        
        self._mesh_valid = False
        self._nodes = None
        self._points = None
        self._elements = []
        self._element_neighbors = []
        self._element_edges = []
    

    def generate_mesh(self, grid_size = [10, 10], alpha_rand = 0.0):
        """ Generate polygonal mesh. """
        
        # generate seed points within boundary

        # set size of grid and degree of randomness
        # total number of points is nx*ny
        nx = grid_size[0]
        ny = grid_size[1]

        # get size parameters for grid
        Lx = max(self.vertices[self.boundary_vertices,0]) \
                    - min(self.vertices[self.boundary_vertices,0])
        Ly = max(self._vertices[self.boundary_vertices,1]) \
                    - min(self.vertices[self.boundary_vertices,1])
        dx = Lx/nx
        dy = Ly/ny
        d_scale = np.linalg.norm([dx,dy])

        # generate regular grid
        xc = np.linspace(min(self.vertices[self.boundary_vertices,0])+dx/2, \
                         max(self.vertices[self.boundary_vertices,0])-dx/2, \
                         nx)
        yc = np.linspace(min(self.vertices[self.boundary_vertices,1])+dy/2, \
                         max(self.vertices[self.boundary_vertices,1])-dy/2, \
                         ny)
        xc, yc = np.meshgrid(xc,yc)

        # shift points for hexagonal grid
        for k in range(xc.shape[0]):
            if k % 2:
                xc[k,:] += 0.25*dx
            else:
                xc[k,:] -= 0.25*dx

        # reshape grid into list of points
        self._points = np.vstack([xc.ravel(), yc.ravel()]).T

        # randomly shift seed points
        xc_shift = alpha_rand*dx*(2*np.random.random([xc.size,1]) - 1)
        yc_shift = alpha_rand*dy*(2*np.random.random([yc.size,1]) - 1)
        self.points[:,0] += xc_shift[:,0]
        self.points[:,1] += yc_shift[:,0]
        
        # remove existing points near mesh edges
        # and add reflected points along mesh edges to capture them
        # in the mesh
        for edge in self.mesh_edges:
            
            # get vertices
            e0 = self.vertices[edge[0]]
            e1 = self.vertices[edge[1]]
            
            # find points near the edge for deletion
            keep_points = np.bool_(np.ones(self.num_points))
            for j, p in enumerate(self.points):
                
                # find projection of the point onto the edge
                ee = e1-e0
                ep = p-e0
                pp = e0 + (np.dot(ep,ee) / np.dot(ee,ee)) * ee
                
                # check if point is close to the edge
                # and within the length of the edge
                d = np.linalg.norm(p-pp)
                de = np.linalg.norm(pp-e0)/np.linalg.norm(ee)
                if d < 0.2*d_scale and np.dot(pp-e0,ee) >= 0.0 and de <= 1.0:
                    keep_points[j] = False
                    
            # delete points near the edge
            self._points = self.points[keep_points]
            
            # get unit vector in direction of edge
            # and point step size
            ee_len = np.linalg.norm(ee)
            ee_hat = ee / ee_len
            nn_hat = np.array([ee_hat[1],-ee_hat[0]])
            num_points = int(np.round(ee_len / (0.5*d_scale)))
            de = ee_len / num_points
            
            # make list of points to add along edge
            # and add them to the overall point list
            new_points = []
            dp_list = np.linspace(0.5*de, ee_len-0.5*de, num_points)
            for dp in dp_list:
                
                # add points on both sides of the edge
                new_points.append(e0 + dp*ee_hat + 0.1*d_scale*nn_hat)
                new_points.append(e0 + dp*ee_hat - 0.1*d_scale*nn_hat)
                
            self._points = np.vstack([self.points, new_points])
        

        # add points to ensure boundary vertices are
        # captured in the mesh
        # Note: the added points differ depending on
        #       whether the vertex is convex or concave
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
            # Note: at convex vertex, pp_hat is inward pointing
            #       at concave vertex, pp_hat is outward pointing
            pp_hat = bbf+bbr
            if np.linalg.norm(pp_hat) > 1.e-8 * d_scale:
                pp_hat = pp_hat / np.linalg.norm(pp_hat)
            else:
                pp_hat = bp1-bm1
                pp_hat = np.array([pp_hat[1], -pp_hat[0]])
                pp_hat = pp_hat / np.linalg.norm(pp_hat)
            vv_hat = np.array([pp_hat[1],-pp_hat[0]])

            # check for concave vertex
            if np.cross(bbr, bbf) < 0:
                
                d_scale_loc = np.min([d_scale, d_bbf, d_bbr])
                
                # delete points near vertex b0
                keep_points = np.bool_(np.ones(self.num_points))
                for j, p in enumerate(self.points):
                    if np.linalg.norm(p-b0) < 0.5*d_scale_loc:
                        keep_points[j] = False
                self._points = self.points[keep_points]

                # create two new points near concave vertex
                new_points = [b0 + 0.4*d_scale_loc*vv_hat, b0 - 0.4*d_scale_loc*vv_hat]
                
                # add new points near vertex
                self._points = np.vstack([self.points, new_points])
                
            # otherwise, it is a convex vertex
            # check if adjacent edges are short
            elif d_bbf < d_scale or d_bbr < d_scale:
                
                d_scale_loc = np.min([d_scale, d_bbf, d_bbr])
                
                # delete points near vertex b0
                keep_points = np.bool_(np.ones(self.num_points))
                for j, p in enumerate(self.points):
                    if np.linalg.norm(p-b0) < 0.5*d_scale_loc:
                        keep_points[j] = False
                self._points = self.points[keep_points]
                
                # create three new points near convex vertex
                # adjacent to a short boundary edge
                new_points = [b0 + 0.4*d_scale_loc*pp_hat]
                
                # add new points near vertex
                self._points = np.vstack([self.points, new_points])
                

        # eliminate points that are outside the boundaries
        bpath = path.Path(self.vertices[self.boundary_vertices])
        in_bnd = bpath.contains_points(self.points)
        self._points = self.points[in_bnd]
                
        
        # reflect seed points about boundaries
        # this ensures a voronoi diagram with ridges along each boundary
        dmax = min([1.5*d_scale, Lx, Ly])
        reflected_points = []
        for p in self.points:
            for k, edge in enumerate(self.boundary_edges):
            
                # get previous and next edges
                prv_edge = self.boundary_edges[k-1]
                nxt_edge = self.boundary_edges[(k+1) % self.num_boundary_edges]

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
                pp = b0 + (np.dot(bp,bb) / np.dot(bb,bb)) * bb
                dp = pp-p
                d = np.linalg.norm(dp)
                
                # get outward normal of current edge
                nhat = np.array([-bb[1], bb[0]]) / np.linalg.norm(bb)

                # check distance to boundary, and direction of dp
                # only reflect points within dmax of boundary segment
                # and where dp points outward
                if d < dmax and np.dot(dp,nhat) > 0:
                    
                    # check whether vertices are convex
                    # Note: always reflect if vertices are convex
                    #       but at concave vertices only reflect if
                    #       the projected point pp is within the segment
                    db = pp-b0
                    db = np.sign(np.dot(db,bb)) * np.linalg.norm(db) / np.linalg.norm(bb)
                    if (is_cvx0 or db >= 0.0) and (is_cvx1 or db <= 1.0):
                        reflected_points.append(p + 2*dp)
                        

        reflected_points = np.array(reflected_points)
        
        # create Voronoi diagram of seed points
        all_points = np.vstack([self.points, reflected_points])
        vor = Voronoi(all_points)
        
        # get list of Voronoi regions inside the boundary
        npoint = len(self.points)
        point_region = vor.point_region[:npoint]

        # compile list of elements to keep
        # Note: this is a temporary variable, element objects will be created later
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
        self._element_neighbors = []
        self._element_edges = []
        for rp, rv in zip(vor.ridge_points, vor.ridge_vertices):

            # check if ridge contains at least one point inside the boundary
            if rp[0] < npoint or rp[1] < npoint:

                # save the ridge
                # if either ridge point was outside the boundary, change it to -1
                self.element_neighbors.append([rp[0] if rp[0] < npoint else -1, rp[1] if rp[1] < npoint else -1])

                # save the ridge vertices
                self.element_edges.append(rv)

        # convert node indices to reduced set of those kept in/on boundary
        node_dict = {n : k for k, n in enumerate(nodes_to_keep)}
        for k, e in enumerate(element_nodes):
            for j, v in enumerate(element_nodes[k]):
                element_nodes[k][j] = node_dict[element_nodes[k][j]]
        for k, e in enumerate(self.element_edges):
            for j, v in enumerate(self.element_edges[k]):
                self.element_edges[k][j] = node_dict[self.element_edges[k][j]]
                
        # determine material type of each element
        m0 = mtl.Material()
        element_materials = np.array([m0 for k, _ in enumerate(element_nodes)])
        for mr in self.material_regions:
            bpath = path.Path(self.vertices[mr.vertices,:])
            in_bnd = bpath.contains_points(self.points)
            element_materials[in_bnd] = mr.material

        # create list of elements
        self._elements = []
        for e, m in zip(element_nodes, element_materials):

            # create a new element and add it to the list of elements
            # Note: here, the first argument self initializes the element
            #       with a reference to the current mesh as its parent mesh
            self.elements.append( PolyElement2D( self, e, m ) )
        
        # set mesh valid
        self._mesh_valid = True
            
            
    def plot_boundaries(self, ax = None, line_type = '-k'):
        """ Plot out PolyMesh2D boundaries. """
        
        if ax is None:
            ax = plt.gca()
        
        for edge in self.boundary_edges:
            ax.plot(self.vertices[edge,0], self.vertices[edge,1], line_type)
        
        return ax
    
    def plot_vertices(self, ax = None, line_type = 'sk', markersize = 5.0):
        """ Plot out PolyMesh2D vertices. """
        
        if ax is None:
            ax = plt.gca()
        
        ax.plot(self.vertices[:,0], self.vertices[:,1], line_type, markersize = markersize)
        
        return ax
    
    def plot_mesh_edges(self, ax = None, line_type = '-k'):
        """ Plot out PolyMesh2D mesh edges. """
        
        if ax is None:
            ax = plt.gca()
        
        for edge in self.mesh_edges:
            ax.plot(self.vertices[edge,0], self.vertices[edge,1], line_type)
        
        return ax
    
    def plot_material_regions(self, ax = None, line_type = '-k', fill = True):
        """ Plot out PolyMesh2D material regions. """
        
        if ax is None:
            ax = plt.gca()
            
        for mr in self.material_regions:
            mr.plot(ax, line_type, fill)

        return ax
            
    
    def plot_mesh(self, ax = None, line_type = ':k', fill = True):
        """ Plot out PolyMesh2D elements. """
        
        if ax is None:
            ax = plt.gca()
            
        for e in self.elements:
            e.plot(ax, line_type, fill)
        
        for edge in self.element_edges:
            ax.plot(self.nodes[edge,0], self.nodes[edge,1], line_type)
        
        return ax

    def plot_mesh_boundaries(self, ax = None, line_type = '--b'):
        """ Plot out PolyMesh2D element edges that are on the boundaries. """

        if ax is None:
            ax = plt.gca()

        for ee, en in zip(self.element_edges, self.element_neighbors):
            if en[0] < 0 or en[1] < 0:
                ax.plot(self.nodes[ee,0], self.nodes[ee,1], line_type)

        return ax
    
    def plot_mesh_nodes(self, ax = None, line_type = 'ok', markersize = 2.0):
        """ Plot out PolyMesh2D nodes. """
        
        if ax is None:
            ax = plt.gca()
        
        ax.plot(self.nodes[:,0], self.nodes[:,1], line_type, markersize = markersize)
        
        return ax
    
    def plot_quadrature_points(self, ax = None, line_type = '+k', markersize = 1.5):
        """ Plot out PolyMesh2D quadrature points. """
        
        if ax is None:
            ax = plt.gca()
        
        for e in self.elements:
            e.plot_quadrature_points(ax, line_type, markersize)
        
        return ax

class MaterialRegion2D():
    """ A class for defining material regions and their attributes. """

    def __init__(self, mesh, vertices = None, material = None):

        self.set_mesh(mesh)

        self._vertices = []
        self.insert_vertices(0, vertices)

        self.set_material(material)

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def vertices(self):
        return self._vertices

    @property
    def mesh(self):
        return self._mesh

    def set_mesh(self, mesh):

        if type(mesh) not in [type(None), PolyMesh2D]:
            raise TypeError('type(mesh) not in [NoneType, vcfempy.meshgen.PolyMesh2D]')

        self._mesh = mesh

    @property
    def material(self):
        return self._material

    def set_material(self, material):

        if type(material) not in [type(None), mtl.Material]:
            raise TypeError('type(material) not in [NoneType, vcfempy.materials.Material]')

        self._material = material
    
    
    def insert_vertices(self, i, vertices):
        
        # basic type check of vertices
        if type(vertices) not in [type(None), int, np.int32, list]:
            raise TypeError('type(vertices) not in [NoneType, int, numpy.int32, list]')
            
        # catch case of single vertex
        if type(vertices) in [int, np.int32]:
            
            vertices = [vertices]
           
        # if vertices given as None or empty list, return early
        # Note: if here, we know that vertices is either None or a list
        #       (i.e. not int) because of earlier type check and first if block
        # in this case, we can skip re-processing the mesh since it is still valid
        elif vertices is None or len(vertices) == 0:
            return
            
        # vertices is a non-empty list
        # Note: we know this because of earlier type check on vertices
            
        # check contents of vertices
        for v in vertices:
            
            # check type is integer
            if type(v) not in [int, np.int32]:
                raise TypeError('type of vertices contents not in [int, numpy.int32]')
                
            # check value of vertex is less than number of vertices
            if v >= self.mesh.num_vertices:
                raise ValueError('vertices values must all be less than number of vertices in the parent mesh')
                
        # insert vertices
        # Note: if here, we know that vertices is a valid list of ints
        vertices.reverse()
        for k in vertices:
            self.vertices.insert(i, int(k))
            
        # since valid vertices were added
        # reset the mesh
        self.mesh.invalidate_mesh()


    def plot(self, ax = None, fill = True):

        if ax is None:
            ax = plt.gca()

        if fill:
            ax.fill(self.mesh.vertices[self.vertices,0], \
                    self.mesh.vertices[self.vertices,1], \
                    color=self.material.color)

        vlist = [self.vertices[j % self.num_vertices] for j in range(self.num_vertices+1)]
        ax.plot(self.mesh.vertices[vlist, 0], \
                self.mesh.vertices[vlist, 1], \
                line_type)
    

class PolyElement2D():
    
    def __init__(self, mesh, nodes = None, material = None):

        # initialize parent mesh
        self.set_mesh(mesh)

        # initialize nodes
        self._nodes = []
        self.insert_nodes(0, nodes)

        # initialize material
        self.set_material(material)

        # initialize geometry and quadrature attributes
        self.invalidate_properties()


    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def nodes(self):
        return self._nodes

    @property
    def mesh(self):
        return self._mesh

    def set_mesh(self, mesh):

        if type(mesh) not in [type(None), PolyMesh2D]:
            raise TypeError('type(mesh) not in [NoneType, vcfempy.meshgen.PolyMesh2D]')

        self._mesh = mesh

    @property
    def material(self):
        return self._material

    def set_material(self, material):

        if type(material) not in [type(None), mtl.Material]:
            raise TypeError('type(material) not in [NoneType, vcfempy.materials.Material]')

        self._material = material
    
    @property
    def area(self):

        if self._area is None:
            self._area = polygon_area(self.mesh.nodes[self.nodes])

        return self._area

    @property
    def centroid(self):

        if self._centroid is None:
            self._centroid, _ = polygon_centroid(self.mesh.nodes[self.nodes], self.area)

        return self._centroid

    @property
    def quad_points(self):

        if self._quad_points is None:
            self.generate_quadrature()

        return self._quad_points
    
    @property
    def quad_weights(self):

        if self._quad_weights is None:
            self.generate_quadrature()

        return self._quad_weights

    @property
    def quad_integrals(self):

        if self._quad_integrals is None:
            self.generate_quadrature()

        return self._quad_integrals


    def insert_nodes(self, i, nodes):
        
        # basic type check of nodes
        if type(nodes) not in [type(None), int, np.int32, list]:
            raise TypeError('type(nodes) not in [NoneType, int, numpy.int32, list]')
            
        # catch case of single node
        if type(nodes) in [int, np.int32]:
            
            nodes = [nodes]
           
        # if nodes given as None or empty list, return early
        # Note: if here, we know that nodes is either None or a list
        #       (i.e. not int) because of earlier type check and first if block
        # in this case, we can skip re-processing the mesh since it is still valid
        elif nodes is None or len(nodes) == 0:
            return
            
        # nodes is a non-empty list
        # Note: we know this because of earlier type check on nodes
            
        # check contents of nodes
        for v in nodes:
            
            # check type is integer
            if type(v) not in [int, np.int32]:
                raise TypeError('type of nodes contents not in [int, numpy.int32]')
                
            # check value of node is less than number of nodes in parent mesh
            if v >= self.mesh.num_nodes:
                raise ValueError('nodes values must all be less than number of nodes in the parent mesh')
                
        # insert nodes
        # Note: if here, we know that nodes is a valid list of ints
        nodes.reverse()
        for k in nodes:
            self.nodes.insert(i, int(k))

        # nodes were added, so reset element properties
        self.invalidate_properties()


    def invalidate_properties(self):
        self._area = None
        self._centroid = None
        self._quad_points = None
        self._quad_weights = None
        self._quad_integrals = None

    
    def generate_quadrature(self):
        """ Generate quadrature points and weights for a PolyElement2D. """

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
        self._quad_points = np.array([[0.,0.]])
        self._quad_weights = np.array([1.])
        self._quad_integrals = np.array([self.area])


    def _quadcon5(self):
        
        vertices = self.mesh.nodes[self.nodes] - self.centroid
        cent = np.array([0.,0.])
        area = self.area
            
        # integrate basis functions
        # f(x,y) = {1, x, y, x**2, x*y, y**2}
        
        # using subdivision of polygon into triangles
        # each triangle is integrated using Gaussian quadrature
        # as per: Cowper, G.R. 1973. Gaussian quadrature formulas for triangles,
        #             International Journal for Numerical Methods 7(3): 405-408,
        #             doi: 10.1002/nme.1620070316
        # here, use the 3-point formula  with degree of precision 2
        N = np.array([[0.66666_66666_66667, 0.16666_66666_66667, 0.16666_66666_66667], \
                      [0.16666_66666_66667, 0.66666_66666_66667, 0.16666_66666_66667], \
                      [0.16666_66666_66667, 0.16666_66666_66667, 0.66666_66666_66667]])
        w = np.array([0.33333_33333_33333, \
                      0.33333_33333_33333, \
                      0.33333_33333_33333])
        nphi = 6
        phi = np.zeros(nphi)
        
        # loop over vertices
        n = len(vertices)
        for k, v0 in enumerate(vertices):
            
            # form triangle with 2 vertices + centroid
            v1 = vertices[(k+1)%n]
            x = np.vstack([cent, v0, v1])
            
            # triangle area
            detJ = 0.5 * np.abs((x[1,0]-x[0,0])*(x[2,1]-x[0,1]) - (x[2,0]-x[0,0])*(x[1,1]-x[0,1]))
            
            # perform Gaussian integration over triangle
            for wj, Nj in zip(w,N):
                xj = Nj @ x
                phi += detJ * wj * np.array([1., xj[0], xj[1], xj[0]**2, xj[0]*xj[1], xj[1]**2])
        
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
        PHI = np.array([np.ones(nq), xq[:,0], xq[:,1], xq[:,0]**2, xq[:,0]*xq[:,1], xq[:,1]**2])

        # solve for the quadrature coefficients and normalize integration point weights
        # Note: if nq > nphi, this is a least squares solution
        wq = np.linalg.lstsq(PHI, phi, rcond=None)[0]
        wq /= np.abs(area)
            
        self._quad_points = xq
        self._quad_weights = wq
        self._quad_integrals = phi


    def _quadcon7(self):

        vertices = self.mesh.nodes[self.nodes] - self.centroid
        cent = np.array([0.,0.])
        area = self.area
            
        # integrate basis functions
        # f(x,y) = { 1, 
        #            x, y, 
        #            x**2, x * y, y**2, 
        #            x**3, x**2 * y, x * y**2, y**3, 
        #            x**4, x**3 * y, x**2 * y**2, x * y**3, y**4}
        
        # using subdivision of polygon into triangles
        # each triangle is integrated using Gaussian quadrature
        # as per: Cowper, G.R. 1973. Gaussian quadrature formulas for triangles,
        #             International Journal for Numerical Methods 7(3): 405-408,
        #             doi: 10.1002/nme.1620070316
        # here, use the 6-point formula  with degree of precision 4
        N = np.array([[0.81684_75729_80459, 0.09157_62135_09771, 0.09157_62135_09771], \
                      [0.09157_62135_09771, 0.81684_75729_80459, 0.09157_62135_09771], \
                      [0.09157_62135_09771, 0.09157_62135_09771, 0.81684_75729_80459], \
                      [0.10810_30181_68070, 0.44594_84909_15965, 0.44594_84909_15965], \
                      [0.44594_84909_15965, 0.10810_30181_68070, 0.44594_84909_15965], \
                      [0.44594_84909_15965, 0.44594_84909_15965, 0.10810_30181_68070]])
        w = np.array([0.10995_17436_55322, \
                      0.10995_17436_55322, \
                      0.10995_17436_55322, \
                      0.22338_15896_78011, \
                      0.22338_15896_78011, \
                      0.22338_15896_78011])
        nphi = 15
        phi = np.zeros(nphi)
        
        # loop over vertices
        n = len(vertices)
        for k, v0 in enumerate(vertices):
            
            # form triangle with 2 vertices + centroid
            v1 = vertices[(k+1)%n]
            x = np.vstack([cent, v0, v1])
            
            # triangle area
            detJ = 0.5 * np.abs((x[1,0]-x[0,0])*(x[2,1]-x[0,1]) - (x[2,0]-x[0,0])*(x[1,1]-x[0,1]))
            
            # perform Gaussian integration over triangle
            for wj, Nj in zip(w,N):
                xj = Nj @ x
                phi += detJ * wj * np.array([1., \
                                             xj[0], \
                                             xj[1], \
                                             xj[0]**2, \
                                             xj[0]*xj[1], \
                                             xj[1]**2, \
                                             xj[0]**3, \
                                             xj[0]**2 * xj[1], \
                                             xj[0] * xj[1]**2, \
                                             xj[1]**3, \
                                             xj[0]**4, \
                                             xj[0]**3 * xj[1], \
                                             xj[0]**2 * xj[1]**2, \
                                             xj[0] * xj[1]**3, \
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
        PHI = np.array([np.ones(nq), \
                        xq[:,0], \
                        xq[:,1], \
                        xq[:,0]**2, \
                        xq[:,0]*xq[:,1], \
                        xq[:,1]**2, \
                        xq[:,0]**3, \
                        xq[:,0]**2 * xq[:,1], \
                        xq[:,0] * xq[:,1]**2, \
                        xq[:,1]**3, \
                        xq[:,0]**4, \
                        xq[:,0]**3 * xq[:,1], \
                        xq[:,0]**2 * xq[:,1]**2, \
                        xq[:,0] * xq[:,1]**3, \
                        xq[:,1]**4])

        # solve for the quadrature coefficients and normalize integration point weights
        # Note: if nq > nphi, this is a least squares solution
        wq = np.linalg.lstsq(PHI, phi, rcond=None)[0]
        wq /= np.abs(area)

        self._quad_points = xq
        self._quad_weights = wq
        self._quad_integrals = phi


    def _quadcon10(self):
        
        vertices = self.mesh.nodes[self.nodes] - self.centroid
        cent = np.array([0.,0.])
        area = self.area

        # integrate basis functions
        # f(x,y) = { 1, 
        #            x, y, 
        #            x**2, x * y, y**2, 
        #            x**3, x**2 * y, x * y**2, y**3, 
        #            x**4, x**3 * y, x**2 * y**2, x * y**3, y**4, 
        #            x**5, x**4 * y, x**3 * y**2, x**2 * y**3, x * y**4, y**5, 
        #            x**6, x**5 * y, x**4 * y**2, x**3 * y**3, x**2 * y**4, x * y**5, y**6}
        
        # using subdivision of polygon into triangles
        # each triangle is integrated using Gaussian quadrature
        # as per: Cowper, G.R. 1973. Gaussian quadrature formulas for triangles,
        #             International Journal for Numerical Methods 7(3): 405-408,
        #             doi: 10.1002/nme.1620070316
        # here, use the 12-point formula  with degree of precision 6
        N = np.array([[0.87382_19710_16996, 0.06308_90144_91502, 0.06308_90144_91502], \
                      [0.06308_90144_91502, 0.87382_19710_16996, 0.06308_90144_91502], \
                      [0.06308_90144_91502, 0.06308_90144_91502, 0.87382_19710_16996], \
                      [0.50142_65096_58179, 0.24928_67451_70911, 0.24928_67451_70911], \
                      [0.24928_67451_70911, 0.50142_65096_58179, 0.24928_67451_70911], \
                      [0.24928_67451_70911, 0.24928_67451_70911, 0.50142_65096_58179], \
                      [0.63650_24991_21399, 0.31035_24510_33785, 0.05314_50498_44816], \
                      [0.63650_24991_21399, 0.05314_50498_44816, 0.31035_24510_33785], \
                      [0.31035_24510_33785, 0.63650_24991_21399, 0.05314_50498_44816], \
                      [0.31035_24510_33785, 0.05314_50498_44816, 0.63650_24991_21399], \
                      [0.05314_50498_44816, 0.63650_24991_21399, 0.31035_24510_33785], \
                      [0.05314_50498_44816, 0.31035_24510_33785, 0.63650_24991_21399]])
        w = np.array([0.05084_49063_70207, \
                      0.05084_49063_70207, \
                      0.05084_49063_70207, \
                      0.11678_62757_26379, \
                      0.11678_62757_26379, \
                      0.11678_62757_26379, \
                      0.08285_10756_18374, \
                      0.08285_10756_18374, \
                      0.08285_10756_18374, \
                      0.08285_10756_18374, \
                      0.08285_10756_18374, \
                      0.08285_10756_18374])
        nphi = 28
        phi = np.zeros(nphi)
        
        # loop over vertices
        n = len(vertices)
        for k, v0 in enumerate(vertices):
            
            # form triangle with 2 vertices + centroid
            v1 = vertices[(k+1)%n]
            x = np.vstack([cent, v0, v1])
            
            # triangle area
            detJ = 0.5 * np.abs((x[1,0]-x[0,0])*(x[2,1]-x[0,1]) - (x[2,0]-x[0,0])*(x[1,1]-x[0,1]))
            
            # perform Gaussian integration over triangle
            for wj, Nj in zip(w,N):
                xj = Nj @ x
                phi += detJ * wj * np.array([1., \
                                             xj[0], \
                                             xj[1], \
                                             xj[0]**2, \
                                             xj[0]*xj[1], \
                                             xj[1]**2, \
                                             xj[0]**3, \
                                             xj[0]**2 * xj[1], \
                                             xj[0] * xj[1]**2, \
                                             xj[1]**3, \
                                             xj[0]**4, \
                                             xj[0]**3 * xj[1], \
                                             xj[0]**2 * xj[1]**2, \
                                             xj[0] * xj[1]**3, \
                                             xj[1]**4, \
                                             xj[0]**5, \
                                             xj[0]**4 * xj[1], \
                                             xj[0]**3 * xj[1]**2, \
                                             xj[0]**2 * xj[1]**3, \
                                             xj[0] * xj[1]**4, \
                                             xj[1]**5, \
                                             xj[0]**6, \
                                             xj[0]**5 * xj[1], \
                                             xj[0]**4 * xj[1]**2, \
                                             xj[0]**3 * xj[1]**3, \
                                             xj[0]**2 * xj[1]**4, \
                                             xj[0] * xj[1]**5, \
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
        Ntri = np.array([[0.6, 0.2, 0.2], \
                         [0.2, 0.6, 0.2], \
                         [0.2, 0.2, 0.6]])
        tri_xq0 = []
        nq0 = len(xq0)
        for k, x0 in enumerate(xq0):
            x1 = xq0[(k+1) % nq0]
            x = np.vstack([x0,x1,cent])
            for Nj in Ntri:
                tri_xq0.append(Nj @ x)
        xq = np.vstack([xq0, tri_xq0, cent])
        nq = len(xq)
            
        # evaluate basis functions at integration points
        PHI = np.array([np.ones(nq), \
                        xq[:,0], \
                        xq[:,1], \
                        xq[:,0]**2, \
                        xq[:,0]*xq[:,1], \
                        xq[:,1]**2, \
                        xq[:,0]**3, \
                        xq[:,0]**2 * xq[:,1], \
                        xq[:,0] * xq[:,1]**2, \
                        xq[:,1]**3, \
                        xq[:,0]**4, \
                        xq[:,0]**3 * xq[:,1], \
                        xq[:,0]**2 * xq[:,1]**2, \
                        xq[:,0] * xq[:,1]**3, \
                        xq[:,1]**4, \
                        xq[:,0]**5, \
                        xq[:,0]**4 * xq[:,1], \
                        xq[:,0]**3 * xq[:,1]**2, \
                        xq[:,0]**2 * xq[:,1]**3, \
                        xq[:,0] * xq[:,1]**4, \
                        xq[:,1]**5, \
                        xq[:,0]**6, \
                        xq[:,0]**5 * xq[:,1], \
                        xq[:,0]**4 * xq[:,1]**2, \
                        xq[:,0]**3 * xq[:,1]**3, \
                        xq[:,0]**2 * xq[:,1]**4, \
                        xq[:,0] * xq[:,1]**5, \
                        xq[:,1]**6])

        # solve for the quadrature coefficients and normalize integration point weights
        # Note: if nq > nphi, this is a least squares solution
        wq = np.linalg.lstsq(PHI, phi, rcond=None)[0]
        wq /= np.abs(area)
            
        self._quad_points = xq
        self._quad_weights = wq
        self._quad_integrals = phi
     

    def plot(self, ax = None, line_type = ':k', fill = True, borders = False):

        if ax is None:
            ax = plt.gca()

        if fill:
            ax.fill(self.mesh.nodes[self.nodes,0], \
                    self.mesh.nodes[self.nodes,1], \
                    color=self.material.color)

        if borders:
            vlist = [self.nodes[j % self.num_nodes] for j in range(self.num_nodes+1)]
            ax.plot(self.mesh.nodes[vlist, 0], \
                    self.mesh.nodes[vlist, 1], \
                    line_type)

    def plot_quadrature_points(self, ax = None, line_type = '+k', markersize = 1.5):

        if ax is None:
            ax = plt.gca()

        ax.plot( self.quad_points[:,0] + self.centroid[0], \
                 self.quad_points[:,1] + self.centroid[1], \
                 line_type, markersize = markersize)

        return ax
 
        
def polygon_area(x):
    
    n = len(x)
    area = 0.
    for k, v0 in enumerate(x):
        vm1 = x[k-1]
        vp1 = x[(k+1) % n]
        area += v0[0] * (vp1[1] - vm1[1])
    
    return 0.5*area

def polygon_centroid(x, area = None):
    
    if area is None:
        area = polygon_area(x)
    
    n = len(x)
    cent = np.zeros(2)
    for k, v0 in enumerate(x):
        v1 = x[(k+1) % n]
        d = v0[0]*v1[1] - v0[1]*v1[0]
        cent += (v0+v1) * d
    cent /= (6. * area)
        
    return cent , area


