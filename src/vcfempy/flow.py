""" Module for flow / seepage analysis in the Voronoi Cell Finite Element Method (VCFEM). 

See Also
--------
vcfempy.materials
    A module for materials and their properties in the VCFEM
vcfempy.meshgen
    A module for generating meshes for the VCFEM

Notes
-----
This module is part of the `vcfempy` package and is commonly imported internally as
`import vcfempy.flow as flw`

"""

import numpy as np
import matplotlib.pyplot as plt

import vcfempy.meshgen as msh
import vcfempy.materials as mtl

class PolyFlow2D():
    """ A class for 2D flow analysis in the VCFEM. 
    
    Parameters
    ----------
    mesh : vcfempy.meshgen.PolyMesh2D, optional, default = None
        The parent mesh for the flow analysis

    Attributes
    ----------
    mesh

    Returns
    -------
    None

    Raises
    ------
    None

    Examples
    --------
    """
    
    def __init__(self, mesh = None):
        """ Initialization method for PolyFlow2D """

        self.mesh = mesh


    @property
    def mesh(self):
        """ The parent mesh for the PolyFlow2D analysis

        Parameters
        ----------
        mesh : vcfempy.meshgen.PolyMesh2D

        Returns
        -------
        None | vcfempy.meshgen.PolyMesh2D
            The parent mesh. If None, no parent mesh is assigned.

        Raises
        ------
        TypeError
            If `mesh` is not None or a PolyMesh2D

        Examples
        --------
        """
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        """ Setter for the `mesh` property """

        # basic type check of mesh
        if type(mesh) not in [type(None), msh.PolyMesh2D]:
            raise TypeError('type(mesh) not in [NoneType, vcfempy.meshgen.PolyMesh2D]')

        # if type is valid, assign the mesh
        self._mesh = mesh

