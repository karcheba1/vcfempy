""" Module for flow / seepage analysis in Voronoi Cell Finite Element Method. """

import numpy as np
import matplotlib.pyplot as plt

import vcfempy.meshgen as vcm
import vcfempy.materials as mtl

class PolyFlow2D():
    """ A class for 2D flow analysis in the VCFEM. """
    
    def __init__(self, mesh = None):

        self._mesh = None
        self.set_mesh(mesh)


    @property
    def mesh(self):
        return self._mesh

    def set_mesh(self, mesh):

        if type(mesh) not in [type(None), vcm.PolyMesh2D]:
            raise TypeError('type(mesh) not in [NoneType, vcfempy.meshgen.PolyMesh2D]')

        self._mesh = mesh



