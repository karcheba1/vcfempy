""" 
A module containing scripts that run demonstrative examples of the 
Voronoi Cell Finite Element Method package (vcfempy). 
"""

import numpy as np
import matplotlib.pyplot as plt

import vcfempy.materials as mtl
import vcfempy.meshgen as vcm

def rectangular_mesh():
    """ An example with a simple rectangular mesh.
    """
    print('Simple rectangular domain:\n')


def dam_mesh():
    """ An example with a mesh for a dam with multiple materials.
    """
    print('Dam with multiple material regions:\n')


def tunnel_mesh():
    """ An example with a mesh for a tunnel with a concave boundary.
    """
    print('Symmetric tunnel with concave boundary:\n')


if __name__ == '__main__':
    print('Running all examples:\n\n')

    print('Mesh and quadrature generation:\n\n')
    rectangular_mesh()
    dam_mesh()
    tunnel_mesh()

