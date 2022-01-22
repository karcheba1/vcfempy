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

    print('*** Simple rectangular domain:\n')
    
    rect_mesh = vcm.PolyMesh2D()

    # add main corner vertices
    rect_mesh.add_vertices([[0,0],[0,20.],[20,20],[20,0]])

    # insert boundary vertices
    rect_mesh.insert_boundary_vertices(0, [k for k in range(rect_mesh.num_vertices)])

    # add material types and regions
    # Note: here we test creating a MaterialRegion2D object
    #       and then adding it to the mesh
    rock = mtl.Material('xkcd:stone')
    rock_region = vcm.MaterialRegion2D(rect_mesh, [k for k in rect_mesh.boundary_vertices], rock)
    rect_mesh.add_material_regions(rock_region)

    # generate mesh and print properties
    rect_mesh.generate_mesh([16,16], 0.2)
    print(rect_mesh)

    # plot histogram of number of nodes per element
    fig = plt.figure()
    ax = plt.gca()
    num_nodes_per_element = [e.num_nodes for e in rect_mesh.elements]
    ax.hist(num_nodes_per_element, bins=[3,4,5,6,7,8,9,10], align='left', rwidth=0.95, color='xkcd:gray')
    ax.set_xlabel('# nodes in element', fontsize=12, fontweight='bold')
    ax.set_ylabel('# elements', fontsize=12, fontweight='bold')
    ax.set_title('Rectangular Mesh Histogram', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    plt.savefig('rect_mesh_hist.png')

    # plot mesh
    fig = plt.figure()
    fig.set_size_inches((10,10))
    ax = rect_mesh.plot_mesh()
    rect_mesh.plot_boundaries()
    rect_mesh.plot_mesh_edges()
    rect_mesh.plot_mesh_boundaries()
    rect_mesh.plot_vertices()
    rect_mesh.plot_mesh_nodes()
    rect_mesh.plot_quadrature_points()

    ax.set_xlabel('x [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('y [m]', fontsize=12, fontweight='bold')
    ax.set_title('Rectangular Mesh', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    ax.axis('equal')

    plt.savefig('rect_mesh.png')


    # test quadrature
    int_test = np.zeros(6)
    int_exp = np.array([400., 4000., 4000., 20.*8000./3, 20.*8000./3, 40000.])
    for e in rect_mesh.elements:
        xq = e.quad_points
        wq = e.quad_weights
        cent = e.centroid
        area = e.area

        int_test[0] += np.abs(area) * np.sum(wq)
        for xq_k, wk in zip(xq, wq):
            int_test[1] += np.abs(area) * wk * (xq_k[0] + cent[0])
            int_test[2] += np.abs(area) * wk * (xq_k[1] + cent[1])
            int_test[3] += np.abs(area) * wk * (xq_k[0] + cent[0])**2
            int_test[4] += np.abs(area) * wk * (xq_k[1] + cent[1])**2
            int_test[5] += np.abs(area) * wk * (xq_k[0]*xq_k[1] + xq_k[0]*cent[1] + xq_k[1]*cent[0] + cent[0]*cent[1])

    print('Tst Ints: ', int_test)
    print('Exp Ints: ', int_exp)
    print('Int Errs: ', (int_test - int_exp)/int_exp)
    print('\n')
 


def dam_mesh():
    """ An example with a mesh for a dam with multiple materials.
    """

    print('*** Dam with multiple material regions:\n')


def tunnel_mesh():
    """ An example with a mesh for a tunnel with a concave boundary.
    """

    print('*** Symmetric tunnel with concave boundary:\n')


if __name__ == '__main__':
    print('Running all examples:\n\n')

    print('========================================')
    print('Mesh and quadrature generation:')
    print('========================================\n')

    rectangular_mesh()
    dam_mesh()
    tunnel_mesh()

