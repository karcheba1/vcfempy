""" 
A module containing scripts that run demonstrative examples of the 
Voronoi Cell Finite Element Method package (vcfempy). This can be
run as a standalone script since it has the __name__ == '__main__' idiom.

Uses
----
numpy
matplotlib.pyplot
vcfempy.materials
vcfempy.meshgen

Functions
---------
rectangular_mesh()
    An example demonstrating mesh generation for a simple rectangular domain with a single material
dam_mesh()
    An example demonstrating mesh generation for a polygonal domain with multiple materials
    and mesh edges between the materials. Demonstrates "soft" (no mesh_edge) vs. "hard" edges
    (using a mesh_edge).
tunnel_mesh()
    An example demonstrating mesh generation for a symmetric analysis of a tunnel, which has a
    concave domain boundary. Also demonstrates mesh_edges within a material.

"""

import numpy as np
import matplotlib.pyplot as plt

import vcfempy.materials as mtl
import vcfempy.meshgen as msh

def rectangular_mesh():
    """ 
    An example demonstrating mesh generation for a simple rectangular domain with a single material
    """

    print('*** Simple rectangular domain:\n')
    
    # initialize the mesh object
    rect_mesh = msh.PolyMesh2D()

    # add main corner vertices
    rect_mesh.add_vertices([[0,0],[0,20],[0,40.],[20,40],[20,20],[20,0]])

    # insert boundary vertices
    # here, use a list comprehension to add all vertices in clockwise order
    rect_mesh.insert_boundary_vertices(0, [k for k in range(rect_mesh.num_vertices)])

    # add material types and regions
    # Note: here we create a MaterialRegion2D object and then add it to the mesh
    rock = mtl.Material(color='xkcd:stone')
    rock_region = msh.MaterialRegion2D(rect_mesh, [k for k in rect_mesh.boundary_vertices], rock)
    rect_mesh.add_material_regions(rock_region)

    # generate mesh and print properties
    # Note: here [16,32] is the grid size for mesh seed points 
    # and 0.2 is the degree of random shifting
    rect_mesh.generate_mesh([8,16], 0.2)
    print(rect_mesh)

    # plot histogram of number of nodes per element
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(rect_mesh.num_nodes_per_element, bins=[3,4,5,6,7,8,9,10], align='left', rwidth=0.95, color='xkcd:gray')
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
    # Note: Here we test integration of constant, linear, and quadratic functions
    int_test = np.zeros(6)
    int_exp = np.array([800., 8_000., 16_000., 320_000./3, 1_280_000./3, 160_000.])
    for xq, wq, cent, area in zip(  rect_mesh.element_quad_points, \
                                    rect_mesh.element_quad_weights, \
                                    rect_mesh.element_centroids, \
                                    rect_mesh.element_areas ):

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
    """ 
    An example demonstrating mesh generation for a polygonal domain with multiple materials
    and mesh edges between the materials. Demonstrates "soft" (no mesh_edge) vs. "hard" edges
    (using a mesh_edge).
    """

    print('*** Dam with multiple material regions:\n')
    
    # initialize the mesh object
    dam_mesh = msh.PolyMesh2D()

    # add boundary vertices
    # Note: here we show that vertices can be passed as single coordinate pairs
    #       or as lists of coordinate pairs
    #       numpy arrays can also be used
    dam_mesh.add_vertices([[0,0],[88.5,65],[92.5,65],[180,0]])
    dam_mesh.add_vertices([92.5,0])
    dam_mesh.add_vertices([45,0])
    dam_mesh.add_vertices([55,30])

    # add outer boundary vertices
    dam_mesh.insert_boundary_vertices(0, [0,6,1,2,3])

    # create two different material types
    # they are initialized with colors given as valid matplotlib color strings
    gravel = mtl.Material(color='xkcd:stone')
    clay = mtl.Material(color='xkcd:clay')

    # add material regions
    # Note: here we test three different ways to pass input to add_material_regions
    #       list of lists of vertices and list of material types
    #       list of vertices and single material type
    #       MaterialRegion2D object
    dam_mesh.add_material_regions([[0,6,1,5]],[gravel])
    dam_mesh.add_material_regions([2,3,4], gravel)
    clay_region = msh.MaterialRegion2D(dam_mesh, [1,2,4,5], clay)
    dam_mesh.add_material_regions(clay_region)

    # add edges to be preserved in mesh generation
    # Note: the left edge of the clay region will be a "soft" edge
    #       and the right edge will be a "hard" edge
    dam_mesh.add_mesh_edges([2,4])

    # generate the mesh and print basic mesh properties
    dam_mesh.generate_mesh([44,16], 0.2)
    print(dam_mesh)

    # plot histogram of number of nodes per element
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(dam_mesh.num_nodes_per_element, bins=[3,4,5,6,7,8,9,10], align='left', rwidth=0.95, color='xkcd:gray')
    ax.set_xlabel('# nodes in element', fontsize=12, fontweight='bold')
    ax.set_ylabel('# elements', fontsize=12, fontweight='bold')
    ax.set_title('Dam Mesh Histogram', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    plt.savefig('dam_mesh_hist.png')

    # plot mesh
    fig = plt.figure()
    fig.set_size_inches((10,10))
    ax = dam_mesh.plot_mesh()
    dam_mesh.plot_boundaries()
    dam_mesh.plot_mesh_edges()
    dam_mesh.plot_mesh_boundaries()

    ax.set_xlabel('x [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('y [m]', fontsize=12, fontweight='bold')
    ax.set_title('Dam Mesh', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    ax.axis('equal')

    plt.savefig('dam_mesh.png')

    # test area using generated quadrature
    int_test = np.zeros(1)
    int_exp = np.array([0.5*55*30 + 0.5*(88.5-55)*(30+65) + (92.5-88.5)*65 + 0.5*65*(180-92.5)])
    for e in dam_mesh.elements:
        wq = e.quad_weights
        area = e.area

        int_test[0] += np.abs(area) * np.sum(wq)

    print('Tst Ints: ', int_test)
    print('Exp Ints: ', int_exp)
    print('Int Errs: ', (int_test - int_exp)/int_exp)
    print('\n')
 

def tunnel_mesh():
    """ 
    An example demonstrating mesh generation for a symmetric analysis of a tunnel, which has a
    concave domain boundary. Also demonstrates mesh_edges within a material.
    """

    print('*** Symmetric tunnel with concave boundary:\n')
    
    # initialize the mesh object
    tunnel_mesh = msh.PolyMesh2D()
    
    # add main corners
    # Note: we also insert a vertex in the middle of a straight section of boundary
    #       these can be added to aid in adding boundary conditions
    tunnel_mesh.add_vertices([[0,20.],[20,20],[20,0],[15,0]])

    # create circular arc (concave)
    theta = np.linspace(0, 0.5*np.pi, 20)
    for t in theta:
        tunnel_mesh.add_vertices(10.*np.array([np.cos(t), np.sin(t)]))

    # add boundary vertices in clockwise order
    tunnel_mesh.insert_boundary_vertices(0, [k for k in range(tunnel_mesh.num_vertices)])

    # add material types and regions
    rock = mtl.Material(color='xkcd:greenish')
    tunnel_mesh.add_material_regions([k for k in range(tunnel_mesh.num_vertices)], rock)

    # add mesh edges
    # Note: mesh edges need not be at material region boundaries
    #       they can also be used to force element edge alignment
    #       (e.g. with joints in rock or existing planes of failure)
    nv = tunnel_mesh.num_vertices
    tunnel_mesh.add_vertices([[2.5,17.5],[10.,12.5],[12.5,15.],[17.5,2.5]])
    tunnel_mesh.add_mesh_edges([[nv,nv+1],[nv+3,nv+2]])

    # generate mesh and show properties
    tunnel_mesh.generate_mesh([20,20], 0.3)
    print(tunnel_mesh)

    # plot histogram of number of nodes per element
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(tunnel_mesh.num_nodes_per_element, bins=[3,4,5,6,7,8,9,10], align='left', rwidth=0.95, color='xkcd:gray')
    ax.set_xlabel('# nodes in element', fontsize=12, fontweight='bold')
    ax.set_ylabel('# elements', fontsize=12, fontweight='bold')
    ax.set_title('Tunnel Mesh Histogram', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    plt.savefig('tunnel_mesh_hist.png')

    # plot mesh
    fig = plt.figure()
    fig.set_size_inches((10,10))
    ax = tunnel_mesh.plot_mesh()
    tunnel_mesh.plot_boundaries()
    tunnel_mesh.plot_mesh_edges()
    tunnel_mesh.plot_mesh_boundaries()
    tunnel_mesh.plot_vertices()

    ax.set_xlabel('x [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('y [m]', fontsize=12, fontweight='bold')
    ax.set_title('Tunnel Mesh', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    ax.axis('equal')

    plt.savefig('tunnel_mesh.png')

    # test quadrature
    # Note: here we test integration of constant, linear, and quadratic functions
    int_test = np.zeros(6)
    int_exp = np.array([400. - 0.25*np.pi*10.0**2, 4000. - 1000./3, 4000. - 1000./3, \
                        20.*8000./3 - np.pi*10.**4/16, 20.*8000./3 - np.pi*10.**4/16, \
                        40000. - 0.125*10.**4])
    for e in tunnel_mesh.elements:
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
            int_test[5] += np.abs(area) * wk * (xq_k[0] + cent[0])*(xq_k[1] + cent[1])

    print('Tst Ints: ', int_test)
    print('Exp Ints: ', int_exp)
    print('Int Errs: {}%'.format( np.array2string(100*(int_test - int_exp)/int_exp, precision= 3) ))
    print('\n')
 

if __name__ == '__main__':
    """ If called as a standalone script, run all examples. """

    print('\nRunning all examples:\n')

    print('========================================')
    print('Mesh and quadrature generation:')
    print('========================================\n')

    rectangular_mesh()
    dam_mesh()
    tunnel_mesh()

