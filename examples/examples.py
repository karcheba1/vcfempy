"""A module containing scripts that run demonstrative examples of the
Voronoi Cell Finite Element Method package (vcfempy). This can be
run as a standalone script since it has the __name__ == '__main__' idiom.

"""

import os
import sys

# add relative path to package, in case it is not installed
sys.path.insert(0, os.path.abspath('../src/'))


def rectangular_mesh():
    """An example demonstrating mesh generation for a simple rectangular
    domain with a single material
    """
    print('*** Simple rectangular domain:\n')

    # initialize the mesh object
    rect_mesh = msh.PolyMesh2D('Rectangular Mesh')
    # add main corner vertices
    rect_mesh.add_vertices([[0, 0], [0, 20], [0, 40.],
                           [20, 40], [20, 20], [20, 0]])
    # insert boundary vertices in cw order
    rect_mesh.insert_boundary_vertices(0, [k for k
                                           in range(rect_mesh.num_vertices)])
    # add material type and region
    # note: the MaterialRegion2D adds itself to the parent mesh by default
    clay = mtl.Material(name='clay', color='xkcd:clay',
                        has_interfaces=True, interface_width=0.05)
    sand = mtl.Material(name='sand', color='xkcd:greenish')
    msh.MaterialRegion2D(mesh=rect_mesh,
                         vertices=[0, 1, 4, 5],
                         material=clay)
    msh.MaterialRegion2D(mesh=rect_mesh,
                         vertices=[1, 2, 3, 4],
                         material=sand)
    msh.MeshEdge2D(mesh=rect_mesh, vertices=[1, 4], material=clay)
    # set mesh generation properties, generate mesh, and print mesh properties
    rect_mesh.mesh_scale = 5.0
    rect_mesh.mesh_rand = 0.2
    rect_mesh.generate_mesh()
    print(rect_mesh)
    # plot histogram of number of nodes per element
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(rect_mesh.num_nodes_per_element,
            bins=[k for k in range(3, 11)],
            align='left', rwidth=0.95, color='xkcd:gray')
    ax.set_xlabel('# nodes in element', fontsize=12, fontweight='bold')
    ax.set_ylabel('# elements', fontsize=12, fontweight='bold')
    ax.set_title('Rectangular Mesh Histogram', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    plt.savefig('rect_mesh_hist.png')
    # plot mesh
    fig = plt.figure()
    fig.set_size_inches((10, 10))
    ax = rect_mesh.plot_boundaries()
    rect_mesh.plot_vertices()
    rect_mesh.plot_mesh(ax, element_quad_points=False, show_text=True)
    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('y', fontsize=12, fontweight='bold')
    ax.set_title('Rectangular Mesh', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    ax.axis('equal')
    plt.savefig('rect_mesh.png')
    # test quadrature integration
    # constant, linear, and quadratic functions
    int_exp_0 = 800.
    int_tst_0 = 0.
    int_exp_x = 8_000.
    int_tst_x = 0.
    int_exp_y = 16_000.
    int_tst_y = 0.
    int_exp_x2 = (320_000. / 3.)
    int_tst_x2 = 0.
    int_exp_xy = 160_000.
    int_tst_xy = 0.
    int_exp_y2 = (1_280_000. / 3.)
    int_tst_y2 = 0.
    for xq, wq, cent, area in zip(rect_mesh.element_quad_points,
                                  rect_mesh.element_quad_weights,
                                  rect_mesh.element_centroids,
                                  rect_mesh.element_areas):

        int_tst_0 += area * np.sum(wq)
        for xq_k, wk in zip(xq, wq):
            int_tst_x += area * wk * (xq_k[0] + cent[0])
            int_tst_y += area * wk * (xq_k[1] + cent[1])
            int_tst_x2 += area * wk * (xq_k[0] + cent[0])**2
            int_tst_xy += area * wk * (xq_k[0] * xq_k[1]
                                       + xq_k[0] * cent[1]
                                       + xq_k[1] * cent[0]
                                       + cent[0] * cent[1])
            int_tst_y2 += area * wk * (xq_k[1] + cent[1])**2
    int_tst = np.array([int_tst_0, int_tst_x, int_tst_y,
                        int_tst_x2, int_tst_xy, int_tst_y2])
    int_exp = np.array([int_exp_0, int_exp_x, int_exp_y,
                        int_exp_x2, int_exp_xy, int_exp_y2])
    print('Tst Ints: ', int_tst)
    print('Exp Ints: ', int_exp)
    print('Int Errs: ', (int_tst - int_exp) / int_exp)
    print('\n')


def dam_mesh():
    """An example demonstrating mesh generation for a polygonal domain with
    multiple materials and mesh edges between the materials. Demonstrates
    "soft" (no mesh_edge) vs. "hard" edges (using a mesh_edge).
    """

    print('*** Dam with multiple material regions:\n')

    # initialize the mesh object
    dam_mesh = msh.PolyMesh2D(name='Dam Mesh')

    # add boundary vertices
    # Note: here we show that vertices can be passed as single coordinate pairs
    #       or as lists of coordinate pairs
    #       numpy arrays can also be used
    dam_mesh.add_vertices([[0, 0], [84, 65], [92.5, 65], [180, 0]])
    dam_mesh.add_vertices([92.5, 0])
    dam_mesh.add_vertices([45, 0])
    dam_mesh.add_vertices([55, 30])

    # add outer boundary vertices
    dam_mesh.insert_boundary_vertices(0, [0, 6, 1, 2, 3])

    # create two different material types
    # they are initialized with colors given as valid matplotlib color strings
    gravel = mtl.Material(name='gravel', color='xkcd:stone',
                          has_interfaces=False)
    clay = mtl.Material(name='clay', color='xkcd:clay',
                        has_interfaces=True, interface_width=0.08)

    # add material regions
    # Note: new material regions are added to their parent mesh by default
    msh.MaterialRegion2D(mesh=dam_mesh, vertices=[0, 6, 1, 5], material=gravel)
    msh.MaterialRegion2D(mesh=dam_mesh, vertices=[2, 3, 4], material=gravel)
    msh.MaterialRegion2D(mesh=dam_mesh, vertices=[1, 2, 4, 5], material=clay)

    # add edges to be preserved in mesh generation
    # Note: the left edge of the clay region will be a "soft" edge
    #       and the right edge will be a "hard" edge
    msh.MeshEdge2D(mesh=dam_mesh, vertices=[2, 4], material=clay)

    # generate the mesh and print basic mesh properties
    dam_mesh.mesh_scale = 4.0
    dam_mesh.mesh_rand = 0.2
    dam_mesh.generate_mesh()
    print(dam_mesh)

    # plot histogram of number of nodes per element
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(dam_mesh.num_nodes_per_element,
            bins=[k for k in range(3, 11)],
            align='left', rwidth=0.95, color='xkcd:gray')
    ax.set_xlabel('# nodes in element', fontsize=12, fontweight='bold')
    ax.set_ylabel('# elements', fontsize=12, fontweight='bold')
    ax.set_title('Dam Mesh Histogram', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    plt.savefig('dam_mesh_hist.png')

    # plot mesh
    fig = plt.figure()
    fig.set_size_inches((10, 10))
    ax = dam_mesh.plot_boundaries()
    dam_mesh.plot_vertices(ax)
    dam_mesh.plot_mesh(ax)
    ax.set_xlabel('x [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('y [m]', fontsize=12, fontweight='bold')
    ax.set_title('Dam Mesh', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    ax.axis('equal')

    plt.savefig('dam_mesh.png')

    # test area using generated quadrature
    int_test = np.zeros(1)
    int_exp = shp.Polygon(dam_mesh.vertices[dam_mesh.boundary_vertices]).area
    int_exp = np.array([int_exp])
    for e in dam_mesh.elements:
        wq = e.quad_weights
        area = e.area

        int_test[0] += np.abs(area) * np.sum(wq)

    print('Tst Ints: ', int_test)
    print('Exp Ints: ', int_exp)
    print('Int Errs: ', (int_test - int_exp)/int_exp)
    print('\n')


def tunnel_mesh():
    """An example demonstrating mesh generation for a symmetric analysis of
    a tunnel, which has a concave domain boundary. Also demonstrates
    mesh_edges within a material.
    """

    print('*** Symmetric tunnel with concave boundary:\n')

    # initialize the mesh object
    tunnel_mesh = msh.PolyMesh2D('Tunnel Mesh')

    # add main corners
    # Note: we also insert a vertex in the middle of a straight section of
    #       boundary these can be added to aid in adding boundary conditions
    tunnel_mesh.add_vertices([[0, 15.], [0, 20.], [20, 20], [20, 0], [15, 0]])

    # create circular arc (concave)
    theta = np.linspace(0, 0.5*np.pi, 20)
    for t in theta:
        tunnel_mesh.add_vertices(10.*np.array([np.cos(t), np.sin(t)]))

    # add boundary vertices in clockwise order
    tunnel_mesh.insert_boundary_vertices(0, [k for k in
                                             range(tunnel_mesh.num_vertices)])

    # add material types and regions
    rock = mtl.Material('rock', color='xkcd:greenish', has_interfaces=False)
    msh.MaterialRegion2D(mesh=tunnel_mesh,
                         vertices=tunnel_mesh.boundary_vertices,
                         material=rock)

    # add mesh edges
    # Note: mesh edges need not be at material region boundaries
    #       they can also be used to force element edge alignment
    #       (e.g. with joints in rock or existing planes of failure)
    rock_joint = mtl.Material(name='rock joint', color='xkcd:brown',
                              has_interfaces=True, interface_width=0.003)
    nv = tunnel_mesh.num_vertices
    tunnel_mesh.add_vertices([[2.5, 17.5],
                              [10., 12.5],
                              [12.,  7.5],
                              [8., 17.5],
                              [12.5, 15.],
                              [17.5, 2.5]])
    msh.MeshEdge2D(mesh=tunnel_mesh, vertices=[nv, nv+1, nv+2],
                   material=rock_joint)
    msh.MeshEdge2D(mesh=tunnel_mesh, vertices=[nv+5, nv+4, nv+3],
                   material=rock_joint)

    # generate mesh and show properties
    tunnel_mesh.mesh_scale = 0.5
    tunnel_mesh.mesh_rand = 0.3
    tunnel_mesh.generate_mesh()
    print(tunnel_mesh)

    # plot histogram of number of nodes per element
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(tunnel_mesh.num_nodes_per_element,
            bins=[k for k in range(3, 11)],
            align='left', rwidth=0.95, color='xkcd:gray')
    ax.set_xlabel('# nodes in element', fontsize=12, fontweight='bold')
    ax.set_ylabel('# elements', fontsize=12, fontweight='bold')
    ax.set_title('Tunnel Mesh Histogram', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    plt.savefig('tunnel_mesh_hist.png')

    # plot mesh
    fig = plt.figure()
    fig.set_size_inches((10, 10))
    ax = tunnel_mesh.plot_boundaries()
    tunnel_mesh.plot_vertices(ax)
    tunnel_mesh.plot_mesh(ax)
    ax.set_xlabel('x [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('y [m]', fontsize=12, fontweight='bold')
    ax.set_title('Tunnel Mesh', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    ax.axis('equal')

    plt.savefig('tunnel_mesh.png')

    # test quadrature
    # Note: here we test integration of constant,
    #       linear, and quadratic functions
    int_test = np.zeros(6)
    int_exp = np.array([400. - 0.25*np.pi*10.0**2, 4000. - 1000./3,
                        4000. - 1000./3, 20.*8000./3 - np.pi*10.**4/16,
                        20.*8000./3 - np.pi*10.**4/16, 40000. - 0.125*10.**4])
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
            int_test[5] += np.abs(area) * wk * ((xq_k[0] + cent[0])
                                                * (xq_k[1] + cent[1]))

    print('Tst Ints: ', int_test)
    print('Exp Ints: ', int_exp)
    print(f'Int Errs: {(int_test - int_exp) / int_exp}')
    print('\n')


if __name__ == '__main__':
    """ If called as a standalone script, run all examples. """
    import numpy as np
    import matplotlib.pyplot as plt
    import shapely.geometry as shp

    import vcfempy.materials as mtl
    import vcfempy.meshgen as msh

    print('\nRunning all examples:\n')

    print('========================================')
    print('Mesh and quadrature generation:')
    print('========================================\n')

    rectangular_mesh()
    dam_mesh()
    tunnel_mesh()
