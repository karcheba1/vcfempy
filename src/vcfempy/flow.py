"""A module for flow / seepage analysis in the Voronoi Cell Finite Element
Method (VCFEM).

"""

import vcfempy.meshgen as msh


class PolyFlow2D():
    """A class for 2D flow analysis in the VCFEM.

    Parameters
    ----------
    mesh : vcfempy.meshgen.PolyMesh2D, optional
        The parent mesh for the **PolyFlow2D** analysis.

    Returns
    -------
    `vcfempy.flow.PolyFlow2D`
        A new **PolyFlow2D** object.

    Examples
    --------
    """

    def __init__(self, mesh=None):
        self.mesh = mesh

    @property
    def mesh(self):
        """The parent mesh for the **PolyFlow2D** analysis.

        Parameters
        ----------
        mesh : vcfempy.meshgen.PolyMesh2D
            The parent mesh object

        Returns
        -------
        `None` or `vcfempy.meshgen.PolyMesh2D`
            The parent mesh. If `None`, no parent mesh is assigned.

        Raises
        ------
        TypeError
            If **mesh** is not `None` or a `vcfempy.meshgen.PolyMesh2D`
            object.

        Examples
        --------
        """
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        # basic type check of mesh
        if type(mesh) not in [type(None), msh.PolyMesh2D]:
            raise TypeError('type(mesh) not in [NoneType, '
                            + 'vcfempy.meshgen.PolyMesh2D]')
        self._mesh = mesh
