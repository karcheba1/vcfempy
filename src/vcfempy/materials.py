"""A module for materials in the Voronoi Cell Finite Element Method (VCFEM).

Classes
-------
Material
    A class for materials and their properties in the VCFEM.

See Also
--------
vcfempy.meshgen.py
    A module for generating meshes for the VCFEM.
vcfempy.flow.py
    A module for seepage/flow analysis using the VCFEM.
examples.py
    A script with demonstrative examples of ``vcfempy`` usage.

Notes
-----
This module is part of the ``vcfempy`` package. Internally, it is commonly
imported as
``import vcfempy.materials as mtl``
and is a base level module imported by other modules in ``vcfempy``.

"""

import numpy as np
import matplotlib.colors as mpl_col


class Material():
    """A class for materials and their properties in the VCFEM.

    Parameters
    ----------
    color : color_like, optional
        The material color for plotting, a matplotlib color_like value
        If not provided or None, `self.color` initialized to random RGB

    Attributes
    ----------
    color : color_like
        The material color for plotting, a matplotlib color_like value

    Examples
    --------
    >>> import numpy as np
    >>> import vcfempy.materials

    # initializing a blank Material (color will be random)
    >>> np.random.seed(0) # optional, for unit testing
    >>> m = vcfempy.materials.Material()
    >>> print(m.color)
    (0.5488135039273248, 0.7151893663724195, 0.6027633760716439)
    >>> print(m.hydraulic_conductivity)
    None

    # initializing a Material with an RGB color provided
    >>> m = vcfempy.materials.Material(color=(0.1, 0.5, 0.7))
    >>> print(m.color)
    (0.1, 0.5, 0.7)

    # initializing a Material with a color_like str provided
    >>> m = vcfempy.materials.Material(color='xkcd:sand')
    >>> print(m.color)
    xkcd:sand

    # initializing a Material with a material property provided
    # color will be random
    >>> np.random.seed(0) # optional, for unit testing
    >>> m = vcfempy.materials.Material(hydraulic_conductivity=5.e-5)
    >>> print(m.color)
    (0.5488135039273248, 0.7151893663724195, 0.6027633760716439)
    >>> print(m.hydraulic_conductivity)
    5e-05
    """

    def __init__(self, **kwargs):
        """Initialization method for `vcfempy.materials.Material` object."""
        color = kwargs.get('color', None)
        # if color not provided, set to random RGB
        if color is None:
            color = (np.random.random(),
                     np.random.random(),
                     np.random.random())
        self.color = color
        self.hydraulic_conductivity = kwargs.get('hydraulic_conductivity', None)

    @property
    def color(self):
        """The plotting color of the Material

        Parameters
        ----------
        color : color_like
            New material color, a matplotlib color_like value

        Returns
        -------
        color_like
            A matplotlib color_like value

        Raises
        ------
        ValueError
            `color` is not a valid matplotlib color_like

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material()
        >>> m.color = (0.1, 0.2, 0.3)
        >>> print(m.color)
        (0.1, 0.2, 0.3)

        >>> m.color = (0.1, 0.2, 0.3, 0.9)
        >>> print(m.color)
        (0.1, 0.2, 0.3, 0.9)

        >>> m.color = 'xkcd:sand'
        >>> print(m.color)
        xkcd:sand

        >>> m.color = (1.2, 0.2, 0.3) # invalid matplotlib RGB value 1.2
        Traceback (most recent call last):
        ...
        ValueError: (1.2, 0.2, 0.3) is not a matplotlib color_like value

        >>> m.color = 'xkcd:blech' # invalid matplotlib color_like str
        Traceback (most recent call last):
        ...
        ValueError: xkcd:blech is not a matplotlib color_like value

        >>> m.color = None # invalid matplotlib color_like None
        Traceback (most recent call last):
        ...
        ValueError: None is not a matplotlib color_like value
        """
        return self._color

    @color.setter
    def color(self, color):
        if not mpl_col.is_color_like(color):
            raise ValueError(f'{color} is not a matplotlib color_like value')
        self._color = color

    @property
    def hydraulic_conductivity(self):
        """The hydraulic conductivity of the material.

        Parameters
        ----------
        hyd_cond : float
            The value of the hydraulic conductivity

        Returns
        -------
        float
            The hydraulic conductivity of the material

        Raises
        ------
        TypeError
            If `hyd_cond` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material()
        >>> m.hydraulic_conductivity = 1.e-5
        >>> print(m.hydraulic_conductivity)
        1e-05

        >>> m.hydraulic_conductivity = 5/1000
        >>> print(m.hydraulic_conductivity)
        0.005

        >>> m.hydraulic_conductivity = 2
        >>> print(m.hydraulic_conductivity)
        2.0

        >>> m.hydraulic_conductivity = '5.e-10'
        >>> print(m.hydraulic_conductivity)
        5e-10

        >>> m.hydraulic_conductivity = 'forty two'
        Traceback (most recent call last):
        ...
        ValueError: could not convert string to float: 'forty two'
        """
        return self._hyd_cond

    @hydraulic_conductivity.setter
    def hydraulic_conductivity(self, hyd_cond):
        self._hyd_cond = float(hyd_cond) if hyd_cond is not None else None
