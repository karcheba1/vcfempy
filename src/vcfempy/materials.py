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
    **kwargs
        Any of the Attributes (e.g. `color`, `bulk_modulus`) can be
        passed as keyword arguments when creating a `Material` object

    Attributes
    ----------
    color : color_like
    hydraulic_conductivity : float
    specific_storage : float
    thermal_conductivity : float
    specific_heat : float
    electrical_conductivity : float
    bulk_modulus : float
    shear_modulus : float

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
        # if color is None, set to random RGB
        color = kwargs.get('color', None)
        if color is None:
            color = (np.random.random(),
                     np.random.random(),
                     np.random.random())
        self.color = color

        # initialize other material properties
        self.hydraulic_conductivity = kwargs.get('hydraulic_conductivity',
                                                 None)
        self.specific_storage = kwargs.get('specific_storage', None)
        self.thermal_conductivity = kwargs.get('thermal_conductivity',
                                               None)
        self.specific_heat = kwargs.get('specific_heat', None)
        self.electrical_conductivity = kwargs.get('electrical_conductivity',
                                                  None)
        self.bulk_modulus = kwargs.get('bulk_modulus', None)
        self.shear_modulus = kwargs.get('shear_modulus', None)

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

    @property
    def specific_storage(self):
        """The specific storage of the material.

        Parameters
        ----------
        spc_str : float
            The value of the specific storage

        Returns
        -------
        float
            The specific storage of the material

        Raises
        ------
        TypeError
            If `spc_str` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material()
        >>> print(m.specific_storage)
        None
        >>> m.specific_storage = 0.00014
        >>> print(m.specific_storage)
        0.00014

        >>> m.specific_storage = '4.2e-5'
        >>> print(m.specific_storage)
        4.2e-05

        >>> m.specific_storage = 'forty two'
        Traceback (most recent call last):
        ...
        ValueError: could not convert string to float: 'forty two'
        """
        return self._spc_str

    @specific_storage.setter
    def specific_storage(self, spc_str):
        self._spc_str = float(spc_str) if spc_str is not None else None

    @property
    def thermal_conductivity(self):
        """The thermal conductivity of the material.

        Parameters
        ----------
        thm_cond : float
            The value of the thermal conductivity

        Returns
        -------
        float
            The thermal conductivity of the material

        Raises
        ------
        TypeError
            If `thm_cond` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material()
        >>> print(m.thermal_conductivity)
        None
        >>> m.thermal_conductivity = 1.e-5
        >>> print(m.thermal_conductivity)
        1e-05

        >>> m.thermal_conductivity = 2
        >>> print(m.thermal_conductivity)
        2.0

        >>> m.thermal_conductivity = '5.e-10'
        >>> print(m.thermal_conductivity)
        5e-10

        >>> m.thermal_conductivity = 'forty two'
        Traceback (most recent call last):
        ...
        ValueError: could not convert string to float: 'forty two'
        """
        return self._thm_cond

    @thermal_conductivity.setter
    def thermal_conductivity(self, thm_cond):
        self._thm_cond = float(thm_cond) if thm_cond is not None else None

    @property
    def specific_heat(self):
        """The specific heat of the material.

        Parameters
        ----------
        spc_heat : float
            The value of the specific heat

        Returns
        -------
        float
            The specific heat of the material

        Raises
        ------
        TypeError
            If `spc_heat` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material()
        >>> print(m.specific_heat)
        None
        >>> m.specific_heat = 5000
        >>> print(m.specific_heat)
        5000.0

        >>> m.specific_heat = '4.2e5'
        >>> print(m.specific_heat)
        420000.0

        >>> m.specific_heat = 'forty two'
        Traceback (most recent call last):
        ...
        ValueError: could not convert string to float: 'forty two'
        """
        return self._spc_heat

    @specific_heat.setter
    def specific_heat(self, spc_heat):
        self._spc_heat = float(spc_heat) if spc_heat is not None else None

    @property
    def electrical_conductivity(self):
        """The electrical conductivity of the material.

        Parameters
        ----------
        elc_cond : float
            The value of the electrical conductivity

        Returns
        -------
        float
            The electrical conductivity of the material

        Raises
        ------
        TypeError
            If `elc_cond` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material()
        >>> print(m.electrical_conductivity)
        None
        >>> m.electrical_conductivity = 420
        >>> print(m.electrical_conductivity)
        420.0

        >>> m.electrical_conductivity = '5.5e2'
        >>> print(m.electrical_conductivity)
        550.0

        >>> m.electrical_conductivity = 'forty two'
        Traceback (most recent call last):
        ...
        ValueError: could not convert string to float: 'forty two'
        """
        return self._elc_cond

    @electrical_conductivity.setter
    def electrical_conductivity(self, elc_cond):
        self._elc_cond = float(elc_cond) if elc_cond is not None else None

    @property
    def bulk_modulus(self):
        """The bulk modulus of the material.

        Parameters
        ----------
        blk_mod : float
            The value of the bulk modulus

        Returns
        -------
        float
            The bulk modulus of the material

        Raises
        ------
        TypeError
            If `blk_mod` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material()
        >>> print(m.bulk_modulus)
        None
        >>> m.bulk_modulus = 25000
        >>> print(m.bulk_modulus)
        25000.0

        >>> m.bulk_modulus = '6.9e5'
        >>> print(m.bulk_modulus)
        690000.0

        >>> m.bulk_modulus = 'forty two'
        Traceback (most recent call last):
        ...
        ValueError: could not convert string to float: 'forty two'
        """
        return self._blk_mod

    @bulk_modulus.setter
    def bulk_modulus(self, blk_mod):
        self._blk_mod = float(blk_mod) if blk_mod is not None else None

    @property
    def shear_modulus(self):
        """The shear modulus of the material.

        Parameters
        ----------
        shr_mod : float
            The value of the shear modulus

        Returns
        -------
        float
            The shear modulus of the material

        Raises
        ------
        TypeError
            If `shr_mod` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material()
        >>> print(m.shear_modulus)
        None
        >>> m.shear_modulus = 69000
        >>> print(m.shear_modulus)
        69000.0

        >>> m.shear_modulus = '4.2e5'
        >>> print(m.shear_modulus)
        420000.0

        >>> m.shear_modulus = 'forty two'
        Traceback (most recent call last):
        ...
        ValueError: could not convert string to float: 'forty two'
        """
        return self._shr_mod

    @shear_modulus.setter
    def shear_modulus(self, shr_mod):
        self._shr_mod = float(shr_mod) if shr_mod is not None else None
