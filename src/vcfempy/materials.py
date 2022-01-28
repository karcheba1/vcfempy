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
    name
        A descriptive name for the material
        Will be cast to str regardless of type
    **kwargs
        Any of the Attributes (e.g. `color`, `bulk_modulus`) can be
        passed as keyword arguments when creating a `Material` object

    Attributes
    ----------
    name : str
    color : color_like, optional
        If not provided in initialization, will be set to random RGB
    hydraulic_conductivity : float, optional
    specific_storage : float, optional
    thermal_conductivity : float, optional
    specific_heat : float, optional
    electrical_conductivity : float, optional
    bulk_modulus : float, optional
    shear_modulus : float, optional
    saturated_density : float, optional
    porosity : float, optional

    Other Parameters
    ----------------
    lame_parameter : float
    young_modulus : float
    poisson_ratio : float
    void_ratio : float

    Raises
    ------
    TypeError
        If `name` is not provided

    Notes
    -----
    The attributes listed under Other Parameters cannot be set.
    They are calculated based on the values of other Attributes.
    See the Notes in the docstring for each Other Parameter for
    specific calculation method.

    Examples
    --------
    >>> import numpy as np
    >>> import vcfempy.materials

    # initializing a blank Material (color will be random)
    >>> np.random.seed(0) # optional, for unit testing
    >>> m = vcfempy.materials.Material('random color material')
    >>> print(m.name)
    random color material
    >>> print(m.color)
    (0.5488135039273248, 0.7151893663724195, 0.6027633760716439)
    >>> print(m.hydraulic_conductivity) # attribute not initialized
    None

    # initializing a Material with an RGB color provided
    >>> m = vcfempy.materials.Material('RGB color material',\
                                       color=(0.1, 0.5, 0.7))
    >>> print(m.name)
    RGB color material
    >>> print(m.color)
    (0.1, 0.5, 0.7)

    # initializing a Material with a color_like str provided
    >>> m = vcfempy.materials.Material('xkcd str color material',\
                                       color='xkcd:sand')
    >>> print(m.name)
    xkcd str color material
    >>> print(m.color)
    xkcd:sand

    # initializing a Material with a material property provided
    # color will be random
    >>> np.random.seed(0) # optional, for unit testing
    >>> m = vcfempy.materials.Material('flow property material',\
                                       hydraulic_conductivity=5.e-5)
    >>> print(m.name)
    flow property material
    >>> print(m.color)
    (0.5488135039273248, 0.7151893663724195, 0.6027633760716439)
    >>> print(m.hydraulic_conductivity)
    5e-05

    # trying to initialize a material without a name
    >>> m = vcfempy.materials.Material()
    Traceback (most recent call last):
    ...
    TypeError: __init__() missing 1 required positional argument: 'name'
    """

    def __init__(self, name, **kwargs):
        """Initialization method for `vcfempy.materials.Material` object."""
        self.name = name

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
        self.saturated_density = kwargs.get('saturated_density', None)
        self.porosity = kwargs.get('porosity', None)

    @property
    def name(self):
        """A descriptive name for the Material

        Parameters
        ----------
        name
            New material name, will be cast to str

        Returns
        -------
        str
            The name of the Material

        Raises
        ------
        None

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('sand')
        >>> print(m.name)
        sand

        >>> m.name = 'clay'
        >>> print(m.name)
        clay

        >>> m.name = 1
        >>> print(m.name)
        1

        >>> material_count = 8
        >>> material_count += 1
        >>> m.name = f'Material {material_count}'
        >>> print(m.name)
        Material 9
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

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
        >>> m = vcfempy.materials.Material('m')
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
        ValueError
            If `hyd_cond` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
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
        ValueError
            If `spc_str` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
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
        ValueError
            If `thm_cond` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
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
        ValueError
            If `spc_heat` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
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
        ValueError
            If `elc_cond` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
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
        ValueError
            If `blk_mod` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
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
        ValueError
            If `shr_mod` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
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

    @property
    def lame_parameter(self):
        """The first Lamé parameter of the material.

        Returns
        -------
        float
            The first Lamé parameter of the material

        Notes
        ------
        This attribute cannot be set. It is calculated from the values of
        `bulk_modulus`, *K*, and `shear_modulus`, *G*.
        .. math:: \lambda = K - 2G/3

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
        >>> m.bulk_modulus = 4.2e5
        >>> m.shear_modulus = 6.9e4
        >>> print(m.lame_parameter)
        374000.0
        """
        return self.bulk_modulus - 2*self.shear_modulus/3

    @property
    def young_modulus(self):
        """The Young's modulus of the material.

        Returns
        -------
        float
            The Young's modulus of the material

        Notes
        ------
        This attribute cannot be set. It is calculated from the values of
        `bulk_modulus`, *K*, and `shear_modulus`, *G*.
        .. math:: E = 9KG / (3K + G)

        Examples
        --------
        >>> import numpy as np
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
        >>> m.bulk_modulus = 4.2e5
        >>> m.shear_modulus = 6.9e4
        >>> print(np.around(m.young_modulus, 1))
        196252.8
        """
        return (9*self.bulk_modulus*self.shear_modulus
                / (3*self.bulk_modulus + self.shear_modulus))

    @property
    def poisson_ratio(self):
        """The Poisson's ratio of the material.

        Returns
        -------
        float
            The Poisson's ratio of the material

        Notes
        ------
        This attribute cannot be set. It is calculated from the values of
        `bulk_modulus`, *K*, and `shear_modulus`, *G*.
        .. math:: \nu = (3K - 2G) / (2(3K + G))

        Examples
        --------
        >>> import numpy as np
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
        >>> m.bulk_modulus = 4.2e5
        >>> m.shear_modulus = 6.9e4
        >>> print(np.around(m.poisson_ratio, 4))
        0.4221
        """
        return 0.5*((3*self.bulk_modulus - 2*self.shear_modulus)
                    / (3*self.bulk_modulus + self.shear_modulus))

    @property
    def saturated_density(self):
        """The saturated density of the material.

        Parameters
        ----------
        sat_dns : float
            The value of the saturated density

        Returns
        -------
        float
            The saturated density of the material

        Raises
        ------
        ValueError
            If `sat_dns` cannot be cast to ``float``

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
        >>> print(m.saturated_density)
        None
        >>> m.saturated_density = 1950
        >>> print(m.saturated_density)
        1950.0

        >>> m.saturated_density = '1.95e3'
        >>> print(m.saturated_density)
        1950.0

        >>> m.saturated_density = 'forty two'
        Traceback (most recent call last):
        ...
        ValueError: could not convert string to float: 'forty two'
        """
        return self._sat_dns

    @saturated_density.setter
    def saturated_density(self, sat_dns):
        self._sat_dns = float(sat_dns) if sat_dns is not None else None

    @property
    def porosity(self):
        """The porosity of the material.

        Parameters
        ----------
        por : float
            The value of the porosity

        Returns
        -------
        float
            The porosity of the material

        Raises
        ------
        ValueError
            If `por` cannot be cast to ``float``
            If `por` < 0.0 or `por` >= 1.0

        Examples
        --------
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
        >>> print(m.porosity)
        None
        >>> m.porosity = 0.42
        >>> print(m.porosity)
        0.42

        >>> m.porosity = '6.9e-1'
        >>> print(m.porosity)
        0.69

        >>> m.porosity = 'forty two'
        Traceback (most recent call last):
        ...
        ValueError: could not convert string to float: 'forty two'

        >>> m.porosity = 1.2
        Traceback (most recent call last):
        ...
        ValueError: porosity of 1.2 is not valid, 0.0 <= porosity < 1.0

        >>> m.porosity = -0.1
        Traceback (most recent call last):
        ...
        ValueError: porosity of -0.1 is not valid, 0.0 <= porosity < 1.0
        """
        return self._por

    @porosity.setter
    def porosity(self, por):
        if por is None:
            self._por = None
        else:
            por = float(por)
            if por < 0.0 or por >= 1.0:
                raise ValueError(f'porosity of {por} is not valid, '
                                 + '0.0 <= porosity < 1.0')
            self._por = por

    @property
    def void_ratio(self):
        """The void ratio the material.

        Returns
        -------
        float
            The void ratio of the material

        Notes
        ------
        This attribute cannot be set. It is calculated from the value of
        `porosity`.
        .. math:: e = n / (1 - n)

        Examples
        --------
        >>> import numpy as np
        >>> import vcfempy.materials
        >>> m = vcfempy.materials.Material('m')
        >>> m.porosity = 0.42
        >>> print(np.around(m.void_ratio, 4))
        0.7241
        """
        return self.porosity / (1 - self.porosity)
