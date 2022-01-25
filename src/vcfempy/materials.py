"""A module for materials in the Voronoi Cell Finite Element Method (VCFEM).

Classes
-------
Material
    A class for materials and their properties in the VCFEM

See Also
--------
vcfempy.meshgen.py
    A module for generating meshes for the VCFEM
vcfempy.flow.py
    A module for seepage/flow analysis using the VCFEM
/src/examples.py
    A script with demonstrative examples of ``vcfempy`` usage

Notes
-----
This module is part of the ``vcfempy`` package. Internally, it is commonly
imported as
``import vcfempy.materials as mtl``
and is a base level module imported by other modules in ``vcfempy``.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_col

class Material():
    """A class for materials and their properties in the VCFEM.
    
    Parameters
    ----------
    color : color_like, optional
        The material color for plotting, a valid matplotlib color value
        If not provided or None, self.color initialized to random RGBA with A=0.3

    Attributes
    ----------
    color : color_like
        The material color for plotting, a valid matplotlib color value
        
    Examples
    --------
    >>> m = Material((0.1, 0.5, 0.7)); print(m.color)
    (0.1, 0.5, 0.7)

    >>> m = Material((0.1, 0.2, 0.3, 0.6)); print(m.color)
    (0.1, 0.2, 0.3, 0.6)

    >>> m = Material('xkcd:sand'); print(m.color)
    xkcd:sand

    >>> import numpy as np; np.random.seed(0); \
        m = Material(); print(m.color)
    (0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.3)
    """
    
    def __init__(self, color = None):
        """ Initialization method for vcfempy.Material """
        
        # check if color not provided
        if color is None:
            color = (np.random.random(), np.random.random(), np.random.random(), 0.3)
            
        # initialize color value
        self.color = color
    
    
    @property
    def color(self):
        """ Getter for Material color 

        Parameters
        ----------
        c : color_like
            New material color, a valid matplotlib color value

        Returns
        -------
        color_like
            A matplotlib color value

        Raises
        ------
        ValueError
            `c` is not a valid matplotlib color

        Examples
        --------
        >>> m = Material(); m.color = (0.1, 0.2, 0.3); print(m.color)
        (0.1, 0.2, 0.3)

        >>> m = Material(); m.color = (0.1, 0.2, 0.3, 0.9); print(m.color)
        (0.1, 0.2, 0.3, 0.9)

        >>> m = Material(); m.color = 'xkcd:sand'; print(m.color)
        xkcd:sand

        >>> m = Material(); m.color = (1.2, 0.2, 0.3); print(m.color)
        Traceback (most recent call last):
        ...
        ValueError: (1.2, 0.2, 0.3) is not a valid matplotlib.colors color

        >>> m = Material(); m.color = 'xkcd:blech'; print(m.color)
        Traceback (most recent call last):
        ...
        ValueError: xkcd:blech is not a valid matplotlib.colors color

        >>> m = Material(); m.color = None; print(m.color)
        Traceback (most recent call last):
        ...
        ValueError: None is not a valid matplotlib.colors color

        """
        return self._color
    
    @color.setter
    def color(self, c):
        """ Setter for Material color """
        if not mpl_col.is_color_like(c):
            raise ValueError('{} is not a valid matplotlib.colors color'.format(c))
        self._color = c
        
            
