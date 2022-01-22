"""A module containing attributes, functions, classes and methods 
for materials in the Voronoi Cell Finite Element Method (VCFEM).

Attributes
----------
"""

# code version
ver = '0.0.1'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_col

class Material():
    """A class for Materials in the VCFEM.
    
    Properties
    ----------
    color : tuple | str
        The material color for plotting, a valid matplotlib color value
        
    Private Attributes
    ------------------
    _color : tuple | str
        The material color for plotting, a valid matplotlib color value
        
    Examples
    --------
    """
    
    def __init__(self, color = None):
        
        # initialize material color
        if color is None:
            color = (np.random.random(), np.random.random(), np.random.random(), 0.3)
            
        self.color = color
    
    
    @property
    def color(self):
        """ Getter for Material color. """
        return self._color
    
    @color.setter
    def color(self, c):
        if not mpl_col.is_color_like(c):
            raise ValueError('{} is not a valid matplotlib.colors color'.format(c))
        self._color = c
        
            