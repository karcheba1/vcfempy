""" A script for running unit tests on the 
    Voronoi Cell Finite Element Method package (vcfempy).
"""

import doctest

import vcfempy.materials as mtl

def main():

    print('========================================')
    print('Running vcfempy unit tests using doctest')
    print('========================================\n')

    print('Testing vcfempy.materials:')
    print(doctest.testmod(mtl))

    print('\nTesting complete.\n\n')


if __name__ == '__main__':
    main()
