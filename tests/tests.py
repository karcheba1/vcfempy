""" A script for running unit tests on the Voronoi Cell Finite Element
Method package (vcfempy). This can be run as a standalone script since it
has the if __name__ == '__main__': idiom.

"""

import os
import sys
import doctest

# add relative path to package, in case it is not installed
sys.path.insert(0, os.path.abspath('../src/'))


def main():
    """Run all unit tests for vcfempy modules.
    """
    import vcfempy.materials as mtl
    import vcfempy.meshgen as msh
    import vcfempy.flow as flw

    # initialize counters for tests
    total_tests = 0
    failed_tests = 0

    print('========================================')
    print('Running vcfempy unit tests using doctest')
    print('========================================\n')

    print('Testing vcfempy.materials:')
    t = doctest.testmod(mtl)
    failed_tests += t[0]
    total_tests += t[1]
    print(t)
    print('')

    print('Testing vcfempy.meshgen:')
    t = doctest.testmod(msh)
    failed_tests += t[0]
    total_tests += t[1]
    print(t)
    print('')

    print('Testing vcfempy.flow')
    t = doctest.testmod(flw)
    failed_tests += t[0]
    total_tests += t[1]
    print(t)
    print('')

    print([f'Testing complete... failed_tests = {failed_tests}, ',
           f'total_tests = {total_tests}'])
    print('')


if __name__ == '__main__':
    main()
