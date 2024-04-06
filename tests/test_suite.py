#!/usr/bin/env python
import unittest

from .parser_tests import ParserTests
from .model_tests import ModelTests
from .grid_tests import GridTests

"""
Full test suite
"""

def create_suite():
    testCases = [ParserTests, ModelTests, GridTests]
    loadedCases = []
    
    for case in testCases:
        loadedCases.append(unittest.defaultTestLoader.loadTestsFromTestCase(case))

    return unittest.TestSuite(loadedCases)


if __name__ == '__main__':
    suite = create_suite()
    runner = unittest.TextTestRunner()
    runner.run(suite)
