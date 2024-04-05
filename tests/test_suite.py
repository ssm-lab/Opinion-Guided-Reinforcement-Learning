#!/usr/bin/env python
import unittest

from .parser_tests import ParserTests
from .sl_tests import SLTests

"""
Full test suite
"""

def create_suite():
    testCases = [ParserTests, SLTests]
    loadedCases = []
    
    for case in testCases:
        loadedCases.append(unittest.defaultTestLoader.loadTestsFromTestCase(case))

    return unittest.TestSuite(loadedCases)


if __name__ == '__main__':
    suite = create_suite()
    runner = unittest.TextTestRunner()
    runner.run(suite)