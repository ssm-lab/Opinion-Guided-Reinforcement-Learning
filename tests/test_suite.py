#!/usr/bin/env python
import unittest

from .grid_tests import GridTests
from .model_tests import ModelTests
from .opinion_parser_tests import OpinionParserTests
from .sl_tests import SLTests


"""
Full test suite
"""

def create_suite():
    testCases = [GridTests, ModelTests, OpinionParserTests, SLTests]
    loadedCases = []
    
    for case in testCases:
        loadedCases.append(unittest.defaultTestLoader.loadTestsFromTestCase(case))

    return unittest.TestSuite(loadedCases)


if __name__ == '__main__':
    suite = create_suite()
    runner = unittest.TextTestRunner()
    runner.run(suite)
