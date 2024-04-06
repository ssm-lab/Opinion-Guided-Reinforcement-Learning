import os
import unittest
from src.model import Cell, Hint


class ModelTests(unittest.TestCase):
    
    def testNormalization(self):
        opinion = 1
        u = 0.2
        expectedBelief = 0.6
        expectedDisbelief = 0.2
        
        cell = Cell(0, 0, 0)
        
        hint = Hint(cell, opinion, u)
        
        self.assertEqual(hint.opinion, opinion)
        
        self.assertAlmostEqual(hint.b, expectedBelief, delta=0.0001)
        self.assertAlmostEqual(hint.d, expectedDisbelief, delta=0.0001)
        
if __name__ == "__main__":
    unittest.main()
