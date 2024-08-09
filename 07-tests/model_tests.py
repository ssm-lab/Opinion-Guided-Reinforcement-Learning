import os
import unittest
from src.model import Cell, Opinion


class ModelTests(unittest.TestCase):
    
    def testNormalization(self):
        opinion_value = 1
        u = 0.2
        expectedBelief = 0.6
        expectedDisbelief = 0.2
        
        cell = Cell(0, 0, 0)
        
        opinion = Opinion(cell, opinion_value)
        
        self.assertEqual(opinion.value, opinion_value)
        
        # compile (should be taken care of by the Advice constructor)
        opinion.u = u
        opinion.normalize_belief_for_uncertainty()
        
        self.assertAlmostEqual(opinion.b, expectedBelief, delta=0.0001)
        self.assertAlmostEqual(opinion.d, expectedDisbelief, delta=0.0001)
        
if __name__ == "__main__":
    unittest.main()
