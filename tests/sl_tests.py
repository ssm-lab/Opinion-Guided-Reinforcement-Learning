import unittest
from src.model import Hint
import os

class SLTests(unittest.TestCase):
    
    def testNormalization(self):
        cellid = 10
        opinion = 1
        u = 0.2
        expectedBelief = 0.6
        expectedDisbelief = 0.2
        
        hint = Hint(cellid, opinion, u)
        
        self.assertEqual(hint.cellid, cellid)
        self.assertEqual(hint.opinion, opinion)
        
        self.assertAlmostEqual(hint.b, expectedBelief, delta=0.0001)
        self.assertAlmostEqual(hint.d, expectedDisbelief, delta=0.0001)
        
if __name__ == "__main__":
    unittest.main()
