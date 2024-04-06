import os
import unittest
from src import sl


class SLTests(unittest.TestCase):
    
    def testBCF(self):
        opinion1 = [0.5, 0, 0.5, 0.25]
        opinion2 = [0.6, 0.2, 0.2, 0.25]
        fused_opinion = sl.beliefConstraintFusion(opinion1, opinion2)
        
        self.assertAlmostEqual(fused_opinion[0], 0.7777, delta=0.001)
        self.assertAlmostEqual(fused_opinion[1], 0.1111, delta=0.001)
        self.assertAlmostEqual(fused_opinion[2], 0.1111, delta=0.001)
        self.assertAlmostEqual(fused_opinion[3], 0.25, delta=0.001)
        
    def testBCFVacuous(self):
        opinion1 = [0.4, 0.6, 0, 0.25]
        opinion2 = [0.8, 0.2, 0, 0.25]
        fused_opinion = sl.beliefConstraintFusion(opinion1, opinion2)
        
        self.assertAlmostEqual(fused_opinion[0], 0.727, delta=0.001)
        self.assertAlmostEqual(fused_opinion[1], 0.273, delta=0.001)
        self.assertEqual(fused_opinion[2], 0)
        self.assertAlmostEqual(fused_opinion[3], 0.25, delta=0.001)
        
if __name__ == "__main__":
    unittest.main()
