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
        
    def testProbabilityToOpinionAtZeroUncertainty(self):
        probability = 1/6
        
        opinion = sl.probability_to_opinion(probability)
        
        self.assertAlmostEqual(opinion[0], probability)
        self.assertAlmostEqual(opinion[1], 1-probability)
        self.assertEqual(opinion[2], 0)
        self.assertAlmostEqual(opinion[3], probability)
        
    def testOpinionToProbabilityAtZeroUncertainty(self):
        original_probability = 1/6
        probability = sl.opinion_to_probability([original_probability, 1-original_probability, 0, original_probability])
        
        self.assertAlmostEqual(probability, original_probability)
        
    def testOpinionToProbabilityAtNonZeroUncertainty(self):
        original_probability = 1/6
        uncertainty = 1
        opinion_based_probability = sl.opinion_to_probability([original_probability, 1-original_probability-uncertainty, uncertainty, original_probability])
        
        self.assertTrue(opinion_based_probability >= original_probability)
        
if __name__ == "__main__":
    unittest.main()
