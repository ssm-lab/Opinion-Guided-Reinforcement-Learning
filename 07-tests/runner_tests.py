import os
import unittest
from src.runner import Runner
import numpy as np
from opinion_parser import OpinionParser
from model import Advice

class RunnerTests(unittest.TestCase):
    
    def testShaping(self):
        #3x3 grid
        
        policy = np.full((9, 4), 0.25)
        p1 = policy
        
        print(p1)
        
        #advice about the middle cell
        file = os.path.abspath(f'input/opinions-test.txt')
        opinion_parser = OpinionParser()
        
        human_input = opinion_parser.parse(file)
        #advice = Advice(human_input, 0)
        advice = Advice(human_input, 0.99)
        
        print(f'{advice} @u={advice.u}.')
        
        r = Runner(12, 63, 2, 10)
        policy = r.shape_policy(policy, advice)
        
        print(policy)
                
        print(np.array_equal(p1, policy))
        
if __name__ == "__main__":
    unittest.main()
