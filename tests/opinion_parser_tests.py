import os
import unittest
from src.opinion_parser import OpinionParser


class OpinionParserTests(unittest.TestCase):
    
    def setUp(self):
        self._parser = OpinionParser()
        
    def tearDown(self):
        del(self._parser)

    def testValidInput(self):
        file = os.path.abspath("tests/validinput.txt")
        
        with open(file, 'r') as f:
            lines = len(f.readlines())
            expectedNumberOfOpinions = lines-1
        
        human_input = self._parser.parse(file)
        
        self.assertTrue(human_input.map_size >= 0)
        self.assertEqual(len(human_input.opinions), expectedNumberOfOpinions)
        
if __name__ == "__main__":
    unittest.main()
