import unittest
from src.parser import Parser
import os

class ParserTests(unittest.TestCase):
    
    def setUp(self):
        self._parser = Parser()
        
    def tearDown(self):
        del(self._parser)

    def testValidInput(self):
        file = os.path.abspath("tests/validinput.txt")
        
        with open(file, 'r') as f:
            lines = len(f.readlines())
            expectedNumberOfHints = lines-2
        
        configuration = self._parser.parse(file)
        
        self.assertTrue(configuration.u >= 0)
        self.assertTrue(configuration.u <= 1)
        self.assertEqual(len(configuration.hints), expectedNumberOfHints)
        
if __name__ == "__main__":
    unittest.main()
