import unittest
from src.parser import Parser

class ParserTests(unittest.TestCase):
    
    def setUp(self):
        self._parser = Parser()
        
    def tearDown(self):
        del(self._parser)

    def testAddModel(self):
        self._parser.parse()
        
if __name__ == "__main__":
    unittest.main()
