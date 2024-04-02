import re
from model import Hint, Configuration, Cell

class Parser():

    def parse(self, file):
        with open(file, 'r') as f:
            n = f.readline()
            u = f.readline()
            hints = []
            for line in f:
                [cell, opinion] = re.split(r' ', line)
                coordinates = re.search(r'[0-9]{1},[0-9]{1}', cell)
                [y, x] = re.split(r',', coordinates.group(0))
                cell = Cell(int(y), int(x), int(n))
                hints.append(Hint(cell, int(opinion), float(u)))
                
        return Configuration(int(n), float(u), hints)