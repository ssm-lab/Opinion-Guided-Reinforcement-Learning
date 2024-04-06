import re
from model import Cell, Hint, HumanInput


class Parser():

    def parse(self, file):
        with open(file, 'r') as f:
            map_size = f.readline()
            u = f.readline()
            hints = []
            for line in f:
                [cell, opinion] = re.split(r' ', line)
                coordinates = re.search(r'[0-9]{1},[0-9]{1}', cell)
                [y, x] = re.split(r',', coordinates.group(0))
                cell = Cell(int(y), int(x), int(map_size))
                hints.append(Hint(cell, int(opinion), float(u)))
                
        return HumanInput(int(map_size), float(u), hints)
