import re
from model import Cell, Opinion, HumanInput


class OpinionParser():

    def parse(self, file):
        with open(file, 'r') as f:
            map_size = f.readline()
            opinions = []
            for line in f:
                [cell, value] = re.split(r' ', line)
                coordinates = re.search(r'[0-9]{1},[0-9]{1}', cell)
                [y, x] = re.split(r',', coordinates.group(0))
                cell = Cell(int(y), int(x), int(map_size))
                opinions.append(Opinion(cell, int(value)))
                
        return HumanInput(int(map_size), opinions)
