import re
from model import Cell, Advice, AdvisorInput


class AdviceParser():

    def parse(self, file):
        with open(file, 'r') as f:
            map_size = f.readline()
            advice_list = []
            for line in f:
                [cell, value] = re.split(r' ', line)
                coordinates = re.search(r'[0-9]+,[0-9]+', cell)
                [y, x] = re.split(r',', coordinates.group(0))
                cell = Cell(int(y), int(x), int(map_size))
                advice_list.append(Advice(cell, int(value)))
                
        return AdvisorInput(int(map_size), advice_list)
