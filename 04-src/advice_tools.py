import argparse
import os
from map_tools import MapTools
from abc import ABC, abstractmethod
from openpyxl import load_workbook, Workbook
from model import Cell, Advice, Fact, Direction

class AdviceStrategy(ABC):

    def __init__(self, size, seed):
        self._size = size
        self._seed = seed
        
        self._MAPS_PATH = './02-maps'
        self._facts = self.parse_map()
        self._advice = self.generate_advice_from_facts()
    
    def generate_advice_from_facts(self):
        goal_advice = []
        hole_advice = []
        frozen_advice = []
        
        for fact in self._facts:
            if(fact.value == 'G'):
                goal_advice.append(Advice(fact.cell, 2))
            elif(fact.value == 'H'):
                hole_advice.append(Advice(fact.cell, -2))
            else: #F or S
                neighboring_holes = len([f for c in [n for n in fact.cell.get_neighbors() if n is not None] for f in self._facts if f.cell.row == c[0] and f.cell.col == c[1] and f.value=='H'])
                if(neighboring_holes == 0):
                    frozen_advice.append(Advice(fact.cell, 1))
                elif(neighboring_holes == 1):
                    frozen_advice.append(Advice(fact.cell, 0))
                else:
                    frozen_advice.append(Advice(fact.cell, -1))
        
        return {'goal': goal_advice, 'holes': hole_advice, 'frozen': frozen_advice}
    
    @abstractmethod
    def select_advice(self):
        pass

    def parse_map(self):
        print(f'parsing map lake-{self._size}x{self._size}-seed{self._seed}.xlsx')
        file = os.path.abspath(f'{self._MAPS_PATH}/lake-{self._size}x{self._size}-seed{self._seed}.xlsx')
        workbook = load_workbook(filename=file)
        
        sheet = workbook.active
        first_row = sheet[1]
        size = int(first_row[-1].value)+1
        
        facts = []
        for rid in range(0, size):
            row = sheet[rid+2]
            for cid in range(0, size):
                cell = Cell(rid, cid, self._size)
                facts.append(Fact(cell, row[cid+1].value))
        
        return facts
    
    def save_advice_file(self, advice, strategy_name):
        results_folder = os.path.abspath(self._MAPS_PATH)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        
        with open(f'{self._MAPS_PATH}/advice-{self._size}x{self._size}-seed{self._seed}-{strategy_name}.txt', 'w') as file:
            file.write(f'{self._size}')
            for advice in advice:
                advice_string = f'\n[{advice.cell.row},{advice.cell.col}], {advice.value:+}'
                file.write(advice_string)

    
class EveryCellStrategy(AdviceStrategy):
    
    def select_advice(self):
        self.save_advice_file([v for k, vs in self._advice.items() for v in vs], str(self))
    
    def __str__(self):
        return 'all'    
    
class JustTheHolesStrategy(AdviceStrategy):
    
    def select_advice(self):
        self.save_advice_file(self._advice['goal']+self._advice['holes'], str(self))
        
    def __str__(self):
        return 'holes'


#move advice_parser here
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate', required=True, choices=['all', 'holes'])
    parser.add_argument('--size', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)

    options = parser.parse_args()
    
    assert options.size
    assert options.seed
    size = int(options.size)
    seed = int(options.seed)
    
    if(options.generate == 'all'):
        EveryCellStrategy(size, seed).select_advice()
    elif(options.generate == 'holes'):
        JustTheHolesStrategy(size, seed).select_advice()
    else:
        raise Error('Invalid argument')