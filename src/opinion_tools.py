import argparse
import os
from map_tools import MapTools
from abc import ABC, abstractmethod
from openpyxl import load_workbook, Workbook
from model import Cell, Opinion, Fact, Direction

class OpinionStrategy(ABC):

    def __init__(self, size, seed):
        self._size = size
        self._seed = seed
        
        self._MAPS_PATH = './maps'
        self._facts = self.parse_map()
        self._opinions = self.generate_opinions_from_facts()
    
    def generate_opinions_from_facts(self):
        goal_opinions = []
        hole_opinions = []
        frozen_opinions = []
        
        for fact in self._facts:
            if(fact.value == 'G'):
                goal_opinions.append(Opinion(fact.cell, 2))
            elif(fact.value == 'H'):
                hole_opinions.append(Opinion(fact.cell, -2))
            elif(fact.value == 'F'):
                # todo: this is broken here
                neighboring_holes = len([f for c in [n for n in fact.cell.get_neighbors() if n is not None] for f in self._facts if f.cell.row == c[0] and f.cell.col == c[1] and f.value=='H'])
                if(neighboring_holes == 0):
                    frozen_opinions.append(Opinion(fact.cell, 1))
                elif(neighboring_holes == 1):
                    frozen_opinions.append(Opinion(fact.cell, 0))
                else:
                    frozen_opinions.append(Opinion(fact.cell, -1))
        
        return {'goal': goal_opinions, 'holes': hole_opinions, 'frozen': frozen_opinions}
    
    @abstractmethod
    def select_opinions(self):
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
    
    def save_opinion_file(self, opinions):
        results_folder = os.path.abspath(self._MAPS_PATH)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        
        with open(f'{self._MAPS_PATH}/opinions-{self._size}x{self._size}-seed{self._seed}.txt', 'w') as file:
            file.write(f'{self._size}')
            for opinion in opinions:
                opinion_string = f'\n[{opinion.cell.row},{opinion.cell.col}], {opinion.value:+}'
                file.write(opinion_string)

    
class EveryCellStrategy(OpinionStrategy):
    
    def select_opinions(self):
        print('Generating hints for all cells')
    
class JustTheHolesStrategy(OpinionStrategy):
    
    def select_opinions(self):
        #[print(o) for o in self._opinions['goal']]
        #[print(o) for o in self._opinions['holes']]
        
        self.save_opinion_file(self._opinions['goal']+self._opinions['holes'])
    
class RandomSampleStrategy(OpinionStrategy):
    
    def select_opinions(self):
        print('Generating hints for a random sample')


#move opinion_parser here
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate', required=True, choices=['all', 'holes', 'random'])
    parser.add_argument('--size', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)

    options = parser.parse_args()
    
    assert options.size
    assert options.seed
    size = int(options.size)
    seed = int(options.seed)
    
    if(options.generate == 'all'):
        EveryCellStrategy(size, seed).select_opinions()
    elif(options.generate == 'holes'):
        JustTheHolesStrategy(size, seed).select_opinions()
    elif(options.generate == 'random'):
        RandomSampleStrategy(size, seed).select_opinions()
    else:
        raise Error('Invalid argument')
