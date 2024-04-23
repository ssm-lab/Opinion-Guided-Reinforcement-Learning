from enum import Enum
import sl

class Direction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

"""
Represents a grid of cells
"""
class Grid():
    def __init__(self, edge_size:int):
        self.edge_size = edge_size
        self.cells =  []
        
        for row in range(edge_size):
            for col in range(edge_size):
                cell = Cell(row, col, edge_size)
                self.cells.append(cell)
    
    def get_cell_by_coordinates(self, row:int, col:int):
        return self.cells[row*self.edge_size + col]
        
    def get_cell_by_sequence_number(self, sequence_number:int):
        return self.cells[sequence_number]

"""
Represents a cell
"""
class Cell():
    def __init__(self, row:int, col:int, edge_size:int):
        self.row = row
        self.col = col
        self.edge_size = edge_size
        
    def __str__(self):
        return '({}, {})'.format(self.row, self.col)
        
    def get_sequence_number_in_grid(self):
        return self.row*self.edge_size + self.col
        
    def get_cell_in_direction(self, direction:Direction):
        if(direction == Direction.LEFT):
            return None if self.col == 0 else [self.row, self.col-1]
        elif(direction == Direction.DOWN):
            return None if self.row == self.edge_size-1 else [self.row+1, self.col]
        elif(direction == Direction.RIGHT):
            return None if self.col == self.edge_size-1 else [self.row, self.col+1]
        elif(direction == Direction.UP):
            return None if self.row == 0 else [self.row-1, self.col]
        
    def get_neighbors(self):
        return [self.get_cell_in_direction(d) for d in Direction]
    
    def get_action_to_me_from_neighbor(self, direction:Direction):
        if self.get_cell_in_direction(direction) is None:
            return None
        
        if(direction == Direction.LEFT):
            return Direction.RIGHT
        elif(direction == Direction.DOWN):
            return Direction.UP
        elif(direction == Direction.RIGHT):
            return Direction.LEFT
        elif(direction == Direction.UP):
            return Direction.DOWN
    
    def get_actions_to_me_from_all_neighbors(self):
        state_action_pairs = [(self.get_cell_in_direction(d), self.get_action_to_me_from_neighbor(d)) for d in Direction]
        return [sap for sap in state_action_pairs if None not in sap]

"""
Represents an opinion about a cell
"""
class Opinion():

    def __init__(self, cell:Cell, value:int):
        self.cell = cell
        self.value = value
        
    def __str__(self):
        return 'Opinion(cell: {}, value: {}). (At uncertainty = {}. =>belief = {}, =>disbelief = {}.)'.format(self.cell, self.value, self.u, self.b, self.d)
        
    def normalize_belief_for_uncertainty(self):
        # these are hard-coded values to be replaced when we generalize the framework
        self.b = (self.value + 2) * ((1 - self.u)/(4))
        self.d = 1 - (self.b + self.u)
        
        # some floating-point issues, as usual with Python
        self.b = round(self.b, 4)
        self.d = round(self.d, 4)
        
    def get_binomial_opinion(self, base_rate):
        return [self.b, self.d, self.u, base_rate]
        
    def project(self, base_rate):
        return sl.opinion_to_probability([self.b, self.d, self.u, base_rate])

"""
Human input: a collection of opinions without the level of uncertainty
"""
class HumanInput():
    
    def __init__(self, map_size: int, opinions):
        self.map_size = map_size
        self.opinions = opinions
        
    def __str__(self):
        return f'Human input with {len(self.opinions)} opinions.'
        
"""
Advice compiled from human input at a specific level of uncertainty
"""        
class Advice():
    
    def __init__(self, human_input: HumanInput, u: float):
        self.opinions = human_input.opinions
        self.u = u
        self.compile_advice()
        
    def compile_advice(self):
        for opinion in self.opinions:
            opinion.u = self.u
            opinion.normalize_belief_for_uncertainty()
        
    def __str__(self):
        return f'Advice with {len(self.opinions)} opinions.'