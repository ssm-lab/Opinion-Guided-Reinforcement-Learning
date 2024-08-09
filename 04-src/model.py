from enum import Enum
import numpy as np

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
        
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
        
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
Represents an fact about a cell (F/H/G in Frozen Lake)
"""
class Fact():

    def __init__(self, cell:Cell, value:int):
        self.cell = cell
        self.value = value
        
    def __str__(self):
        return 'Fact(cell: {}, value: {}).'.format(self.cell, self.value)

"""
Advisor Input: A list of advice
"""   
class AdvisorInput():

    def __init__(self, map_size: int, advice_list: list):
        self.map_size = map_size
        self.advice_list = advice_list

    def __str__(self):
        return f'Advisor 03-input with {len(self.advice_list)} pieces of advice.'

"""
Advice: Value of a cell
"""
class Advice():

    def __init__(self, cell:Cell, value:int):
        self.cell = cell
        self.value = value

    def __str__(self):
        return f'Advice at cell {self.cell} with value {self.value}'

"""
Advisor Opinions
"""
class AdvisorOpinions():
    def __init__(self):
        self.opinion_list = []
        self.advice_to_opinions()

    def __str__(self):
        return f'Advisor opinions with {len(self.opinion_list)} opinions.'
    
    def normalize_belief_for_uncertainty(self, advice_value, u: float):
        # these are hard-coded values to be replaced when we generalize the framework ... some floating-point issues, as usual with Python
        b = round((advice_value + 2) * ((1 - u)/(4)), 4)
        d = round(1 - (b + u), 4)
        return b, d

    def advice_to_opinions(self):
        pass

"""
Synthetic-Advisor Opinions: A list of opinions at uniform uncertainty
"""
class SyntheticAdvisorOpinions(AdvisorOpinions):

    def __init__(self, advisor_input: AdvisorInput, u: float, base_rate: float):
        self.advisor_input = advisor_input
        self.u = u
        self.base_rate = base_rate
        self.opinion_list = []
        self.advice_to_opinions()

    def advice_to_opinions(self):
        for advice in self.advisor_input.advice_list:
            b, d = self.normalize_belief_for_uncertainty(advice_value = advice.value, u = self.u)
            opinion = Opinion(advice.cell, b, d, self.u, self.base_rate)
            self.opinion_list.append(opinion)

"""
Human-Advisor Opinions: A list of opinions with uncertainty modulated as a function of advisor distance
"""
class HumanAdvisorOpinions(AdvisorOpinions):
    def __init__(self, advisor_input: AdvisorInput, advisor_position: str, base_rate: float):
        self.advisor_input = advisor_input
        self.advisor_position = advisor_position
        self.base_rate = base_rate
        self.map_size = advisor_input.map_size
        self.opinion_list = []
        self.set_advisor_cell()
        self.advice_to_opinions()

    def set_advisor_cell(self):
        advisor_cell_dict = {
            'topleft': Cell(0,0, self.map_size),
            'topright': Cell(0, self.map_size-1, self.map_size),
            'bottomleft': Cell(self.map_size-1, 0, self.map_size),
            'bottomright': Cell(self.map_size-1, self.map_size-1, self.map_size)
        }
        self.advisor_cell = advisor_cell_dict[self.advisor_position]

    def advice_to_opinions(self):
        for advice in self.advisor_input.advice_list:
            if advice.cell == self.advisor_cell:
                u = 0.01
            else:
                u = round(self.get_uncertainty(cell = advice.cell), 4)
            b, d = self.normalize_belief_for_uncertainty(advice_value = advice.value, u = u)
            opinion = Opinion(advice.cell, b, d, u, self.base_rate)
            self.opinion_list.append(opinion)

    def get_uncertainty(self, cell: Cell):
        distance = self.get_manhattan_distance(cell)
        max_distance = (self.map_size - 1) * 2
        return distance / max_distance

    def get_manhattan_distance(self, other_cell: Cell):
        return np.abs(self.advisor_cell.row - other_cell.row) + np.abs(self.advisor_cell.col - other_cell.col)

"""
Opinion: An binomial opinion (belief, disbelief, uncertainty, base rate) about a cell
"""
class Opinion():

    def __init__(self, cell:Cell, b: float, d: float, u: float, a:float):
        self.cell = cell
        self.b, self.d, self.u, self.a = b, d, u, a
        self.opinion_tuple = (self.b, self.d, self.u, self.a)
        assert(self.b + self.d + self.u == 1) 
        assert(0.0 <= (self.b and self.d and self.u and self.a) <= 1.0) 
        
    def __str__(self):
        return f'Opinion (b = {self.b}, d = {self.d}, u = {self.u}, a = {self.a}) about cell {self.cell}.'