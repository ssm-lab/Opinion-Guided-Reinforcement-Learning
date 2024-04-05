from enum import Enum

class Direction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

"""
Represents a grid of cells
"""
class Grid():
    def __init__(self, size:int):
        self.size = size
        self.cells =  []
        
        for y in range(size):
            for x in range(size):
                cell = Cell(y, x, size)
                self.cells.append(cell)
    
    def get_cell_by_coordinates(self, y:int, x:int):
        return self.cells[y*self.size + x]

"""
Represents a cell
"""
class Cell():
    def __init__(self, y:int, x:int, map_size:int):
        self.y = y
        self.x = x
        self.map_size = map_size
        
    def __str__(self):
        return '({}, {})'.format(self.y, self.x)
        
    def get_sequence_number_in_grid(self):
        return self.y*self.map_size + self.x
        
    def get_cell_in_direction(self, direction:Direction):
        if(direction == Direction.LEFT):
            return None if self.x == 0 else [self.y, self.x-1]
        elif(direction == Direction.DOWN):
            return None if self.y == self.map_size-1 else [self.y+1, self.x]
        elif(direction == Direction.RIGHT):
            return None if self.x == self.map_size-1 else [self.y, self.x+1]
        elif(direction == Direction.UP):
            return None if self.y == 0 else [self.y-1, self.x]
        
    def get_neighbors(self):
        return [self.get_cell_in_direction(d) for d in Direction]
    
    def get_action_to_me_from_neighbor(self, direction:Direction):
        if(direction == Direction.LEFT):
            return Direction.RIGHT
        elif(direction == Direction.DOWN):
            return Direction.UP
        elif(direction == Direction.RIGHT):
            return Direction.LEFT
        elif(direction == Direction.UP):
            return Direction.DOWN

"""
Represents an opinion about a cell
"""
class Hint():

    def __init__(self, cell:Cell, opinion:int, uncertainty:float):
        self.cell = cell
        self.opinion = opinion
        self.u = uncertainty
        self.normalize_belief_for_uncertainty()
        
    def __str__(self):
        return 'Hint(cell: {}, hint: {}). (At uncertainty = {}. =>belief = {}, =>disbelief = {}.)'.format(self.cell, self.opinion, self.u, self.b, self.d)
        
    def normalize_belief_for_uncertainty(self):
        #these are hard-coded values to be replaced when we generalize the framework...
        self.b = (self.opinion + 2) * ((1 - self.u)/(4))
        self.d = 1 - (self.b + self.u)
        
        # some floating-point issues, as usual with Python
        self.b = round(self.b, 4)
        self.d = round(self.b, 4)
        
    def project(self):
        return b + a*u

"""
Human input: a collection of hints and the uncertainty parameter
"""
class HumanInput():
    
    def __init__(self, map_size: int, u: float, hints):
        self.map_size = map_size
        self.u = u
        self.hints = hints
        
    def __str__(self):
        return f'Human input with {len(self.hints)} hints at uncertainty level {self.u}.'
        

'''        
cell00 = Cell(0, 0, 4)
cell01 = Cell(0, 1, 4)
#print(cell00)
#print(cell01)

grid = Grid(4)
for cell in grid.cells:
    print(cell)

print(grid.get_cell_by_coordinates(0, 2))   # should print 0,2
print(grid.get_cell_by_coordinates(3, 3))   # should print 3,3
print(grid.get_cell_by_coordinates(3, 3).get_sequence_number_in_grid())   # should print 15

print(grid.get_cell_by_coordinates(3, 3).get_cell_in_direction(Direction.UP))  # should print 2,3
print(grid.get_cell_by_coordinates(3, 3).get_cell_in_direction(Direction.DOWN))  # should print None
print(grid.get_cell_by_coordinates(3, 3).get_cell_in_direction(Direction.LEFT))  # should print 3,2
print(grid.get_cell_by_coordinates(3, 3).get_cell_in_direction(Direction.RIGHT))  # should print None
print(grid.get_cell_by_coordinates(3, 3).get_neighbors())  # should print (3,2), None, None, (2,3)

print(grid.cells[15].get_action_to_me_from_neighbor(Direction.RIGHT))
print(grid.cells[15].get_action_to_me_from_neighbor(Direction.RIGHT).value)
'''