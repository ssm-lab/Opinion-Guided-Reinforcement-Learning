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
        
    def get_cell_above(self):
        return None if self.y == 0 else [self.y-1, self.x]
        
    def get_cell_below(self):
        return None if self.y == self.map_size-1 else [self.y+1, self.x]
    
    def get_cell_to_the_left(self):
        return None if self.x == 0 else [self.y, self.x-1]
        
    def get_cell_to_the_right(self):
        return None if self.x == self.map_size-1 else [self.y, self.x+1]
        
    def get_neighbors(self):
        return [self.get_cell_to_the_left(), self.get_cell_below(), self.get_cell_to_the_right(), self.get_cell_above()]
        
    def get_sequence_number(self):
        return self.y*self.map_size + self.x

"""
Represents an opinion about a cell
"""
class Hint():

    def __init__(self, cell:Cell, opinion:int, uncertainty:float):
        self.cell = cell
        self.opinion = opinion
        self.u = uncertainty
        self.normalizeBeliefForUncertainty()
        
    def __str__(self):
        return 'Hint(cell: {}, hint: {}). (At uncertainty = {}. =>belief = {}, =>disbelief = {}.)'.format(self.cell, self.opinion, self.u, self.b, self.d)
        
    def normalizeBeliefForUncertainty(self):
        #these are hard-coded values to be replaced when we generalize the framework...
        self.b = (self.opinion + 2) * ((1 - self.u)/(4))
        self.d = 1 - (self.b + self.u)
        
        # some floating-point issues, as usual with Python
        self.b = round(self.b, 4)
        self.d = round(self.b, 4)
        
    def get_P(self):
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
        
        
cell00 = Cell(0, 0, 4)
cell01 = Cell(0, 1, 4)
#print(cell00)
#print(cell01)

grid = Grid(4)
for cell in grid.cells:
    print(cell)
    
print(grid.get_cell_by_coordinates(0, 2))   # should print 0,2
print(grid.get_cell_by_coordinates(3, 3))   # should print 3,3
print(grid.get_cell_by_coordinates(3, 3).get_sequence_number())   # should print 15

print(grid.get_cell_by_coordinates(3, 3).get_cell_above())  # should print 2,3
print(grid.get_cell_by_coordinates(3, 3).get_cell_below())  # should print None
print(grid.get_cell_by_coordinates(3, 3).get_cell_to_the_left())  # should print 3,2
print(grid.get_cell_by_coordinates(3, 3).get_cell_to_the_right())  # should print None
print(grid.get_cell_by_coordinates(3, 3).get_neighbors())  # should print (3,2), None, None, (2,3)