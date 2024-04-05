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
        
    def getCellAbove(self):
        return None if self.y = 0 else [self.y-1, self.x]
        
    def getCellBelow(self):
        return None if self.y = self.map_size-1 else [self.y+1, self.x]
    
    def getCellToTheLeft(self):
        return None if self.x = 0 else [self.y, self.x-1]
        
    def getCellToTheRight(self):
        return None if self.x = self.map_size-1 else [self.y, self.x+1]

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