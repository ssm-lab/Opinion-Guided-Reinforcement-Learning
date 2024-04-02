"""
Represents a cell
"""
class Cell():
    def __init__(self, y:int, x:int, n:int):
        self.y = y
        self.x = x
        self.n = n
        
    def __str__(self):
        return '({}, {})'.format(self.y, self.x)
        
    def getCellAbove(self):
        return Cell(max(self.y-1, 0), self.x)
        
    def getCellBelow(self):
        return Cell(min(self.y+1, self.n-1), ,self.x)
    
    def getCellToTheRight(self):
        return Cell(self.y, max(self.x-1, 0))
        
    def getCellToTheLeft(self):
        return Cell(self.y, min(self.x+1, self.n-1))

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
        #these are hard-coded values to be replaced when we generalize the framework... famous last words
        self.b = (self.opinion + 2) * ((1 - self.u)/(4))
        self.d = 1 - (self.b + self.u)
        
    def get_P(self):
        return b + a*u

"""
Full configuration with every Hint and the uncertainty parameter
"""
class Configuration():
    
    def __init__(self, n: float, u: float, hints):
        self.n = n
        self.u = u
        self.hints = hints