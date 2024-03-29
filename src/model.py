"""
Represents an opinion about a cell
"""
class Hint():

    def __init__(self, cellid:int, opinion:int, uncertainty:float):
        self.cellid = cellid
        self.opinion = opinion
        self.u = uncertainty
        self.normalizeBeliefForUncertainty()
        
    def __str__(self):
        return 'Hint(cellid: {}, hint: {}). (At uncertainty = {}. =>belief = {}, =>disbelief = {}.)'.format(self.cellid, self.opinion, self.u, self.b, self.d)
        
    def normalizeBeliefForUncertainty(self):
        #these are hard-coded values to be replaced when we generalize the framework... famous last words
        self.b = (self.opinion + 2) * ((1 - self.u)/(4))
        self.d = 1 - (self.b + self.u)

"""
Full configuration with every Hint and the uncertainty parameter
"""
class Configuration():
    
    def __init__(self, u: float, hints):
        self.u = u
        self.hints = hints