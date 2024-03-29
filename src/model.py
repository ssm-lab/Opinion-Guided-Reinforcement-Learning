class Hint():

    def __init__(self, cellid:int, opinion:int):
        self.cellid = cellid
        self.opinion = opinion
        
    def __str__(self):
        return 'Hint(cellid: {}, hint: {})'.format(self.cellid, self.opinion)


class Configuration():
    
    def __init__(self, u: float, hints):
        self.u = u
        self.hints = hints