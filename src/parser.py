import re
from .model import Hint, Configuration

class Parser():

    def __init__(self):
        pass
        
    def parse(self, file):
        hints = []
        
        regex = re.compile('^[0-9]+,[+-]{1}[0-2]{1}')
        
        with open(file, 'r') as f:
            u = f.readline()
            for line in f:
                line = line.replace(" ", "")[:-1]
                [cellid, opinion] = re.split(r',', line)
                hints.append(Hint(int(cellid), int(opinion)))
                
        return Configuration(float(u), hints)