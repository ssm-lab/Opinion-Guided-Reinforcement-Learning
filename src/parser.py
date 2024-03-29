import re
from .model import Hint, Configuration

class Parser():

    def parse(self, file):        
        regex = re.compile('^[0-9]+,[+-]{1}[0-2]{1}')
        
        with open(file, 'r') as f:
            u = f.readline()
            hints = []
            for line in f:
                line = line.replace(" ", "")[:-1]
                [cellid, opinion] = re.split(r',', line)
                hints.append(Hint(int(cellid), int(opinion)))
                
        return Configuration(float(u), hints)