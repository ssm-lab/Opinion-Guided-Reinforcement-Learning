# Subjective logic module

import numpy as np


'''
Belief constraint fusion for two matrices with the same dimensions
'''
def beliefConstraintFusion(opinion1, opinion2):
    assert(len(opinion1) == 4)
    assert(len(opinion2) == 4)
    
    [b1, d1, u1, a1] = opinion1
    [b2, d2, u2, a2] = opinion2
    
    assert(b1+d1+u1 == 1)
    assert(b2+d2+u2 == 1)
    
    harmony = b1*b2 + b1*u2 + b2*u1
    conflict = b1*d2 + b2*d1
    b = harmony / (1 - conflict)
    u = u1 * u2 / (1 - conflict)
    d = 1 - (b + u)
    a = (a1 * (1 - u1) + a2 * (1 - u2)) / (2 - u1 - u2)
    
    
    return [b, d, u, a]