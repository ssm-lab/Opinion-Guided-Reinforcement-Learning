# Subjective logic module

import numpy as np
from model import AdvisorOpinions, Opinion, Cell

'''
Belief constraint fusion for two matrices with the same dimensions
'''
def beliefConstraintFusion(opinion1: Opinion, opinion2: Opinion):
    assert(opinion1.cell == opinion2.cell and (opinion1.cell or opinion2.cell != None)) # TODO does not allow fusion if both cells are None, should change this?

    [b1, d1, u1, a1] = opinion1.opinion_tuple
    [b2, d2, u2, a2] = opinion2.opinion_tuple

    harmony = b1*b2 + b1*u2 + b2*u1
    conflict = b1*d2 + b2*d1
    b = harmony / (1 - conflict)
    u = u1 * u2 / (1 - conflict)
    d = 1 - (b + u)
    a = (a1 * (1 - u1) + a2 * (1 - u2)) / (2 - u1 - u2)

    fused_opinion = Opinion(cell = opinion1.cell, b = b, d = d, u = u, a = a)
    
    return fused_opinion
    
def probability_to_opinion(cell: Cell, probability, uncertainty = 0):
    opinion = Opinion(cell, b=probability, d=1-probability, u=uncertainty, a=probability)
    return opinion
    
def opinion_to_probability(opinion: Opinion):
    return opinion.b + opinion.a * opinion.u

def fuse_advisor_opinions(advisor1_opinions: AdvisorOpinions, advisor2_opinions: AdvisorOpinions):
    advisor1_opinion_list = advisor1_opinions.opinion_list.copy()
    advisor2_opinion_list = advisor2_opinions.opinion_list.copy()
    fused_opinions_list = []
    for opinion1 in advisor1_opinion_list:
        for opinion2 in advisor2_opinion_list:
            if opinion1.cell == opinion2.cell:
                # fuse opinions
                fused_opinions_list.append(beliefConstraintFusion(opinion1, opinion2))
                # remove from orginal lists
                advisor1_opinion_list.remove(opinion1)
                advisor2_opinion_list.remove(opinion2)


    fused_opinions_list.extend(advisor1_opinion_list)
    fused_opinions_list.extend(advisor2_opinion_list)

    fused_opinions = AdvisorOpinions()
    fused_opinions.opinion_list = fused_opinions_list

    return fused_opinions