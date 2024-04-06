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
    
'''

cell00 = Cell(0, 0, 4)
cell01 = Cell(0, 1, 4)
#print(cell00)
#print(cell01)
'''
'''
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


print(f'me: {grid.cells[5]}')
print(grid.cells[5].get_actions_to_me_from_all_neighbors())
print(f'me: {grid.cells[0]}')
print(grid.cells[0].get_actions_to_me_from_all_neighbors())
'''
