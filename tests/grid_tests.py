import unittest
from src.model import Grid, Cell, Direction
import os
import random

class GridTests(unittest.TestCase):
    
    def setUp(self):
        self._size = 10
        self._grid = Grid(self._size)
        
    def tearDown(self):
        del(self._size)
        del(self._grid)

    def testNumberOfCellsIsEdgeSizeSquared(self):
        self.assertEqual(len(self._grid.cells), self._size**2)
    
    def testCornerCellsHaveTwoNeighbors(self):
        corners = [
            self._grid.get_cell_by_coordinates(0,0),
            self._grid.get_cell_by_coordinates(0,self._size-1),
            self._grid.get_cell_by_coordinates(self._size-1,0),
            self._grid.get_cell_by_coordinates(self._size-1,self._size-1)]
        
        for corner in corners:
            self.assertEqual(len([c for c in corner.get_neighbors() if c is not None]), 2)
            
    def testUndefinedNeighborsAreNoneType(self):
        top_left_corner = self._grid.get_cell_by_coordinates(0,0)
        self.assertTrue(top_left_corner.get_cell_in_direction(Direction.UP) is None)
        
    def testSequenceNumber(self):
        cells = [
            [0,0],
            [0, self._size-1],
            [self._size-1, 0],
            [self._size-1, self._size-1],
            [random.randint(1, self._size-2), random.randint(1, self._size-2)]]

        for cell in cells:
            row = cell[0]
            col = cell[1]
            self.assertEqual(self._grid.get_cell_by_coordinates(row, col).get_sequence_number_in_grid(), row*self._size+col)
    
    def testActionsToMeFromNeighborsCorrespondToTheOppositeDirection(self):
        middle_cell = self._grid.get_cell_by_coordinates(random.randint(1, self._size-2), random.randint(1, self._size-2))
        
        self.assertEqual(middle_cell.get_action_to_me_from_neighbor(Direction.UP), Direction.DOWN)
        self.assertEqual(middle_cell.get_action_to_me_from_neighbor(Direction.DOWN), Direction.UP)
        self.assertEqual(middle_cell.get_action_to_me_from_neighbor(Direction.LEFT), Direction.RIGHT)
        self.assertEqual(middle_cell.get_action_to_me_from_neighbor(Direction.RIGHT), Direction.LEFT)
        
    
if __name__ == "__main__":
    unittest.main()
