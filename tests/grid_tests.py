import pytest
import random
from src.model import Direction, Grid, Cell


@pytest.fixture()
def grid():
    return Grid(10)


@pytest.mark.parametrize('invalid_edge_size_type', (2.5, 1.0, 4.5))
def test_valid_grid(invalid_edge_size_type):
    with pytest.raises(TypeError):
        Grid(invalid_edge_size_type)


@pytest.mark.parametrize('invalid_edge_size', (-2, 0))
def test_valid_grid(invalid_edge_size):
    with pytest.raises(AssertionError):
        Grid(invalid_edge_size)


def test_grid_size(grid):
    assert len(grid.cells) == grid.edge_size ** 2


def test_corner_cells_have_two_neighbours(grid):
    corners = [
        grid.get_cell_by_coordinates(0, 0),
        grid.get_cell_by_coordinates(0, grid.edge_size - 1),
        grid.get_cell_by_coordinates(grid.edge_size - 1, 0),
        grid.get_cell_by_coordinates(grid.edge_size - 1, grid.edge_size - 1)]

    for corner in corners:
        assert len([c for c in corner.get_neighbors() if c is not None]) == 2


def test_undefined_neighbors_are_none_type(grid):
    top_left_corner = grid.get_cell_by_coordinates(0, 0)
    assert top_left_corner.get_cell_in_direction(Direction.UP) is None


def test_sequence_number(grid):
    cells = [
        [0, 0],
        [0, grid.edge_size - 1],
        [grid.edge_size - 1, 0],
        [grid.edge_size - 1, grid.edge_size - 1],
        [random.randint(1, grid.edge_size - 2), random.randint(1, grid.edge_size - 2)]]

    for cell in cells:
        row = cell[0]
        col = cell[1]
        assert grid.get_cell_by_coordinates(row, col).get_sequence_number_in_grid() == row * grid.edge_size + col


def test_actions_to_me_from_neighbors_is_opposite_direction(grid):
    middle_cell = grid.get_cell_by_coordinates(random.randint(1, grid.edge_size-2), random.randint(1, grid.edge_size-2))

    assert middle_cell.get_action_to_me_from_neighbor(Direction.UP) == Direction.DOWN
    assert middle_cell.get_action_to_me_from_neighbor(Direction.DOWN) == Direction.UP
    assert middle_cell.get_action_to_me_from_neighbor(Direction.LEFT) == Direction.RIGHT
    assert middle_cell.get_action_to_me_from_neighbor(Direction.RIGHT) == Direction.LEFT


if __name__ == "main":
    pytest.main()
