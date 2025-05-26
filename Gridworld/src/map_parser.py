import numpy as np

class Cell:
    def __init__(self, row, col, cell_type, index):
        self.row = row
        self.col = col
        self.cell_type = cell_type # e.g., '#', 'X', ' '
        self.index = index # Unique index for each cell

    def is_wall(self):
        return self.cell_type == '#'

    def is_goal(self):
        return self.cell_type == 'X'

    def is_empty(self):
        return self.cell_type == ' '

    def get_coords(self):
        return (self.row, self.col)

    def get_index(self):
        return self.index

class Map:
    def __init__(self, grid_cells, width, height):
        self.grid_cells = grid_cells # A flattened list of Cell objects
        self.width = width
        self.height = height

    def get_cell_by_coords(self, row, col):
        if 0 <= row < self.height and 0 <= col < self.width:
            # Calculate index from row and col
            return self.grid_cells[row * self.width + col]
        return None

    def get_cells(self):
        return self.grid_cells

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

class MapParser:
    def parse_map(self, file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError("Map file is empty.")

        height = len(lines)
        width = len(lines[0])
        grid_cells = []
        index = 0

        for r, line in enumerate(lines):
            if len(line) != width:
                raise ValueError("Map rows must have consistent width.")
            for c, char in enumerate(line):
                grid_cells.append(Cell(r, c, char, index))
                index += 1
        
        return Map(grid_cells, width, height)