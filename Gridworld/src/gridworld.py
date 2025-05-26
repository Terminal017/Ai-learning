from src.map_parser import Map  # 修改为绝对导入

class GridWorld:
    def __init__(self, grid_map):
        self.grid_map = grid_map

    def get_cells(self):
        return self.grid_map.get_cells()

    def get_cell_by_coords(self, row, col):
        return self.grid_map.get_cell_by_coords(row, col)

    def get_width(self):
        return self.grid_map.get_width()

    def get_height(self):
        return self.grid_map.get_height()

    def propose_move(self, current_cell, action):
        """Simulates the effect of an action from a given cell."""
        if current_cell.is_goal():
            return current_cell # Agent stays at goal

        new_row, new_col = current_cell.row, current_cell.col

        if action == 'GO_NORTH':
            new_row -= 1
        elif action == 'GO_EAST':
            new_col += 1
        elif action == 'GO_SOUTH':
            new_row += 1
        elif action == 'GO_WEST':
            new_col -= 1
        else:
            # Handle 'NONE' or invalid actions: agent stays put
            return current_cell

        proposed_cell = self.grid_map.get_cell_by_coords(new_row, new_col)

        if proposed_cell is None or proposed_cell.is_wall():
            # Illegal move, agent remains in current state
            return current_cell
        else:
            return proposed_cell

    def get_transition_probability(self, old_state, new_state, action):
        """
        Determines the transition probability.
        In this deterministic gridworld, it's 1 for the correct next state, 0 otherwise.
        """
        if old_state.is_goal():
            # Rule 4: Don't transition from the goal cell, stay at goal
            return 1 if old_state == new_state else 0
        
        proposed_next_state = self.propose_move(old_state, action)
        
        if proposed_next_state is None: # Should not happen with current propose_move logic
            return 0
        
        if proposed_next_state.get_index() == new_state.get_index():
            return 1
        else:
            return 0


    def get_reward(self, old_state, new_state, action):
        """
        Defines the reward function R(s, s', a).
        For gridworld, R(s,s') is 1 if s' is goal, -1 otherwise.
        """
        if new_state and new_state.is_goal():
            return 1
        else:
            return -1