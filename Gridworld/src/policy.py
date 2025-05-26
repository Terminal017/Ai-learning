import numpy as np
import copy

class Policy:
    def __init__(self, policy_map=None, width=0, height=0):
        # policy_map is a flattened list of actions corresponding to cells
        self.policy = policy_map if policy_map is not None else []
        self.width = width
        self.height = height
        self.values = np.zeros(len(self.policy)) # To store V(s)

    def policy_action_for_cell(self, cell):
        if not self.policy:
            return None # No policy set
        return self.policy[cell.get_index()]

    def pi(self, cell, action):
        if not self.policy:
            # Value Iteration case: consider all actions equally
            return 1 # Or 1/num_actions if you want proper probability distribution
        
        # Policy Evaluation/Iteration case: policy specifies exact action
        if self.policy_action_for_cell(cell) == action:
            return 1
        else:
            return 0

    def get_values(self):
        return self.values

    def set_value_for_cell(self, cell_index, value):
        if 0 <= cell_index < len(self.values):
            self.values[cell_index] = value

    def reset_values(self):
        self.values = np.zeros(len(self.policy))

    def __eq__(self, other):
        if not isinstance(other, Policy):
            return NotImplemented
        return self.policy == other.policy

    def __ne__(self, other):
        return not self == other
    
    def __len__(self):
        return len(self.policy)

    # Policy Evaluation methods
    def evaluate_policy(self, grid_world, gamma=1, theta=0.01):
        if len(self.policy) != len(grid_world.get_cells()):
            raise Exception("Policy dimension doesn't fit gridworld dimension.")
        
        max_iterations = 500
        V_old = None
        # Initialize values for viable cells to 0, goal cell to 0 or a positive reward if it's the target.
        # For Gridworld, we can initialize all to 0 and let Bellman equation propagate.
        V_new = np.zeros(len(grid_world.get_cells()))
        for cell in grid_world.get_cells():
            if cell.is_goal():
                V_new[cell.get_index()] = 0 # Goal state value is 0 or high reward based on your R(s,s')

        iter_count = 0
        
        while True:
            V_old = np.copy(V_new)
            iter_count += 1
            
            # Policy Evaluation Sweep
            for cell in grid_world.get_cells():
                if cell.is_goal():
                    continue # Goal state value is fixed
                
                s_idx = cell.get_index()
                
                # V(s) = sum_a pi(s,a) * sum_s' P(s'|s,a) * (R(s,s',a) + gamma * V(s'))
                expected_value = 0
                
                # Get the action specified by the policy for this cell
                policy_action = self.policy_action_for_cell(cell)
                
                # We are in a deterministic environment, so pi(s,a) will be 1 for policy_action
                # and 0 for others.
                
                # Simulate the proposed move by the policy
                proposed_next_cell = grid_world.propose_move(cell, policy_action)
                
                # Calculate R(s,s',a)
                reward = grid_world.get_reward(cell, proposed_next_cell, policy_action)
                
                # Calculate P(s'|s,a) and V(s')
                # In this deterministic gridworld, P(s'|s,a) is 1 for the actual next state and 0 for others
                next_state_value = V_old[proposed_next_cell.get_index()] if proposed_next_cell else V_old[s_idx]
                
                expected_value = reward + gamma * next_state_value
                V_new[s_idx] = expected_value

            max_diff = np.max(np.abs(V_new - V_old))
            if max_diff < theta:
                print(f"Policy evaluation converged after iteration: {iter_count}")
                break
            
            if iter_count > max_iterations:
                print(f"Policy evaluation reached max iterations ({max_iterations}).")
                break
        
        self.values = V_new
        return self.values

    # Policy Improvement methods
    def improve_policy(self, grid_world, gamma=1):
        new_policy_map = list(self.policy) # Start with current policy
        
        # Iterate over all cells (states)
        for cell in grid_world.get_cells():
            if cell.is_wall() or cell.is_goal():
                continue # No action or improvement needed for walls/goal
            
            max_q_value = -np.inf
            best_action = 'NONE'
            
            # Consider all possible actions
            possible_actions = ['GO_NORTH', 'GO_EAST', 'GO_SOUTH', 'GO_WEST']
            
            for action in possible_actions:
                # Calculate Q(s,a)
                # Q(s,a) = sum_s' P(s'|s,a) * (R(s,s',a) + gamma * V(s'))
                
                # Simulate the proposed move
                proposed_next_cell = grid_world.propose_move(cell, action)
                
                # Get the reward
                reward = grid_world.get_reward(cell, proposed_next_cell, action)
                
                # Get the value of the next state (from current policy's V-function)
                next_state_value = self.values[proposed_next_cell.get_index()] if proposed_next_cell else self.values[cell.get_index()]
                
                q_value = reward + gamma * next_state_value
                
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            
            # Update the policy for this cell
            new_policy_map[cell.get_index()] = best_action
            
        return Policy(new_policy_map, self.width, self.height)

    # Policy Iteration
    @staticmethod
    def policy_iteration(initial_policy, grid_world, gamma=1, theta=0.01):
        last_policy = copy.deepcopy(initial_policy)
        improved_policy = None
        
        iter_count = 0
        while True:
            iter_count += 1
            print(f"\nPolicy Iteration: Iteration {iter_count}")
            
            # Policy Evaluation Step
            last_policy.evaluate_policy(grid_world, gamma, theta)
            
            # Policy Improvement Step
            improved_policy = last_policy.improve_policy(grid_world, gamma)
            
            if improved_policy == last_policy:
                print("Policy Iteration converged.")
                break
            
            last_policy = improved_policy
        
        return improved_policy

    # Value Iteration
    @staticmethod
    def value_iteration(grid_world, gamma=1, theta=0.01):
        # Initialize V(s) arbitrarily (e.g., all zeros)
        V = np.zeros(len(grid_world.get_cells()))
        # Ensure goal state has initial value (often 0 or reward for reaching it)
        for cell in grid_world.get_cells():
            if cell.is_goal():
                V[cell.get_index()] = 0 # Or a positive reward if R(s,s') for goal is 0

        max_iterations = 500
        iter_count = 0

        while True:
            V_old = np.copy(V)
            iter_count += 1
            
            for cell in grid_world.get_cells():
                if cell.is_wall() or cell.is_goal():
                    continue # Walls don't have values, goal is fixed
                
                s_idx = cell.get_index()
                
                max_q_value = -np.inf
                possible_actions = ['GO_NORTH', 'GO_EAST', 'GO_SOUTH', 'GO_WEST']
                
                for action in possible_actions:
                    proposed_next_cell = grid_world.propose_move(cell, action)
                    reward = grid_world.get_reward(cell, proposed_next_cell, action)
                    next_state_value = V_old[proposed_next_cell.get_index()] if proposed_next_cell else V_old[s_idx]
                    
                    q_value = reward + gamma * next_state_value
                    
                    if q_value > max_q_value:
                        max_q_value = q_value
                
                V[s_idx] = max_q_value
            
            max_diff = np.max(np.abs(V - V_old))
            if max_diff < theta:
                print(f"Value Iteration converged after iteration: {iter_count}")
                break
            
            if iter_count > max_iterations:
                print(f"Value Iteration reached max iterations ({max_iterations}).")
                break
        
        # After V converges, derive the optimal policy
        optimal_policy_map = ['NONE'] * len(grid_world.get_cells())
        for cell in grid_world.get_cells():
            if cell.is_wall() or cell.is_goal():
                if cell.is_goal():
                    optimal_policy_map[cell.get_index()] = 'X' # Mark goal in policy map
                continue
            
            max_q_value = -np.inf
            best_action = 'NONE'
            possible_actions = ['GO_NORTH', 'GO_EAST', 'GO_SOUTH', 'GO_WEST']
            
            for action in possible_actions:
                proposed_next_cell = grid_world.propose_move(cell, action)
                reward = grid_world.get_reward(cell, proposed_next_cell, action)
                next_state_value = V[proposed_next_cell.get_index()] if proposed_next_cell else V[cell.get_index()]
                
                q_value = reward + gamma * next_state_value
                
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            
            optimal_policy_map[cell.get_index()] = best_action
            
        optimal_policy = Policy(optimal_policy_map, grid_world.get_width(), grid_world.get_height())
        optimal_policy.values = V # Store the converged values
        return optimal_policy