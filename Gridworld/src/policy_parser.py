from src.policy import Policy  # 修改为绝对导入

class PolicyParser:
    def parse_policy(self, file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError("Policy file is empty.")

        height = len(lines)
        width = len(lines[0])
        policy_map = []
        
        for r, line in enumerate(lines):
            if len(line) != width:
                raise ValueError("Policy rows must have consistent width.")
            for c, char in enumerate(line):
                # Map characters to actions
                action = self._char_to_action(char)
                policy_map.append(action)
        
        return Policy(policy_map, width, height)

    def _char_to_action(self, char):
        if char == 'N':
            return 'GO_NORTH'
        elif char == 'E':
            return 'GO_EAST'
        elif char == 'S':
            return 'GO_SOUTH'
        elif char == 'W':
            return 'GO_WEST'
        else:
            return 'NONE' # For walls or goal cells where no action is taken