"""
A* Implementation for n-Puzzle Problem,
following the pseudocode from Slide 32 - Chapter 4 so A* with duplicate elimination and no reopening
"""

import heapq
from typing import List, Tuple, Set, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass(order=True)
class Node:
    """Node in the search tree"""
    f: float = field(compare=True)  # f(n) = g(n) + h(n)
    g: float = field(compare=False)  # Cost 
    state: Tuple = field(compare=False)  # State 
    parent: Optional['Node'] = field(default=None, compare=False)
    action: Optional[str] = field(default=None, compare=False)
    
    def __hash__(self):
        return hash(self.state)


class Problem(ABC):
    """Abstract class for a search problem"""
    
    @abstractmethod
    def initial_state(self) -> Tuple:
        """Return the initial state"""
        pass
    
    @abstractmethod
    def goal_test(self, state: Tuple) -> bool:
        """Return True if state is a goal state"""
        pass
    
    @abstractmethod
    def actions(self, state: Tuple) -> List[str]:
        """Return list of valid actions from state"""
        pass
    
    @abstractmethod
    def result(self, state: Tuple, action: str) -> Tuple:
        """Return the state that results from executing action in state"""
        pass
    
    @abstractmethod
    def step_cost(self, state: Tuple, action: str, next_state: Tuple) -> float:
        """Return the cost of taking action from state to next_state"""
        pass


class NPuzzle(Problem):
    """n-Puzzle problem implementation"""
    
    def __init__(self, initial: List[List[int]]):
        """
        Initialize n-puzzle with initial configuration
        """
        self.n = len(initial)
        self.initial = tuple(tuple(row) for row in initial)
        
        goal = list(range(1, self.n * self.n)) + [0]
        self.goal = tuple(tuple(goal[i*self.n:(i+1)*self.n]) 
                         for i in range(self.n))
    
    def initial_state(self) -> Tuple:
        return self.initial
    
    def goal_test(self, state: Tuple) -> bool:
        return state == self.goal
    
    def find_blank(self, state: Tuple) -> Tuple[int, int]:
        for i in range(self.n):
            for j in range(self.n):
                if state[i][j] == 0:
                    return (i, j)
        return (-1, -1)
    
    def actions(self, state: Tuple) -> List[str]:
        row, col = self.find_blank(state)
        valid_actions = []
        
        if row > 0:
            valid_actions.append("UP")
        if row < self.n - 1:
            valid_actions.append("DOWN")
        if col > 0:
            valid_actions.append("LEFT")
        if col < self.n - 1:
            valid_actions.append("RIGHT")
        
        return valid_actions
    
    def result(self, state: Tuple, action: str) -> Tuple:
        """Execute action and return new state"""
        row, col = self.find_blank(state)
        new_state = [list(r) for r in state]
        
        if action == "UP":
            new_state[row][col], new_state[row-1][col] = \
                new_state[row-1][col], new_state[row][col]
        elif action == "DOWN":
            new_state[row][col], new_state[row+1][col] = \
                new_state[row+1][col], new_state[row][col]
        elif action == "LEFT":
            new_state[row][col], new_state[row][col-1] = \
                new_state[row][col-1], new_state[row][col]
        elif action == "RIGHT":
            new_state[row][col], new_state[row][col+1] = \
                new_state[row][col+1], new_state[row][col]
        
        return tuple(tuple(r) for r in new_state)
    
    def step_cost(self, state: Tuple, action: str, next_state: Tuple) -> float:
        return 1.0


class Heuristic(ABC):
    """Abstract class for heuristic functions"""
    
    @abstractmethod
    def __call__(self, state: Tuple, goal: Tuple) -> float:
        """Calculate heuristic value for state"""
        pass


class ManhattanDistance(Heuristic):
    """Manhattan Distance heuristic"""
    
    def __call__(self, state: Tuple, goal: Tuple) -> float:
        n = len(state)
        distance = 0
        
        # Create goal position map
        goal_pos = {}
        for i in range(n):
            for j in range(n):
                if goal[i][j] != 0:
                    goal_pos[goal[i][j]] = (i, j)
        
        # Calculate Manhattan distance for each tile
        for i in range(n):
            for j in range(n):
                tile = state[i][j]
                if tile != 0:
                    goal_i, goal_j = goal_pos[tile]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        
        return float(distance)


class MisplacedTiles(Heuristic):
    """Misplaced Tiles heuristic"""
    
    def __call__(self, state: Tuple, goal: Tuple) -> float:
        n = len(state)
        count = 0
        
        for i in range(n):
            for j in range(n):
                if state[i][j] != 0 and state[i][j] != goal[i][j]:
                    count += 1
        
        return float(count)

class LinearConflict(Heuristic):
    """
    Linear Conflict heuristic for n-Puzzle
    Adds 2 to Manhattan Distance for each pair of tiles in conflict.
    
    Two tiles are in linear conflict if:
    - They are in the same row/column
    - Both are in their goal row/column
    - Their relative position is reversed
    """
    
    def __call__(self, state: Tuple, goal: Tuple) -> float:
        n = len(state)
        
        # Start with Manhattan Distance
        manhattan = ManhattanDistance()
        distance = manhattan(state, goal)
        
        # Create goal position map
        goal_pos = {}
        for i in range(n):
            for j in range(n):
                if goal[i][j] != 0:
                    goal_pos[goal[i][j]] = (i, j)
        
        conflicts = 0
        
        # Check row conflicts
        for row in range(n):
            tiles_in_goal_row = []
            for col in range(n):
                tile = state[row][col]
                if tile != 0:
                    goal_row, goal_col = goal_pos[tile]
                    # Check if tile belongs to this row
                    if goal_row == row:
                        tiles_in_goal_row.append((tile, col, goal_col))
            
            # Count conflicts in this row
            for i in range(len(tiles_in_goal_row)):
                for j in range(i + 1, len(tiles_in_goal_row)):
                    tile1, curr_col1, goal_col1 = tiles_in_goal_row[i]
                    tile2, curr_col2, goal_col2 = tiles_in_goal_row[j]
                    
                    # If tiles are in reversed order
                    if curr_col1 < curr_col2 and goal_col1 > goal_col2:
                        conflicts += 1
                    elif curr_col1 > curr_col2 and goal_col1 < goal_col2:
                        conflicts += 1
        
        # Check column conflicts
        for col in range(n):
            tiles_in_goal_col = []
            for row in range(n):
                tile = state[row][col]
                if tile != 0:
                    goal_row, goal_col = goal_pos[tile]
                    # Check if tile belongs to this column
                    if goal_col == col:
                        tiles_in_goal_col.append((tile, row, goal_row))
            
            # Count conflicts in this column
            for i in range(len(tiles_in_goal_col)):
                for j in range(i + 1, len(tiles_in_goal_col)):
                    tile1, curr_row1, goal_row1 = tiles_in_goal_col[i]
                    tile2, curr_row2, goal_row2 = tiles_in_goal_col[j]
                    
                    # If tiles are in reversed order
                    if curr_row1 < curr_row2 and goal_row1 > goal_row2:
                        conflicts += 1
                    elif curr_row1 > curr_row2 and goal_row1 < goal_row2:
                        conflicts += 1
        
        # Each conflict adds 2 moves (one tile must move out and back)
        return distance + 2 * conflicts


def astar(problem: Problem, heuristic: Heuristic) -> Tuple[Optional[Node], dict]:
    """
    A* search with duplicate elimination and no reopening
    Following pseudocode from Slide 32
    
    Returns: (solution_node, statistics)
    """
    
    import time
    
    # Statistics - COMPLETE
    stats = {
        'expanded_nodes': 0,
        'generated_nodes': 0,
        'max_frontier_size': 0,
        'max_explored_size': 0,  
        'max_memory_nodes': 0,    
        'branching_factors': [],  
        'running_time': 0.0,      
        'search_time': 0.0,       
    }
    
    start_time = time.time()
    
    # Initialize
    initial_state = problem.initial_state()
    h_initial = heuristic(initial_state, 
                          problem.goal if hasattr(problem, 'goal') else initial_state)
    
    node = Node(f=h_initial, g=0.0, state=initial_state, parent=None, action=None)
    
    # Frontier: priority queue ordered by ascending f = g + h
    frontier = []
    heapq.heappush(frontier, node)
    stats['generated_nodes'] += 1
    
    # Explored: set of states
    explored = set()
    
    # Keep track of states in frontier for duplicate checking
    frontier_states = {node.state: node}
    
    while frontier:
        # Track max sizes
        stats['max_frontier_size'] = max(stats['max_frontier_size'], len(frontier))
        stats['max_explored_size'] = max(stats['max_explored_size'], len(explored))
        stats['max_memory_nodes'] = max(stats['max_memory_nodes'], 
                                       len(frontier) + len(explored))
        
        # Pop node with minimum f value
        node = heapq.heappop(frontier)
        del frontier_states[node.state]
        
        # Goal test
        if problem.goal_test(node.state):
            stats['running_time'] = time.time() - start_time
            stats['search_time'] = stats['running_time']
            
            # Calculate branching factor statistics
            if stats['branching_factors']:
                stats['avg_branching_factor'] = sum(stats['branching_factors']) / len(stats['branching_factors'])
                stats['max_branching_factor'] = max(stats['branching_factors'])
                stats['min_branching_factor'] = min(stats['branching_factors'])
            else:
                stats['avg_branching_factor'] = 0
                stats['max_branching_factor'] = 0
                stats['min_branching_factor'] = 0
            
            return node, stats
        
        # Add to explored
        explored.add(node.state)
        stats['expanded_nodes'] += 1
        
        # Expand node and count children for branching factor
        children_count = 0
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            child_g = node.g + problem.step_cost(node.state, action, child_state)
            child_h = heuristic(child_state, 
                               problem.goal if hasattr(problem, 'goal') else child_state)
            child_f = child_g + child_h
            
            child = Node(f=child_f, g=child_g, state=child_state, 
                        parent=node, action=action)
            
            # Check if child.state is not in explored ∪ States(frontier)
            if child_state not in explored and child_state not in frontier_states:
                heapq.heappush(frontier, child)
                frontier_states[child_state] = child
                stats['generated_nodes'] += 1
                children_count += 1
            
            # Check if exists n'' in frontier with same state but higher g
            elif child_state in frontier_states:
                existing = frontier_states[child_state]
                if child_g < existing.g:
                    # Replace n'' in frontier with n'
                    frontier = [n for n in frontier if n.state != child_state]
                    heapq.heapify(frontier)
                    heapq.heappush(frontier, child)
                    frontier_states[child_state] = child
                    children_count += 1
        
        # Record branching factor for this node
        if children_count > 0:
            stats['branching_factors'].append(children_count)
    
    # No solution found
    stats['running_time'] = time.time() - start_time
    stats['search_time'] = stats['running_time']
    return None, stats


def reconstruct_path(node: Node) -> List[str]:
    """Reconstruct path from initial state to goal"""
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    path.reverse()
    return path
