"""
Generate n-puzzle instances with controlled difficulty
Scaling parameter: number of random moves from goal state
"""

import random
from typing import List, Tuple
from astar import NPuzzle


def generate_goal_state(n: int) -> List[List[int]]:
    """Generate standard goal state for n×n puzzle"""
    goal = list(range(1, n * n)) + [0]
    return [goal[i*n:(i+1)*n] for i in range(n)]


def generate_random_instance(n: int, num_moves: int, seed: int = None) -> List[List[int]]:
    """
    Generate a solvable n-puzzle instance by making random moves from goal
    
    Args:
        n: size of puzzle (n×n)
        num_moves: number of random moves (scaling parameter for difficulty)
        seed: random seed for reproducibility
    
    Returns:
        n×n puzzle configuration
    """
    if seed is not None:
        random.seed(seed)
    
    # Start from goal state
    problem = NPuzzle(generate_goal_state(n))
    state = problem.goal
    
    # Make random moves
    visited = {state}
    for _ in range(num_moves):
        actions = problem.actions(state)
        
        # Try to avoid going back to visited states (but allow if necessary)
        unvisited_actions = []
        for action in actions:
            next_state = problem.result(state, action)
            if next_state not in visited:
                unvisited_actions.append(action)
        
        # Prefer unvisited, but use any action if all visited
        action = random.choice(unvisited_actions if unvisited_actions else actions)
        state = problem.result(state, action)
        visited.add(state)
    
    # Convert tuple back to list
    return [list(row) for row in state]


def generate_test_suite(n: int, difficulties: List[int], instances_per_difficulty: int = 3) -> List[Tuple[int, List[List[int]]]]:
    """
    Generate a test suite with varying difficulties
    
    Args:
        n: puzzle size
        difficulties: list of num_moves (e.g., [5, 10, 20, 50, 100])
        instances_per_difficulty: how many instances per difficulty
    
    Returns:
        List of (difficulty, puzzle) tuples
    """
    test_suite = []
    
    for difficulty in difficulties:
        for i in range(instances_per_difficulty):
            seed = difficulty * 1000 + i  # Reproducible seeds
            puzzle = generate_random_instance(n, difficulty, seed=seed)
            test_suite.append((difficulty, puzzle))
    
    return test_suite


# Example usage
if __name__ == "__main__":
    # Generate test suite for 3×3 puzzle
    difficulties = [5, 10, 20, 30, 50]
    test_suite = generate_test_suite(n=3, difficulties=difficulties, instances_per_difficulty=3)
    
    print(f"Generated {len(test_suite)} test instances")
    print(f"\nExample (difficulty {test_suite[0][0]}):")
    for row in test_suite[0][1]:
        print(row)