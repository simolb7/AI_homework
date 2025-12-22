"""
Automated experiment runner for n-Puzzle
Runs A* (3 heuristics) and PDDL on multiple instances with increasing difficulty
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

from astar import NPuzzle, astar, ManhattanDistance, MisplacedTiles, LinearConflict, reconstruct_path
from pddl import PDDLSolver
from puzzle_generator import generate_test_suite


def run_astar_experiment(puzzle: List[List[int]], heuristic_name: str) -> Dict:
    """Run A* on a puzzle instance and collect metrics"""
    
    problem = NPuzzle(puzzle)
    
    # Select heuristic
    heuristics = {
        "manhattan": ManhattanDistance(),
        "misplaced": MisplacedTiles(),
        "linear_conflict": LinearConflict()
    }
    
    if heuristic_name not in heuristics:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")
    
    heuristic = heuristics[heuristic_name]
    
    solution, stats = astar(problem, heuristic)
    
    if solution:
        path = reconstruct_path(solution)
        stats['plan_length'] = len(path)
        stats['found'] = True
    else:
        stats['plan_length'] = 0
        stats['found'] = False
    
    return stats


def run_pddl_experiment(puzzle: List[List[int]], instance_id: int, 
                       search_config: str = "astar(lmcut())") -> Dict:
    """Run PDDL planner on a puzzle instance and collect metrics"""
    
    problem = NPuzzle(puzzle)
    
    output_dir = f"./pddl_files/instance_{instance_id}"
    solver = PDDLSolver(problem, output_dir=output_dir)
    
    solution, stats_obj = solver.solve(search_config=search_config)

    stats = {
        'found': stats_obj.planner_success,
        'plan_length': stats_obj.plan_length,
        'running_time': stats_obj.total_time,
        'search_time': stats_obj.search_time,
        'expanded_nodes': stats_obj.expanded_nodes,
        'generated_nodes': stats_obj.generated_nodes,
        'memory_kb': stats_obj.memory_kb,
        'avg_branching_factor': stats_obj.avg_branching_factor,
        'evaluations': stats_obj.evaluations,
        'instance_dir': output_dir,  
    }
    
    return stats


def run_experiments(n: int = 3, difficulties: List[int] = [5, 10, 20, 30, 50], 
                   instances_per_difficulty: int = 3,
                   run_all_heuristics: bool = True):
    """
    Run complete experimental suite
    
    Args:
        n: puzzle size (n×n)
        difficulties: list of difficulty levels (number of moves)
        instances_per_difficulty: number of instances per difficulty
        run_all_heuristics: if True, run all 4 A* heuristics; if False, only Manhattan + Linear Conflict
    """
    
    print("="*70)
    print(f"EXPERIMENTAL EVALUATION: {n}×{n} n-Puzzle")
    print("="*70)
    
    print(f"\nGenerating test suite...")
    test_suite = generate_test_suite(n, difficulties, instances_per_difficulty)
    print(f"Generated {len(test_suite)} instances")

    results = {
        'astar_manhattan': [],
        'astar_linear_conflict': [],
        'pddl_lmcut': [],
    }
    
    if run_all_heuristics:
        results['astar_misplaced'] = []
        total_algorithms = 4
    else:
        total_algorithms = 3
    
    for i, (difficulty, puzzle) in enumerate(test_suite):
        print(f"\n{'='*70}")
        print(f"Instance {i+1}/{len(test_suite)} - Difficulty: {difficulty} moves")
        print(f"{'='*70}")
        
        algo_count = 0
        
        # A* with Manhattan Distance
        algo_count += 1
        print(f"\n[{algo_count}/{total_algorithms}] Running A* with Manhattan Distance...")
        try:
            stats = run_astar_experiment(puzzle, "manhattan")
            stats['difficulty'] = difficulty
            stats['instance'] = i
            results['astar_manhattan'].append(stats)
            print(f"      ✓ Found: {stats['found']}, Length: {stats['plan_length']}, "
                  f"Expanded: {stats['expanded_nodes']}, Time: {stats['running_time']:.3f}s")
        except Exception as e:
            print(f"      ✗ Error: {e}")
            results['astar_manhattan'].append({'difficulty': difficulty, 'instance': i, 'found': False, 'error': str(e)})
        
        # A* with Misplaced Tiles 
        if run_all_heuristics:
            algo_count += 1
            print(f"[{algo_count}/{total_algorithms}] Running A* with Misplaced Tiles...")
            try:
                stats = run_astar_experiment(puzzle, "misplaced")
                stats['difficulty'] = difficulty
                stats['instance'] = i
                results['astar_misplaced'].append(stats)
                print(f"      ✓ Found: {stats['found']}, Length: {stats['plan_length']}, "
                      f"Expanded: {stats['expanded_nodes']}, Time: {stats['running_time']:.3f}s")
            except Exception as e:
                print(f"      ✗ Error: {e}")
                results['astar_misplaced'].append({'difficulty': difficulty, 'instance': i, 'found': False, 'error': str(e)})
        
        # A* with Linear Conflict
        algo_count += 1
        print(f"[{algo_count}/{total_algorithms}] Running A* with Linear Conflict...")
        try:
            stats = run_astar_experiment(puzzle, "linear_conflict")
            stats['difficulty'] = difficulty
            stats['instance'] = i
            results['astar_linear_conflict'].append(stats)
            print(f"      ✓ Found: {stats['found']}, Length: {stats['plan_length']}, "
                  f"Expanded: {stats['expanded_nodes']}, Time: {stats['running_time']:.3f}s")
        except Exception as e:
            print(f"      ✗ Error: {e}")
            results['astar_linear_conflict'].append({'difficulty': difficulty, 'instance': i, 'found': False, 'error': str(e)})
        
        # PDDL with LM-Cut
        algo_count += 1
        print(f"[{algo_count}/{total_algorithms}] Running PDDL with LM-Cut...")
        try:
            stats = run_pddl_experiment(puzzle, instance_id=i, search_config="astar(lmcut())")
            stats['difficulty'] = difficulty
            stats['instance'] = i
            results['pddl_lmcut'].append(stats)
            print(f"      ✓ Found: {stats['found']}, Length: {stats['plan_length']}, "
                  f"Expanded: {stats['expanded_nodes']}, "
                  f"BF: {stats.get('avg_branching_factor', 'N/A'):.2f}, "
                  f"Time: {stats['running_time']:.3f}s")
        except Exception as e:
            print(f"      ✗ Error: {e}")
            results['pddl_lmcut'].append({'difficulty': difficulty, 'instance': i, 'found': False, 'error': str(e)})
    
    # Save results
    suffix = "all" if run_all_heuristics else "main"
    output_file = f"results_{n}x{n}_puzzle_{suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
  
    results = run_experiments(
        n=3,
        difficulties=[5, 10, 20, 30, 50, 75, 100],
        instances_per_difficulty=3,
        run_all_heuristics=True  # Set to False for quick experiments
    )
    
