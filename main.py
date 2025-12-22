"""
Default: runs automatic batch experiments
With --manual flag: opens interactive menu
"""

import sys
import argparse
from typing import List, Dict
from pathlib import Path

from astar import NPuzzle, astar, ManhattanDistance, MisplacedTiles, LinearConflict, reconstruct_path
from pddl import PDDLSolver
from puzzle_generator import generate_random_instance
from experiment import run_experiments


def print_puzzle(puzzle: List[List[int]]):
    n = len(puzzle)
    width = len(str(n * n - 1))
    
    for row in puzzle:
        print("  ", end="")
        for tile in row:
            if tile == 0:
                print(f"{'_':^{width}} ", end="")
            else:
                print(f"{tile:^{width}} ", end="")
        print()


def print_header(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)
    
def print_headerA(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")


def print_stats(stats: Dict, algorithm: str, is_pddl: bool = False):
    
    # Solution status
    print(f"\n✓ Solution Found: {stats.get('found', False)}")
    print(f"✓ Plan Length: {stats.get('plan_length', 0)} moves")
    
    # Time metrics
    print(f"\n TIME METRICS:")
    print(f"   • Running Time: {stats.get('running_time', 0):.6f} seconds")
    if 'search_time' in stats and stats['search_time'] > 0:
        print(f"   • Search Time: {stats.get('search_time', 0):.6f} seconds")
    if 'total_time' in stats and is_pddl:
        print(f"   • Total Time (incl. translation): {stats.get('total_time', 0):.6f} seconds")
    
    # Node statistics
    print(f"\n NODE STATISTICS:")
    print(f"   • Expanded Nodes: {stats.get('expanded_nodes', 0):,}")
    print(f"   • Generated Nodes: {stats.get('generated_nodes', 0):,}")
    
    if is_pddl and 'evaluations' in stats:
        print(f"   • Evaluations: {stats.get('evaluations', 0):,}")
    
    # Memory statistics for A*
    if not is_pddl:
        print(f"\n MEMORY STATISTICS:")
        if 'max_frontier_size' in stats:
            print(f"   • Max Frontier Size: {stats.get('max_frontier_size', 0):,} nodes")
        if 'max_explored_size' in stats:
            print(f"   • Max Explored Size: {stats.get('max_explored_size', 0):,} nodes")
        if 'max_memory_nodes' in stats:
            print(f"   • Max Total Memory: {stats.get('max_memory_nodes', 0):,} nodes")
    else:
        # PDDL memory
        if 'memory_kb' in stats and stats['memory_kb'] > 0:
            print(f"\n MEMORY STATISTICS:")
            print(f"   • Peak Memory: {stats['memory_kb']:,} KB ({stats['memory_kb']/1024:.2f} MB)")
    
    # Branching factor 
    if 'avg_branching_factor' in stats and stats['avg_branching_factor'] > 0:
        print(f"\n BRANCHING FACTOR:")
        print(f"   • Average: {stats['avg_branching_factor']:.3f}")
        
        if 'max_branching_factor' in stats:
            print(f"   • Maximum: {stats['max_branching_factor']:.3f}")
        if 'min_branching_factor' in stats:
            print(f"   • Minimum: {stats['min_branching_factor']:.3f}")
    
    print("\n" + "=" * 70)


def show_pddl_plan_file(plan_file: Path):
    """Display the content of the PDDL plan file"""
    
    if not plan_file.exists():
        print(f"\n Plan file not found: {plan_file}")
        return
    
    print(f"\n PDDL PLAN FILE CONTENT ({plan_file}):")
    print("=" * 70)
    
    with open(plan_file, 'r') as f:
        content = f.read()
        print(content)
    
    print("=" * 70)


def verify_solution(problem: NPuzzle, actions: List[str], verbose: bool = True) -> bool:

    if verbose:
        print("\n SOLUTION VERIFICATION")
        print("=" * 70)
        print(f"Verifying {len(actions)} actions...\n")
    
    state = problem.initial_state()
    
    for i, action in enumerate(actions):
        # Check if action is valid in current state
        valid_actions = problem.actions(state)
        
        if action not in valid_actions:
            if verbose:
                print(f" INVALID ACTION at step {i+1}/{len(actions)}")
                print(f"   Action: {action}")
                print(f"   Valid actions in this state: {valid_actions}")
                print(f"\n   Current state:")
                print_puzzle([list(row) for row in state])
            return False
        
        state = problem.result(state, action)
        
        if verbose and (i < 5 or i >= len(actions) - 3):
            print(f"Step {i+1}: {action} ✓")
        elif verbose and i == 5:
            print(f"... (steps 6-{len(actions)-3} omitted) ...")
    
    is_goal = problem.goal_test(state)
    
    if verbose:
        print(f"\n  Final state check:")
        if is_goal:
            print("     Final state matches GOAL state!")
        else:
            print("     Final state does NOT match goal!")
            print("\n   Final state reached:")
            print_puzzle([list(row) for row in state])
            print("\n   Expected goal state:")
            print_puzzle([list(row) for row in problem.goal])
        
        print("=" * 70)
    
    return is_goal


def manual_mode():
    """Interactive manual testing mode"""
    
    while True:
        print_header("MANUAL MODE - Single Test")
        
        while True:
            try:
                difficulty = int(input("\n Enter difficulty (number of random moves, e.g., 20): "))
                if difficulty > 0:
                    break
                print("     Difficulty must be positive!")
            except ValueError:
                print("     Please enter a valid number!")

        print("\n Choose algorithm:")
        print("   1. A* - Manhattan Distance")
        print("   2. A* - Misplaced Tiles")
        print("   3. A* - Linear Conflict")
        print("   4. PDDL (astar with lmcut)")
        
        while True:
            try:
                choice = int(input("\nYour choice (1-4): "))
                if 1 <= choice <= 4:
                    break
                print("     Invalid choice! Enter a number between 1 and 4.")
            except ValueError:
                print("     Please enter a valid number!")
        
        print(f"\n Generating puzzle with difficulty {difficulty}...")
        puzzle = generate_random_instance(n=3, num_moves=difficulty)
        
        print("\n Initial puzzle configuration:")
        print_puzzle(puzzle)
        
        print("\n Goal configuration:")
        goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        print_puzzle(goal)
        
        problem = NPuzzle(puzzle)
        is_pddl = (choice == 4)
        
        if is_pddl:
            # PDDL
            print_header("Running PDDL Solver (astar with lmcut)")
            
            solver = PDDLSolver(problem, output_dir=f"./pddl_files/manual_test")
            solution, stats_obj = solver.solve(search_config="astar(lmcut())")
            
            stats = {
                'found': stats_obj.planner_success,
                'plan_length': stats_obj.plan_length,
                'running_time': stats_obj.total_time,
                'search_time': stats_obj.search_time,
                'total_time': stats_obj.total_time,
                'expanded_nodes': stats_obj.expanded_nodes,
                'generated_nodes': stats_obj.generated_nodes,
                'evaluations': stats_obj.evaluations,
                'memory_kb': stats_obj.memory_kb,
                'avg_branching_factor': stats_obj.avg_branching_factor,
            }
            
            algorithm_name = "PDDL (astar with lmcut)"
            actions = solution
            
        else:
            heuristics = {
                1: ("Manhattan Distance", ManhattanDistance()),
                2: ("Misplaced Tiles", MisplacedTiles()),
                3: ("Linear Conflict", LinearConflict())
            }
            
            heuristic_name, heuristic = heuristics[choice]
            algorithm_name = f"A* - {heuristic_name}"
            
            print_header(f"Running {algorithm_name}")
            
            solution_node, stats = astar(problem, heuristic)
            
            if solution_node:
                actions = reconstruct_path(solution_node)
                stats['found'] = True
                stats['plan_length'] = len(actions)
            else:
                actions = None
                stats['found'] = False
                stats['plan_length'] = 0
        
        print_stats(stats, algorithm_name, is_pddl=is_pddl)
        
        if stats['found'] and actions:
            print(f"\n SOLUTION PATH ({len(actions)} moves):")
            print("=" * 70)
            
            max_display = 30
            if len(actions) <= max_display:
                for i in range(0, len(actions), 10):
                    chunk = actions[i:i+10]
                    print(f"   {' → '.join(chunk)}")
            else:
                for i in range(0, 20, 10):
                    chunk = actions[i:i+10]
                    print(f"   {' → '.join(chunk)}")
                print(f"   ... ({len(actions) - 30} moves omitted) ...")
                for i in range(len(actions) - 10, len(actions), 10):
                    chunk = actions[i:i+10]
                    print(f"   {' → '.join(chunk)}")
            
            print("=" * 70)

            if is_pddl:
                show_choice = input("\n Show PDDL plan file content? (y/n): ").strip().lower()
                if show_choice == 'y':
                    plan_file = Path("./pddl_files/manual_test/sas_plan")
                    show_pddl_plan_file(plan_file)
            
            verify_choice = input("\n Verify solution step-by-step? (y/n): ").strip().lower()
            if verify_choice == 'y':
                is_valid = verify_solution(problem, actions, verbose=True)
                
                if is_valid:
                    print("\n SOLUTION IS VALID!")
                else:
                    print("\n SOLUTION IS INVALID!")
                    
        else:
            print("\n No solution found!")
        
        print("\n" + "-" * 70)
        continue_test = input("\n Run another test? (y/n): ").strip().lower()
        if continue_test != 'y':
            break


def automatic_mode():
    """Run automatic batch experiments"""
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  n-PUZZLE SOLVER - Automatic Experiments".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\n Configuration:")
    print("   • Puzzle size: 3×3")
    print("   • Difficulties: [5, 10, 20, 30, 50, 75, 100] moves")
    print("   • Instances per difficulty: 3")
    print("   • Algorithms:")
    print("      - A* with Manhattan Distance")
    print("      - A* with Linear Conflict")
    print("      - PDDL with astar(lmcut)")
    
    print("\n Starting batch experiments...")
    print("   (This may take several minutes)\n")
    
    results = run_experiments(
        n=3,
        difficulties=[5, 10, 20, 30, 50, 75, 100],
        instances_per_difficulty=3
    )
    
    print("\n Experiments completed!")
    print("   Results saved to: results_3x3_puzzle_main.json")
    print("\n Next steps:")
    print("   - Run 'python analyze_results.py' to generate plots")
    print("   - Use 'python main.py --manual' for interactive testing")


def manual_menu():
    """Interactive manual testing menu"""
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  n-PUZZLE SOLVER - Manual Testing Mode".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    while True:
        print("\n MANUAL TESTING MENU")
        print("   1. Run single test")
        print("   2. Exit")
        
        try:
            choice = input("\nYour choice (1-2): ").strip()
            
            if choice == '1':
                manual_mode()
            elif choice == '2':
                print("\n Goodbye!")
                sys.exit(0)
            else:
                print("    Invalid choice! Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='n-Puzzle Solver: A* and PDDL Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 # Run automatic experiments (default)
  python main.py --manual        # Open interactive manual testing mode
  
For analyzing results after experiments:
  python analyze_results.py
        """
    )
    
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Enable manual testing mode with interactive menu'
    )
    
    args = parser.parse_args()
    
    try:
        if args.manual:
            # Manual mode: interactive testing
            manual_menu()
        else:
            # Default: automatic experiments
            automatic_mode()
            
    except KeyboardInterrupt:
        print("\n\n Interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()