from astar import NPuzzle, astar, ManhattanDistance, LinearConflict
from pddl import PDDLSolver, verify_pddl_solution


# Example usage showing integration with A*
if __name__ == "__main__":
    # Import NPuzzle from your A* implementation

    
    # You would import: from astar import NPuzzle, astar, ManhattanDistance
    # For demo, assuming NPuzzle is available
    
    print("=" * 70)
    print("PDDL vs A* Comparison on n-Puzzle")
    print("=" * 70)
    
    # Test instance
    initial = [
        [1, 2, 3],
        [4, 0, 5],
        [6, 7, 8]
    ]
    
    print("\nTest puzzle:")
    for row in initial:
        print(row)
    
    problem = NPuzzle(initial)
    
    print("\n" + "=" * 70)
    print("METHOD 1: A* with Manhattan Distance")
    print("=" * 70)
    
    heuristic = ManhattanDistance()
    solution_astar, stats_astar = astar(problem, heuristic)
    
    if solution_astar:
        from astar import reconstruct_path
        path = reconstruct_path(solution_astar)
        print(f"✓ Solution found")
        print(f"  Plan length:     {len(path)}")
        print(f"  Expanded nodes:  {stats_astar['expanded_nodes']}")
        print(f"  Generated nodes: {stats_astar['generated_nodes']}")
        print(f"  Max frontier:    {stats_astar['max_frontier_size']}")
    
    
    print("\n" + "=" * 70)
    print("METHOD 2: A* with Linear Conflict")
    print("=" * 70)
    
    heuristic = LinearConflict()
    solution_astar, stats_astar = astar(problem, heuristic)
    
    if solution_astar:
        from astar import reconstruct_path
        path = reconstruct_path(solution_astar)
        print(f"✓ Solution found")
        print(f"  Plan length:     {len(path)}")
        print(f"  Expanded nodes:  {stats_astar['expanded_nodes']}")
        print(f"  Generated nodes: {stats_astar['generated_nodes']}")
        print(f"  Max frontier:    {stats_astar['max_frontier_size']}")
    
    print("\n" + "=" * 70)
    print("METHOD 3: PDDL Planning (A* with LM-Cut)")
    print("=" * 70)
    
    pddl_solver = PDDLSolver(problem, fd_path="./fast-downward/fast-downward.py")
    solution_pddl, stats_pddl = pddl_solver.solve(search_config="astar(lmcut())")
    
    if solution_pddl:
        print(f"✓ Solution found")
        print(f"  Plan length:     {stats_pddl.plan_length}")
        print(f"  Expanded nodes:  {stats_pddl.expanded_nodes}")
        print(f"  Generated nodes: {stats_pddl.generated_nodes}")
        print(f"  Search time:     {stats_pddl.search_time:.3f}s")
        print(f"  Total time:      {stats_pddl.total_time:.3f}s")
        print(f"  Peak memory:     {stats_pddl.memory_kb} KB")
        
        # Verify solution
        if verify_pddl_solution(problem, solution_pddl):
            print("  ✓ Solution verified!")
    else:
        print("✗ No solution found")
        print("  Make sure Fast Downward is installed:")
        print("  pip install fast-downward")
    
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    if solution_astar and solution_pddl:
        print(f"\n{'Metric':<20} {'A* (Manhattan)':<20} {'PDDL (LM-Cut)':<20}")
        print("-" * 70)
        print(f"{'Plan Length':<20} {len(path):<20} {stats_pddl.plan_length:<20}")
        print(f"{'Expanded Nodes':<20} {stats_astar['expanded_nodes']:<20} {stats_pddl.expanded_nodes:<20}")
        print(f"{'Generated Nodes':<20} {stats_astar['generated_nodes']:<20} {stats_pddl.generated_nodes:<20}")
        
        print("\n💡 Insights:")
        print("- Both should find optimal solutions (same plan length)")
        print("- A* with domain-specific heuristic (Manhattan) typically expands fewer nodes")
        print("- PDDL planner uses general-purpose heuristic (LM-Cut)")
        print("- PDDL has overhead from translation but is more general")