"""
PDDL Planning Solver for n-Puzzle Problem
"""

import subprocess
import os
import re
import time
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PlanningStats:
    """Statistics from PDDL planner"""
    plan_length: int = 0
    running_time: float = 0.0
    expanded_nodes: int = 0  # From planner output
    generated_nodes: int = 0  # From planner output
    search_time: float = 0.0  # Time spent in search
    total_time: float = 0.0  # Including translation
    planner_success: bool = False
    
    # Planner-specific metrics
    evaluations: int = 0  # Total state evaluations
    memory_kb: int = 0  # Peak memory usage
    
    avg_branching_factor: float = 0.0


class PDDLSolver:
    
    def __init__(self, problem, output_dir: str = "./pddl_files", fd_path: str = "./fast-downward/fast-downward.py"):
        """
        Initialize PDDL solver with NPuzzle instance 
        """
        self.problem = problem
        self.n = problem.n
        self.fd_path = str(Path(fd_path).resolve())
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.domain_file = self.output_dir / "domain.pddl"
        self.problem_file = self.output_dir / "problem.pddl"
        self.plan_file = self.output_dir / "sas_plan"
    
    def generate_domain_file(self):
        """Generate PDDL domain file for n-puzzle"""
        
        domain = """(define (domain npuzzle)
            (:requirements :strips :typing)
            
            (:types
                tile position
            )
            
            (:predicates
                (at ?t - tile ?p - position)
                (blank ?p - position)
                (adjacent ?p1 ?p2 - position)
            )
            
            (:action move-tile
                :parameters (?t - tile ?from ?to - position)
                :precondition (and
                (at ?t ?from)
                (blank ?to)
                (adjacent ?from ?to)
                )
                :effect (and
                (at ?t ?to)
                (blank ?from)
                (not (at ?t ?from))
                (not (blank ?to))
                )
            )
            )
            """
        
        with open(self.domain_file, 'w') as f:
            f.write(domain)
    
    def generate_problem_file(self):
        """Generate PDDL problem file from current problem instance"""
        
        # Generate positions
        positions = [f"pos{i}{j}" for i in range(self.n) for j in range(self.n)]
        
        # Generate tiles (excluding blank/0)
        tiles = [f"tile{i}" for i in range(1, self.n * self.n)]
        
        # Generate adjacency (all directions)
        adjacencies = set()
        for i in range(self.n):
            for j in range(self.n):
                if j < self.n - 1:  # Right
                    adjacencies.add(f"(adjacent pos{i}{j} pos{i}{j+1})")
                    adjacencies.add(f"(adjacent pos{i}{j+1} pos{i}{j})")
                if i < self.n - 1:  # Down
                    adjacencies.add(f"(adjacent pos{i}{j} pos{i+1}{j})")
                    adjacencies.add(f"(adjacent pos{i+1}{j} pos{i}{j})")
        adjacencies = sorted(list(adjacencies))
        
        # Initial state from problem
        initial_state = []
        for i in range(self.n):
            for j in range(self.n):
                tile = self.problem.initial[i][j]
                if tile == 0:
                    initial_state.append(f"(blank pos{i}{j})")
                else:
                    initial_state.append(f"(at tile{tile} pos{i}{j})")
        
        # Goal state from problem
        goal_state = []
        for i in range(self.n):
            for j in range(self.n):
                tile = self.problem.goal[i][j]
                if tile != 0:
                    goal_state.append(f"(at tile{tile} pos{i}{j})")
        
        # Build problem file
        problem_pddl = f"""(define (problem npuzzle-instance)
            (:domain npuzzle)
            
            (:objects
                {' '.join(tiles)} - tile
                {' '.join(positions)} - position
            )
            
            (:init
                {chr(10).join('    ' + adj for adj in adjacencies)}
                {chr(10).join('    ' + init for init in initial_state)}
            )
            
            (:goal
                (and
                {chr(10).join('      ' + goal for goal in goal_state)}
                )
            )
            )
            """
        
        with open(self.problem_file, 'w') as f:
            f.write(problem_pddl)
    
    def call_planner(self, search_config: str = "astar(lmcut())") -> PlanningStats:
        """
        Call Fast Downward planner and collect statistics
        """
        
        stats = PlanningStats()
        
        # Clean up old plan files in varie posizioni
        if self.plan_file.exists():
            os.remove(self.plan_file)
        
        cwd_plan = Path.cwd() / "sas_plan"
        if cwd_plan.exists():
            os.remove(cwd_plan)
        
        try:
            cmd = [
                "python3",
                self.fd_path,
                str(self.domain_file),
                str(self.problem_file),
                "--search", search_config
            ]
            
            print(f"\nRunning: {' '.join(cmd)}\n")
            
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            stats.total_time = time.time() - start_time
            
            # Parse Fast Downward output for statistics
            output = result.stdout + result.stderr
            stats = self._parse_planner_output(output, stats)
            
            print(f"Return code: {result.returncode}")
            
            if cwd_plan.exists():
                print(f"✓ Plan found at: {cwd_plan}")

                import shutil
                shutil.move(str(cwd_plan), str(self.plan_file))
                print(f"  Moved to: {self.plan_file}")
                stats.planner_success = True
            else:
                print(f"✗ Plan not found at: {cwd_plan}")
                
                if "Solution found" in output:
                    print("  Search says 'Solution found' but no plan file!")
                elif result.returncode != 0:
                    print(f"  Search failed with return code: {result.returncode}")
                    print(f"  Last output:\n{output[-500:]}")
            
            return stats
            
        except FileNotFoundError as e:
            print(f"ERROR: Fast Downward not found at {self.fd_path}")
            stats.planner_success = False
            return stats
            
        except subprocess.TimeoutExpired:
            print("ERROR: Planner timeout (300 seconds)")
            stats.planner_success = False
            return stats
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            stats.planner_success = False
            return stats
    
    def _parse_planner_output(self, output: str, stats: PlanningStats) -> PlanningStats:
        """Extract statistics from Fast Downward output"""
        
        # Search time
        match = re.search(r'Search time: ([\d.]+)s', output)
        if match:
            stats.search_time = float(match.group(1))
        
        # Total time
        match = re.search(r'Total time: ([\d.]+)s', output)
        if match:
            stats.running_time = float(match.group(1))
        
        # Expanded nodes
        match = re.search(r'Expanded (\d+) state\(s\)', output)
        if match:
            stats.expanded_nodes = int(match.group(1))
        
        # Evaluations (generated nodes equivalent)
        match = re.search(r'Evaluations: (\d+)', output)
        if match:
            stats.evaluations = int(match.group(1))
            stats.generated_nodes = stats.evaluations
        
        # Generated states
        match = re.search(r'Generated (\d+) state\(s\)', output)
        if match:
            stats.generated_nodes = int(match.group(1))
        
        # Peak memory
        match = re.search(r'Peak memory: (\d+) KB', output)
        if match:
            stats.memory_kb = int(match.group(1))
        
        # Plan cost/length
        match = re.search(r'Plan cost: (\d+)', output)
        if match:
            stats.plan_length = int(match.group(1))
        if stats.expanded_nodes > 0 and stats.generated_nodes > 0:
            stats.avg_branching_factor = stats.generated_nodes / stats.expanded_nodes
            
        return stats
    
    def parse_plan(self) -> Optional[List[str]]:
        """Parse PDDL plan and convert to action list"""
        
        if not self.plan_file.exists():
            return None
        
        actions = []
        
        with open(self.plan_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';') or line.startswith('cost'):
                    continue
                
                match = re.search(r'\(move-tile\s+tile(\d+)\s+pos(\d)(\d)\s+pos(\d)(\d)\)', line)
                if match:
                    tile = int(match.group(1))
                    from_row, from_col = int(match.group(2)), int(match.group(3))
                    to_row, to_col = int(match.group(4)), int(match.group(5))
                    
                    # Determine tile direction
                    if to_row < from_row:
                        tile_direction = "UP"
                    elif to_row > from_row:
                        tile_direction = "DOWN"
                    elif to_col < from_col:
                        tile_direction = "LEFT"
                    else:
                        tile_direction = "RIGHT"
                    
                    # inverting direction for blank space
                    direction_map = {
                        "UP": "DOWN",
                        "DOWN": "UP",
                        "LEFT": "RIGHT",
                        "RIGHT": "LEFT"
                    }
                    
                    blank_direction = direction_map[tile_direction]
                    actions.append(blank_direction)
        
        return actions
    
    def solve(self, search_config: str = "astar(lmcut())") -> Tuple[Optional[List[str]], PlanningStats]:
        """
        Complete PDDL solving pipeline
        Returns: (action_list, statistics)
        """
        
        # Generate PDDL files
        self.generate_domain_file()
        self.generate_problem_file()
        
        # Call planner and collect stats
        stats = self.call_planner(search_config)
        
        if not stats.planner_success:
            return None, stats
        
        # Parse plan
        actions = self.parse_plan()
        if actions:
            stats.plan_length = len(actions)
        
        return actions, stats


def verify_pddl_solution(problem, actions: List[str]) -> bool:
    """Verify PDDL solution using the same problem instance"""
    state = problem.initial_state()
    
    for action in actions:
        if action not in problem.actions(state):
            return False
        state = problem.result(state, action)
    
    return problem.goal_test(state)