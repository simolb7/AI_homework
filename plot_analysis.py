"""
Plot analysis for n-Puzzle experiments
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_results(filename: str) -> Dict:
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def aggregate_by_difficulty(results: List[Dict], is_pddl: bool = False) -> Dict:
    """
    Aggregate metrics by difficulty level
    
    Args:
        results: List of experiment results
        is_pddl: If True, use PDDL-specific memory metrics
    """
    by_difficulty = {}
    
    for result in results:
        if 'error' in result or not result.get('found', False):
            continue
            
        difficulty = result['difficulty']
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {
                'times': [],
                'expanded': [],
                'generated': [],
                'plan_lengths': [],
                'memory': [], 
                'avg_bf': [],
                'max_bf': [],
                'min_bf': [],
                'found': []
            }
        
        by_difficulty[difficulty]['times'].append(result.get('running_time', 0))
        by_difficulty[difficulty]['expanded'].append(result.get('expanded_nodes', 0))
        by_difficulty[difficulty]['generated'].append(result.get('generated_nodes', 0))
        by_difficulty[difficulty]['plan_lengths'].append(result.get('plan_length', 0))
        by_difficulty[difficulty]['found'].append(result.get('found', False))
        
        if is_pddl:
            # PDDL: peak memory in KB
            by_difficulty[difficulty]['memory'].append(result.get('memory_kb', 0))
        else:
            # A*: max_memory_nodes (frontier + explored)
            by_difficulty[difficulty]['memory'].append(result.get('max_memory_nodes', 0))
        
        if 'avg_branching_factor' in result:
            by_difficulty[difficulty]['avg_bf'].append(result['avg_branching_factor'])
        if 'max_branching_factor' in result:
            by_difficulty[difficulty]['max_bf'].append(result['max_branching_factor'])
        if 'min_branching_factor' in result:
            by_difficulty[difficulty]['min_bf'].append(result['min_branching_factor'])
    

    aggregated = {}
    for diff, data in by_difficulty.items():
        if not data['times']:  
            continue
            
        stats = {

            'avg_time': np.mean(data['times']),
            'std_time': np.std(data['times']),
            'min_time': np.min(data['times']),
            'max_time': np.max(data['times']),
            
            'avg_expanded': np.mean(data['expanded']),
            'std_expanded': np.std(data['expanded']),
            'avg_generated': np.mean(data['generated']),
            'std_generated': np.std(data['generated']),

            'avg_memory': np.mean(data['memory']),
            'std_memory': np.std(data['memory']),
            'max_memory': np.max(data['memory']),
 
            'avg_plan_length': np.mean(data['plan_lengths']),
            'std_plan_length': np.std(data['plan_lengths']),

            'success_rate': sum(data['found']) / len(data['found']) * 100,
            'num_instances': len(data['times'])
        }

        if data['avg_bf']:
            stats['avg_bf'] = np.mean(data['avg_bf'])
            stats['std_bf'] = np.std(data['avg_bf'])
        if data['max_bf']:
            stats['max_bf'] = np.mean(data['max_bf'])
        if data['min_bf']:
            stats['min_bf'] = np.mean(data['min_bf'])
        
        aggregated[diff] = stats
    
    return aggregated


def create_homework_plots(results: Dict, output_dir: str = "plots"):
    """
    Create plots optimized for homework report:
    - Running time vs difficulty
    - Expanded nodes vs difficulty
    - Memory usage vs difficulty (A*: nodes, PDDL: KB)
    - Branching factor analysis (avg/max/min)
    - Solution quality check
    """
    
    Path(output_dir).mkdir(exist_ok=True)

    methods_data = {}

    if 'astar_manhattan' in results:
        methods_data['A* Manhattan'] = aggregate_by_difficulty(results['astar_manhattan'], is_pddl=False)
    if 'astar_linear_conflict' in results:
        methods_data['A* Linear Conflict'] = aggregate_by_difficulty(results['astar_linear_conflict'], is_pddl=False)
    if 'pddl_lmcut' in results:
        methods_data['PDDL LM-Cut'] = aggregate_by_difficulty(results['pddl_lmcut'], is_pddl=True)
    if 'astar_misplaced' in results:
        methods_data['A* Misplaced'] = aggregate_by_difficulty(results['astar_misplaced'], is_pddl=False)

    all_difficulties = sorted(set().union(*[set(data.keys()) for data in methods_data.values()]))

    colors = {
        'A* Manhattan': '#2E86AB', 
        'A* Linear Conflict': '#A23B72', 
        'PDDL LM-Cut': '#F18F01',
        'A* Misplaced': '#6C757D'       # Gray
    }
    markers = {
        'A* Manhattan': 'o', 
        'A* Linear Conflict': 's', 
        'PDDL LM-Cut': '^',
        'A* Misplaced': 'v'             # Triangle down
    }
    

    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('n-Puzzle: A* vs PDDL Performance Comparison', 
                  fontsize=15, fontweight='bold', y=0.995)
    
    # --- Plot 1.1: Running Time ---
    ax = axes[0, 0]
    for method_name, data in methods_data.items():
        if not data:
            continue
        difficulties = sorted(data.keys())
        times = [data[d]['avg_time'] for d in difficulties]
        stds = [data[d]['std_time'] for d in difficulties]
        
        ax.errorbar(difficulties, times, yerr=stds, 
                   marker=markers[method_name], label=method_name, 
                   linewidth=2, markersize=7, capsize=4, 
                   color=colors[method_name], alpha=0.85)
    
    ax.set_xlabel('Difficulty (moves from goal)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Running Time (seconds)', fontsize=10, fontweight='bold')
    ax.set_title('(a) Running Time', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # --- Plot 1.2: Expanded Nodes ---
    ax = axes[0, 1]
    for method_name, data in methods_data.items():
        if not data:
            continue
        difficulties = sorted(data.keys())
        expanded = [data[d]['avg_expanded'] for d in difficulties]
        stds = [data[d]['std_expanded'] for d in difficulties]
        
        ax.errorbar(difficulties, expanded, yerr=stds, 
                   marker=markers[method_name], label=method_name,
                   linewidth=2, markersize=7, capsize=4,
                   color=colors[method_name], alpha=0.85)
    
    ax.set_xlabel('Difficulty (moves from goal)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Expanded Nodes', fontsize=10, fontweight='bold')
    ax.set_title('(b) Search Effort (Expanded Nodes)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}'))
    
    # --- Plot 1.3: Memory Usage (DUAL Y-AXIS for A* vs PDDL) ---
    ax = axes[1, 0]
    ax2 = ax.twinx()  
    
    for method_name, data in methods_data.items():
        if not data:
            continue
        difficulties = sorted(data.keys())
        memory = [data[d]['avg_memory'] for d in difficulties]
        
        if 'PDDL' in method_name:
            # PDDL: KB (right axis)
            ax2.plot(difficulties, memory, 
                   marker=markers[method_name], label=method_name,
                   linewidth=2, markersize=7,
                   color=colors[method_name], alpha=0.85, linestyle='--')
        else:
            # A*: nodes (left axis)
            ax.plot(difficulties, memory, 
                   marker=markers[method_name], label=method_name,
                   linewidth=2, markersize=7,
                   color=colors[method_name], alpha=0.85)
    
    ax.set_xlabel('Difficulty (moves from goal)', fontsize=10, fontweight='bold')
    ax.set_ylabel('A* Memory (nodes in frontier+explored)', fontsize=10, fontweight='bold', color='black')
    ax2.set_ylabel('PDDL Memory (KB)', fontsize=10, fontweight='bold', color='#F18F01')
    ax.set_title('(c) Memory Usage', fontsize=11, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='#F18F01')
    
    # --- Plot 1.4: Solution Quality ---
    ax = axes[1, 1]
    for method_name, data in methods_data.items():
        if not data:
            continue
        difficulties = sorted(data.keys())
        lengths = [data[d]['avg_plan_length'] for d in difficulties]
        
        ax.plot(difficulties, lengths, 
               marker=markers[method_name], label=method_name,
               linewidth=2, markersize=7,
               color=colors[method_name], alpha=0.85)
    
    # Optimal reference line
    ax.plot(all_difficulties, all_difficulties, 'k--', 
           label='Optimal', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Difficulty (moves from goal)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Solution Length (moves)', fontsize=10, fontweight='bold')
    ax.set_title('(d) Solution Quality (Optimality)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    fig1.savefig(Path(output_dir) / 'main_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: plots/main_performance.png")
    
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig2.suptitle('Branching Factor Analysis (Search Efficiency)', 
                  fontsize=14, fontweight='bold')
    
    for method_name, data in methods_data.items():
        if not data:
            continue
        difficulties = sorted(data.keys())

        avg_bf = [data[d].get('avg_bf', 0) for d in difficulties]
        
        if not any(avg_bf): 
            continue

        ax.plot(difficulties, avg_bf, 
               marker=markers[method_name], label=method_name,
               linewidth=2.5, markersize=8,
               color=colors[method_name], alpha=0.9)
    
    ax.set_xlabel('Difficulty (moves from goal)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Branching Factor', fontsize=11, fontweight='bold')
    ax.set_title('Average Branching Factor', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.text(0.98, 0.02, 
            'Lower values indicate more efficient search\n(fewer duplicate states generated)', 
            transform=ax.transAxes, 
            fontsize=9, 
            ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    fig2.savefig(Path(output_dir) / 'branching_factor.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: plots/branching_factor.png")

    astar_methods = {k: v for k, v in methods_data.items() if k.startswith('A*')}
    
    if len(astar_methods) >= 3:
        print(f"✓ Detected {len(astar_methods)} A* heuristics - creating comparison plot...")
        
        fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig3.suptitle('A* Heuristics Comparison', 
                      fontsize=14, fontweight='bold')
        
        # Plot 3.1: Expanded Nodes Comparison
        ax = axes[0]
        for method_name, data in astar_methods.items():
            if not data:
                continue
            difficulties = sorted(data.keys())
            expanded = [data[d]['avg_expanded'] for d in difficulties]
            
            ax.plot(difficulties, expanded, 
                   marker=markers[method_name], label=method_name,
                   linewidth=2, markersize=7,
                   color=colors[method_name], alpha=0.85)
        
        ax.set_xlabel('Difficulty (moves from goal)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Expanded Nodes', fontsize=10, fontweight='bold')
        ax.set_title('Expanded Nodes by Heuristic', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}'))
        
        # Plot 3.2: Running Time Comparison
        ax = axes[1]
        for method_name, data in astar_methods.items():
            if not data:
                continue
            difficulties = sorted(data.keys())
            times = [data[d]['avg_time'] for d in difficulties]
            
            ax.plot(difficulties, times, 
                   marker=markers[method_name], label=method_name,
                   linewidth=2, markersize=7,
                   color=colors[method_name], alpha=0.85)
        
        ax.set_xlabel('Difficulty (moves from goal)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Running Time (seconds)', fontsize=10, fontweight='bold')
        ax.set_title('Running Time by Heuristic', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        fig3.savefig(Path(output_dir) / 'astar_heuristics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: plots/astar_heuristics_comparison.png")
    
    plt.show()


def print_summary_table(results: Dict):
    """Print concise summary table for report"""
    
    print("\n" + "="*110)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*110)
    
    for method_name, method_results in results.items():
        print(f"\n{method_name.upper().replace('_', ' ')}")
        print("-"*110)
        
        is_pddl = 'pddl' in method_name.lower()
        aggregated = aggregate_by_difficulty(method_results, is_pddl=is_pddl)
        
        if not aggregated:
            print("  No successful runs.")
            continue

        if is_pddl:
            print(f"{'Diff':<6} {'Time(s)':<10} {'Expanded':<11} {'Generated':<11} "
                  f"{'Avg BF':<9} {'Mem(KB)':<10} {'Length':<8}")
        else:
            print(f"{'Diff':<6} {'Time(s)':<10} {'Expanded':<11} {'Generated':<11} "
                  f"{'Avg BF':<9} {'Mem(nodes)':<12} {'Length':<8}")
        print("-"*110)
        
        # Table rows
        for difficulty in sorted(aggregated.keys()):
            d = aggregated[difficulty]
            print(f"{difficulty:<6} "
                  f"{d['avg_time']:<10.4f} "
                  f"{d['avg_expanded']:<11.1f} "
                  f"{d['avg_generated']:<11.1f} "
                  f"{d.get('avg_bf', 0):<9.2f} "
                  f"{d['avg_memory']:<12.1f} "
                  f"{d['avg_plan_length']:<8.1f}")
    
    print("="*110)

    print("\n" + "="*110)
    print("OVERALL COMPARISON (averaged across all difficulties)")
    print("="*110)
    print(f"{'Method':<25} {'Avg Time':<12} {'Avg Expanded':<14} "
          f"{'Avg BF':<10} {'Success %':<12}")
    print("-"*110)
    
    for method_name, method_results in results.items():
        is_pddl = 'pddl' in method_name.lower()
        aggregated = aggregate_by_difficulty(method_results, is_pddl=is_pddl)
        if not aggregated:
            continue
            
        avg_time = np.mean([d['avg_time'] for d in aggregated.values()])
        avg_expanded = np.mean([d['avg_expanded'] for d in aggregated.values()])
        avg_bf = np.mean([d.get('avg_bf', 0) for d in aggregated.values()])
        avg_success = np.mean([d['success_rate'] for d in aggregated.values()])
        
        print(f"{method_name.replace('_', ' '):<25} "
              f"{avg_time:<12.4f} "
              f"{avg_expanded:<14.1f} "
              f"{avg_bf:<10.2f} "
              f"{avg_success:<12.1f}")
    
    print("="*110 + "\n")


def main():
    """Main analysis pipeline"""
    
    result_file = [
        "results_3x3_puzzle_all.json",   # All heuristics
    ]
    
    results_file = None
    for file in result_file:
        if Path(file).exists():
            results_file = file
            break
    
    if not results_file:
        print("\n   Run experiments first: python experiment.py")
        return
    
    print(f"\n{'='*110}")
    print(f"n-PUZZLE EXPERIMENTAL ANALYSIS")
    print(f"Using: {results_file}")
    print(f"{'='*110}\n")
    
    # Load and analyze
    results = load_results(results_file)
    
    # Detect mode
    num_methods = len([k for k in results.keys() if results[k]])

    # Print statistics
    print_summary_table(results)
    
    # Generate plots
    print("GENERATING PLOTS...")
    
    create_homework_plots(results)
    
    print(f"\n{'='*110}")
    print("✓ ANALYSIS COMPLETE!")
    print("\nGenerated files:")
    print("  plots/main_performance.png        - Main metrics (time, nodes, memory, quality)")
    print("  plots/branching_factor.png        - Branching factor analysis")
    print("  plots/astar_heuristics_comparison.png - A* heuristics comparison")
    print(f"{'='*110}\n")


if __name__ == "__main__":
    main()