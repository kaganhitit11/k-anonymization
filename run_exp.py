import subprocess
import time
import re
import csv
import numpy as np
from datetime import datetime

# Configuration
K_VALUES = [4, 8, 16, 32, 64, 128, 256, 512]
SEEDS = [42, 123, 456]  # Three seeds for all algorithms
NUM_SHUFFLED_DATASETS = 3  # Number of shuffled datasets for clustering

SKELETON_FILE = "skeleton_oguz.py"
DGH_FOLDER = "DGHs"
RAW_DATASET = "adult-hw1.csv"

def shuffle_dataset(input_file, output_file, seed):
    """
    Shuffle a dataset and save it to a new file.
    
    Args:
        input_file: path to input CSV file
        output_file: path to output CSV file
        seed: random seed for shuffling
    """
    # Read the dataset
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        fieldnames = reader.fieldnames
    
    # Shuffle the data
    np.random.seed(seed)
    np.random.shuffle(data)
    
    # Write shuffled data
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def run_anonymizer(algorithm, k, seed=None, input_file=None):
    """
    Run the anonymizer and measure execution time, MD cost, and LM cost.
    
    Args:
        algorithm: 'random', 'clustering', or 'topdown'
        k: k-anonymity parameter
        seed: seed for random anonymizer (optional)
        input_file: custom input file (optional, defaults to RAW_DATASET)
    
    Returns:
        tuple: (execution_time, md_cost, lm_cost, success)
    """
    if input_file is None:
        input_file = RAW_DATASET
    
    output_file = f"temp_output_{algorithm}_k{k}.csv"
    
    # Build command
    if algorithm == "random" and seed is not None:
        cmd = [
            "python3", SKELETON_FILE, algorithm, DGH_FOLDER, 
            input_file, output_file, str(k), str(seed)
        ]
    else:
        cmd = [
            "python3", SKELETON_FILE, algorithm, DGH_FOLDER, 
            input_file, output_file, str(k)
        ]
    
    # Measure execution time
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
        execution_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"Error running {algorithm} with k={k}: {result.stderr}")
            return execution_time, -1, -1, False
        
        # Parse output for MD and LM costs
        output = result.stdout
        md_match = re.search(r'Cost_MD:\s*([\d.]+)', output)
        lm_match = re.search(r'Cost_LM:\s*([\d.]+)', output)
        
        md_cost = float(md_match.group(1)) if md_match else -1
        lm_cost = float(lm_match.group(1)) if lm_match else -1
        
        return execution_time, md_cost, lm_cost, True
    
    except subprocess.TimeoutExpired:
        print(f"Timeout running {algorithm} with k={k}")
        return -1, -1, -1, False
    except Exception as e:
        print(f"Exception running {algorithm} with k={k}: {e}")
        return -1, -1, -1, False


def run_experiments():
    """
    Run all experiments and collect results.
    """
    results = {
        'random': {},
        'clustering': {},
        'topdown': {}
    }
    
    print("=" * 80)
    print("Starting Experiments")
    print("=" * 80)
    print(f"Seeds used: {SEEDS}")
    print("=" * 80)
    
    # Random Anonymizer - run with 3 seeds and average
    print("\n### Running Random Anonymizer ###")
    print("Running 3 times with 3 different seeds")
    for k in K_VALUES:
        print(f"\n  k={k}...")
        times, md_costs, lm_costs = [], [], []
        
        for seed in SEEDS:
            print(f"    Seed {seed}...", end=" ")
            exec_time, md, lm, success = run_anonymizer('random', k, seed)
            if success:
                times.append(exec_time)
                md_costs.append(md)
                lm_costs.append(lm)
                print(f"time={exec_time:.2f}s, MD={md:.2f}, LM={lm:.2f}")
            else:
                print("FAILED")
        
        if times:
            avg_time = sum(times) / len(times)
            avg_md = sum(md_costs) / len(md_costs)
            avg_lm = sum(lm_costs) / len(lm_costs)
            results['random'][k] = {
                'time': avg_time,
                'md': avg_md,
                'lm': avg_lm,
                'runs': len(times)
            }
            print(f"    >>> Average: time={avg_time:.2f}s, MD={avg_md:.2f}, LM={avg_lm:.2f}")
    
    # Clustering Anonymizer - create 3 shuffled datasets and run on each
    print("\n### Running Clustering Anonymizer ###")
    print("Creating 3 shuffled datasets and running clustering on each")
    
    # Create shuffled datasets
    shuffled_files = []
    for i, seed in enumerate(SEEDS):
        shuffled_file = f"temp_shuffled_dataset_{seed}.csv"
        print(f"  Creating shuffled dataset {i+1} with seed {seed}...")
        shuffle_dataset(RAW_DATASET, shuffled_file, seed)
        shuffled_files.append(shuffled_file)
    
    for k in K_VALUES:
        print(f"\n  k={k}...")
        times, md_costs, lm_costs = [], [], []
        
        for i, (shuffled_file, seed) in enumerate(zip(shuffled_files, SEEDS)):
            print(f"    Shuffled dataset {i+1} (seed {seed})...", end=" ")
            exec_time, md, lm, success = run_anonymizer('clustering', k, input_file=shuffled_file)
            if success:
                times.append(exec_time)
                md_costs.append(md)
                lm_costs.append(lm)
                print(f"time={exec_time:.2f}s, MD={md:.2f}, LM={lm:.2f}")
            else:
                print("FAILED")
        
        if times:
            avg_time = sum(times) / len(times)
            avg_md = sum(md_costs) / len(md_costs)
            avg_lm = sum(lm_costs) / len(lm_costs)
            results['clustering'][k] = {
                'time': avg_time,
                'md': avg_md,
                'lm': avg_lm,
                'runs': len(times)
            }
            print(f"    >>> Average: time={avg_time:.2f}s, MD={avg_md:.2f}, LM={avg_lm:.2f}")
    
    # Topdown Anonymizer - run 3 times for consistency check
    print("\n### Running Topdown Anonymizer ###")
    print("Running 3 times for consistency check")
    for k in K_VALUES:
        print(f"\n  k={k}...")
        times, md_costs, lm_costs = [], [], []
        
        for run_num in range(3):
            print(f"    Run {run_num+1}...", end=" ")
            exec_time, md, lm, success = run_anonymizer('topdown', k)
            if success:
                times.append(exec_time)
                md_costs.append(md)
                lm_costs.append(lm)
                print(f"time={exec_time:.2f}s, MD={md:.2f}, LM={lm:.2f}")
            else:
                print("FAILED")
        
        if times:
            avg_time = sum(times) / len(times)
            avg_md = sum(md_costs) / len(md_costs)
            avg_lm = sum(lm_costs) / len(lm_costs)
            results['topdown'][k] = {
                'time': avg_time,
                'md': avg_md,
                'lm': avg_lm,
                'runs': len(times)
            }
            print(f"    >>> Average: time={avg_time:.2f}s, MD={avg_md:.2f}, LM={avg_lm:.2f}")
    
    return results


def write_results_to_markdown(results, output_file="experiment_results.md"):
    """
    Write experiment results to a markdown file.
    """
    with open(output_file, 'w') as f:
        f.write("# K-Anonymity Experiment Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Dataset:** {RAW_DATASET}\n\n")
        f.write(f"**K values tested:** {', '.join(map(str, K_VALUES))}\n\n")
        f.write("---\n\n")
        
        # Summary table for all algorithms
        f.write("## Summary: All Algorithms\n\n")
        f.write("### Execution Time (seconds)\n\n")
        f.write("| k | Random | Clustering | Topdown |\n")
        f.write("|---|--------|------------|----------|\n")
        for k in K_VALUES:
            random_time = results['random'].get(k, {}).get('time', 'N/A')
            clustering_time = results['clustering'].get(k, {}).get('time', 'N/A')
            topdown_time = results['topdown'].get(k, {}).get('time', 'N/A')
            
            random_str = f"{random_time:.2f}" if isinstance(random_time, float) else random_time
            clustering_str = f"{clustering_time:.2f}" if isinstance(clustering_time, float) else clustering_time
            topdown_str = f"{topdown_time:.2f}" if isinstance(topdown_time, float) else topdown_time
            
            f.write(f"| {k} | {random_str} | {clustering_str} | {topdown_str} |\n")
        
        f.write("\n### MD Cost (Distortion Metric)\n\n")
        f.write("| k | Random | Clustering | Topdown |\n")
        f.write("|---|--------|------------|----------|\n")
        for k in K_VALUES:
            random_md = results['random'].get(k, {}).get('md', 'N/A')
            clustering_md = results['clustering'].get(k, {}).get('md', 'N/A')
            topdown_md = results['topdown'].get(k, {}).get('md', 'N/A')
            
            random_str = f"{random_md:.2f}" if isinstance(random_md, float) else random_md
            clustering_str = f"{clustering_md:.2f}" if isinstance(clustering_md, float) else clustering_md
            topdown_str = f"{topdown_md:.2f}" if isinstance(topdown_md, float) else topdown_md
            
            f.write(f"| {k} | {random_str} | {clustering_str} | {topdown_str} |\n")
        
        f.write("\n### LM Cost (Loss Metric)\n\n")
        f.write("| k | Random | Clustering | Topdown |\n")
        f.write("|---|--------|------------|----------|\n")
        for k in K_VALUES:
            random_lm = results['random'].get(k, {}).get('lm', 'N/A')
            clustering_lm = results['clustering'].get(k, {}).get('lm', 'N/A')
            topdown_lm = results['topdown'].get(k, {}).get('lm', 'N/A')
            
            random_str = f"{random_lm:.2f}" if isinstance(random_lm, float) else random_lm
            clustering_str = f"{clustering_lm:.2f}" if isinstance(clustering_lm, float) else clustering_lm
            topdown_str = f"{topdown_lm:.2f}" if isinstance(topdown_lm, float) else topdown_lm
            
            f.write(f"| {k} | {random_str} | {clustering_str} | {topdown_str} |\n")
        
        # Detailed results for each algorithm
        f.write("\n---\n\n")
        f.write("## Detailed Results by Algorithm\n\n")
        
        for algo in ['random', 'clustering', 'topdown']:
            f.write(f"### {algo.capitalize()} Anonymizer\n\n")
            f.write("| k | Time (s) | MD Cost | LM Cost | # Runs |\n")
            f.write("|---|----------|---------|---------|--------|\n")
            
            for k in K_VALUES:
                if k in results[algo]:
                    data = results[algo][k]
                    f.write(f"| {k} | {data['time']:.2f} | {data['md']:.2f} | "
                           f"{data['lm']:.2f} | {data['runs']} |\n")
                else:
                    f.write(f"| {k} | N/A | N/A | N/A | 0 |\n")
            f.write("\n")
        
        # Notes section
        f.write("---\n\n")
        f.write("## Experimental Setup\n\n")
        f.write(f"- **Seeds used**: {SEEDS}\n")
        f.write(f"- **Random Anonymizer**: Run 3 times with 3 different seeds ({SEEDS}), results averaged\n")
        f.write(f"- **Clustering Anonymizer**: Dataset shuffled 3 times using 3 different seeds ({SEEDS}), "
                f"clustering run on each shuffled version, results averaged\n")
        f.write(f"- **Topdown Anonymizer**: Run 3 times for consistency check, results averaged\n")
        f.write("- All times are in seconds\n")
        f.write("- MD = Distortion Metric (lower is better)\n")
        f.write("- LM = Loss Metric (lower is better)\n")
    
    print(f"\nResults written to {output_file}")


def main():
    print("K-Anonymity Experiment Runner")
    print("=" * 80)
    print(f"Skeleton file: {SKELETON_FILE}")
    print(f"DGH folder: {DGH_FOLDER}")
    print(f"Raw dataset: {RAW_DATASET}")
    print(f"K values: {K_VALUES}")
    print("=" * 80)
    
    # Run experiments
    results = run_experiments()
    
    # Write results to markdown
    write_results_to_markdown(results)
    
    print("\n" + "=" * 80)
    print("Experiments completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

