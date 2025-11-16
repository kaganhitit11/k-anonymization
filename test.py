import subprocess
import csv
import sys

def run_anonymization(method, output_file, seed=None, k=256, suffix=''):
    """
    Run an anonymization method and capture its output.
    
    Returns True on success, False on failure.
    """
    # Build the command
    cmd = [
        sys.executable,  # Use sys.executable for 'python3'
        'skeleton' + suffix + '.py', # Use the correct path to the skeleton file
        method,
        'DGHs/',     
        'adult-hw1.csv',        
        output_file,
        str(k),
    ]
    
    # Add seed for random method
    if method == 'random' and seed is not None:
        cmd.append(str(seed))
    
    print(f"\n[Running {method} anonymization method (k={k})...]")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        
        # Print the output (cost metrics)
        print(result.stdout.strip())
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {method} method:")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ Error: 'skeleton.py' not found or python not in path.")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def check_k_anonymity(anonymized_file: str, k: int, sensitive_attribute: str = None):
    """
    Checks if a given anonymized file satisfies k-anonymity.
    
    Args:
        anonymized_file: Path to the .csv file to check.
        k: The k-anonymity parameter.
        sensitive_attribute: The name of the sensitive attribute column.
    """
    print(f"[Checking k-anonymity for {anonymized_file} (k={k})...]")
    groups = {}
    
    try:
        with open(anonymized_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if not fieldnames:
                print("  Warning: File is empty.")
                return True # An empty file is technically k-anonymous

            # Per instructions, all attributes *except* the sensitive one are QIs
            qi_names = [name for name in fieldnames if name != sensitive_attribute and sensitive_attribute is not None]
            
            # Group rows by their QI values
            for row in reader:
                key = tuple(row[qi] for qi in qi_names)
                groups.setdefault(key, []).append(row)
                
    except FileNotFoundError:
        print(f"  ✗ Error: File not found: {anonymized_file}")
        return False
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
        return False

    if not groups:
         print("  Warning: No data rows found in file.")
         return True # No data = no violations

    # Check the size of each equivalence class
    failed_groups = 0
    for ec in groups.values():
        if len(ec) < k:
            failed_groups += 1
    
    if failed_groups > 0:
        print(f"  ✗ FAILED: {failed_groups} equivalence class(es) have fewer than {k} records.")
        # Optional: Print one example of a failing group
        for key, ec in groups.items():
            if len(ec) < k:
                print(f"    - Example failure: Group {key} has {len(ec)} record(s).")
                break
        return False
    else:
        print(f"  ✓ PASSED: All {len(groups)} equivalence class(es) have {k} or more records.")
        return True

def main():
    
    suffix = '_oguz'
    
    k = 128 # The k-value to test with
    sensitive_attribute = 'income' # Based on instructions
    
    methods = [
        ('clustering', 'result_clustering' + suffix + '.csv', None),
        ('random', 'result_random' + suffix + '.csv', 42),
        ('topdown', 'result_topdown' + suffix + '.csv', None)
    ]
    
    for method, output_file, seed in methods:
        # Run anonymization
        success = run_anonymization(method, output_file, seed, k, suffix)
        
        # Check k-anonymity if the run was successful
        if success:
            check_k_anonymity(output_file, k, sensitive_attribute)
        else:
            print(f"[Skipping check for {output_file} due to run failure.]")
        
        print("-" * 30) # Separator
        
if __name__ == "__main__":
    main()