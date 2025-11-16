##############################################################################
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
import glob
import os
import sys
from copy import deepcopy
from typing import Optional
import numpy as np

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]['age']) # debug: testing.
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    tree = {}  
    parent = {}  
    depth = {}  
    root = None
    
    with open(DGH_file) as f:
        lines = f.readlines()
        
    stack = []
    
    for line in lines:
        level = 0
        while level < len(line) and line[level] == '\t':
            level += 1
        
        node = line.strip()
        
        if not node:  
            continue
        
        tree[node] = []
        depth[node] = level
        
        if level == 0:
            root = node
            parent[node] = None
            stack = [node]
        else:
            stack = stack[:level]
            
            parent_node = stack[-1]
            parent[node] = parent_node
            
            tree[parent_node].append(node)
            
            stack.append(node)
    
    num_leaves = sum(1 for children in tree.values() if len(children) == 0)
    
    leaf_descendants = {}
    
    def get_leaf_descendants_for_node(node):
        """Get all leaf descendants of a node (including the node itself if it's a leaf)."""
        if node in leaf_descendants:
            return leaf_descendants[node]
        
        leaves = set()
        if len(tree[node]) == 0:  
            leaves.add(node)
        else:
            for child in tree[node]:
                leaves.update(get_leaf_descendants_for_node(child))
        
        leaf_descendants[node] = leaves
        return leaves
    
    for node in tree.keys():
        get_leaf_descendants_for_node(node)
    
    return {
        'tree': tree,
        'parent': parent,
        'root': root,
        'depth': depth,
        'num_leaves': num_leaves,
        'leaf_descendants': leaf_descendants
    }


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file);

    return DGHs


## Helper Functions

def get_ancestors(value, dgh):
    """Get all ancestors of a value in the DGH tree, including the value itself."""
    ancestors = []
    current = value
    while current is not None:
        ancestors.append(current)
        current = dgh['parent'].get(current)
    return ancestors
    
def find_common_ancestor(values, dgh, cache=None):
    """Find the minimal common ancestor (LCA) of a set of values in the DGH tree."""
    if len(values) == 0:
        return dgh['root']
    
    unique_values = set(values)
    if len(unique_values) == 1:
        return list(unique_values)[0]
    
    if cache is not None:
        cache_key = frozenset(unique_values)
        if cache_key in cache:
            return cache[cache_key]
    
    all_ancestors = []
    for value in unique_values:
        ancestors = get_ancestors(value, dgh)
        all_ancestors.append(set(ancestors))
    
    common = all_ancestors[0]
    for ancestor_set in all_ancestors[1:]:
        common = common.intersection(ancestor_set)
    
    if len(common) == 0:
        return dgh['root']
    
    max_depth = -1
    lca = dgh['root']
    for ancestor in common:
        if dgh['depth'][ancestor] > max_depth:
            max_depth = dgh['depth'][ancestor]
            lca = ancestor
    
    if cache is not None:
        cache_key = frozenset(unique_values)
        cache[cache_key] = lca
    
    return lca


def generalize_cluster(cluster, DGHs):
    """Generalize all QI attributes in a cluster to achieve k-anonymity."""
    for attribute in DGHs.keys():
        
        values = [record[attribute] for record in cluster]        
        dgh = DGHs[attribute]
        lca = find_common_ancestor(values, dgh)
        
        for record in cluster:
            record[attribute] = lca
    
    return cluster

def calculate_distance(records, DGHs, lca_cache=None):
    """
    Calculate the distance metric for a set of records.
    Distance = total LM cost of placing these records in one EC with minimum generalization.
    """
    
    assert len(records) == 2, "Distance calculation requires exactly 2 records"
    
    M = len(DGHs)
    total_cost = 0.0
    
    for attribute in DGHs.keys():
        values = [record[attribute] for record in records]
        
        dgh = DGHs[attribute]
        lca = find_common_ancestor(values, dgh, cache=lca_cache.get(attribute) if lca_cache else None)
        
        if lca in dgh['leaf_descendants']:
            num_leaves = len(dgh['leaf_descendants'][lca])
            total_leaves_attr = dgh['num_leaves']
            attribute_loss = (num_leaves - 1) / (total_leaves_attr - 1)
            total_cost += attribute_loss * (1.0 / M)
            
    return total_cost

class TopDownAnonNode:
    """Node class to represent nodes in the top-down specialization tree."""
    def __init__(self, records, generalization):
        self.records = records  # List of record indices
        self.generalization = generalization  # Dict: attribute -> current generalized value
        self.children = []  # List of child nodes
    
    def is_leaf(self):
        return len(self.children) == 0


def create_root_node(dataset, DGHs):
    """Create root node with all records, maximally generalized."""
    all_indices = list(range(len(dataset)))
    
    # Initialize with root values for all QI attributes
    generalization = {}
    for attribute, dgh in DGHs.items():
        generalization[attribute] = dgh['root']
    
    return TopDownAnonNode(all_indices, generalization)


def get_child_value(attribute, record, dgh, current_value):
    """Determine which child value a record belongs to when specializing an attribute."""
    record_value = record[attribute]
    children = dgh['tree'][current_value]
    
    ancestors = set()
    current = record_value
    while current is not None:
        ancestors.add(current)
        current = dgh['parent'].get(current)
    
    for child in children:
        if child in ancestors:
            return child
    
    return children[0] if children else current_value


def get_possible_specializations(node, DGHs, k, raw_dataset):
    """Find all valid specializations for this node."""
    candidates = []
    
    for attribute in DGHs.keys():
        current_value = node.generalization[attribute]
        dgh = DGHs[attribute]
        children = dgh['tree'][current_value]
        
        if len(children) == 0:
            continue  # Already at leaf
        
        # Initialize empty buckets
        groups = {child: [] for child in children}
        
        # Assign records to child buckets
        for rec_idx in node.records:
            record = raw_dataset[rec_idx]
            child_val = get_child_value(attribute, record, dgh, current_value)
            groups[child_val].append(rec_idx)
        
        # Filter only non-empty children
        nonempty_groups = {c: g for c, g in groups.items() if len(g) > 0}
        
        # Valid specialization requires: all non-empty children have >= k
        valid = True
        for rec_list in nonempty_groups.values():
            if len(rec_list) < k:
                valid = False
                break
        
        if not valid:
            continue
        
        candidates.append({
            'attribute': attribute,
            'children_values': children,
            'groups': groups,
            'nonempty': nonempty_groups
        })
    
    return candidates


def select_best_specialization(candidates):
    if len(candidates) == 0:
        return None
    
    # Minimize number of non-empty children
    min_children = min(len(c['nonempty']) for c in candidates)
    filtered = [c for c in candidates if len(c['nonempty']) == min_children]
    
    if len(filtered) == 1:
        return filtered[0]
    
    # Among ties, choose distribution closest to uniform
    best = None
    best_distance = float('inf')
    uniform_prob = 1.0 / min_children
    
    for spec in filtered:
        counts = [len(v) for v in spec['nonempty'].values()]
        total = sum(counts)
        distribution = [c / total for c in counts]
        
        dist = sum(abs(p - uniform_prob) for p in distribution)
        
        if dist < best_distance:
            best_distance = dist
            best = spec
    
    return best


def split_node(node, specialization):
    attribute = specialization['attribute']
    nonempty_groups = specialization['nonempty']
    
    children = []
    
    for child_value, rec_list in nonempty_groups.items():
        new_generalization = node.generalization.copy()
        new_generalization[attribute] = child_value
        
        child_node = TopDownAnonNode(rec_list, new_generalization)
        children.append(child_node)
    
    return children


def specialize_node(node, DGHs, k, raw_dataset):
    candidates = get_possible_specializations(node, DGHs, k, raw_dataset)
    
    if len(candidates) == 0:
        return  
    
    best_spec = select_best_specialization(candidates)
    if best_spec is None:
        return
    
    children = split_node(node, best_spec)
    node.children = children
    
    for child in children:
        specialize_node(child, DGHs, k, raw_dataset)



def get_all_leaf_nodes(node):
    """Get all leaf nodes in the tree."""
    if node.is_leaf():
        return [node]
    
    leaves = []
    for child in node.children:
        leaves.extend(get_all_leaf_nodes(child))
    return leaves


def apply_generalizations_to_dataset(dataset, root, DGHs):
    """Apply generalizations from the tree to create the anonymized dataset."""
    result = deepcopy(dataset)
    
    leaves = get_all_leaf_nodes(root)
    
    for leaf in leaves:
        for record_idx in leaf.records:
            for attribute in DGHs.keys():
                result[record_idx][attribute] = leaf.generalization[attribute]
    
    return result


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    ## Start of my code.

    total_cost = 0.0
    for i in range(len(raw_dataset)):
        raw_record = raw_dataset[i]
        anon_record = anonymized_dataset[i]
        
        # Iterate through each quasi-identifier attribute
        for attribute in DGHs.keys():
            raw_value = raw_record[attribute]
            anon_value = anon_record[attribute]
            
            # Get depths from the DGH
            depth_dict = DGHs[attribute]['depth']
            
            # Calculate distortion as difference in depths
            if raw_value in depth_dict and anon_value in depth_dict:
                distortion = abs(depth_dict[anon_value] - depth_dict[raw_value])
                total_cost += distortion
    
    return total_cost


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    ## Start of my code.
    
    total_cost = 0.0
    M = len(DGHs)
    
    for i in range(len(anonymized_dataset)):
        anon_record = anonymized_dataset[i]
        
        record_cost = 0.0
        
        for attribute in DGHs.keys():
            anon_value = anon_record[attribute]
            dgh = DGHs[attribute]
            
            attribute_loss = 0.0
            
            if anon_value in dgh['leaf_descendants']:
                num_leaves = len(dgh['leaf_descendants'][anon_value])
                total_leaves_attr = dgh['num_leaves']
                
                if total_leaves_attr > 1:
                    attribute_loss = (num_leaves - 1) / (total_leaves_attr - 1)
                
            record_cost += attribute_loss * (1.0 / M)
        
        total_cost += record_cost
    
    return total_cost


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymize a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize
    
    clusters = []

    D = len(raw_dataset)
    
    ################### Start of my code. ###################
    
    # Divide the shuffled dataset into clusters of size at least k
    num_full_clusters = D // k
    remainder = D % k
    
    idx = 0
    for i in range(num_full_clusters):
        cluster = list(raw_dataset[idx:idx + k])
        idx += k
        clusters.append(cluster)
    
    if remainder > 0:
        clusters[-1].extend(list(raw_dataset[idx:]))
        
    # Generalize each cluster to achieve k-anonymity
    for i in range(len(clusters)):
        clusters[i] = generalize_cluster(clusters[i], DGHs)

    ################### End of my code. ###################

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization of a dataset.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    
    anonymized_dataset = []
    
    ################### Start of my code. ###################

    lca_cache = {attribute: {} for attribute in DGHs.keys()}
    unused_indices = set(range(len(raw_dataset)))
    clusters = []
    cluster_indices = []
    
    while len(unused_indices) >= k:
        rec_idx = min(unused_indices)
        rec = raw_dataset[rec_idx]
        distances = []
        for idx in unused_indices:
            if idx != rec_idx:
                # Calculate distance between rec and this record
                dist = calculate_distance([rec, raw_dataset[idx]], DGHs, lca_cache)
                distances.append((dist, idx))
        
        # Sort by distance and pick k-1 closest records
        distances.sort(key=lambda x: (x[0], x[1])) # Sort by distance, then by index
        closest_indices = [rec_idx]  # Start with the picked record
        for i in range(min(k - 1, len(distances))):
            closest_indices.append(distances[i][1])
        
        # Create a k-anonymous EC from these records
        cluster = [raw_dataset[idx] for idx in closest_indices]
        generalized_cluster = deepcopy(cluster)
        generalize_cluster(generalized_cluster, DGHs)
        clusters.append(generalized_cluster)
        cluster_indices.append(closest_indices)
        
        for idx in closest_indices:
            unused_indices.remove(idx)
    
    # Handle remaining records (j > 0 unused records)
    if len(unused_indices) > 0:
        remaining_indices = list(unused_indices)
        last_cluster_indices = cluster_indices[-1]
        combined_indices = last_cluster_indices + remaining_indices
        combined_records = [raw_dataset[idx] for idx in combined_indices]
        generalized_combined = deepcopy(combined_records)
        generalize_cluster(generalized_combined, DGHs)
        clusters[-1] = generalized_combined
        cluster_indices[-1] = combined_indices
    
    anonymized_dataset = [None] * len(raw_dataset)

    # Iterate through the final clusters and their corresponding original indices
    for i in range(len(clusters)):
        cluster = clusters[i]
        indices = cluster_indices[i]
        
        for j in range(len(cluster)):
            original_index = indices[j]
            generalized_record = cluster[j]
            anonymized_dataset[original_index] = generalized_record

    write_dataset(anonymized_dataset, output_file)


def topdown_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Top-down anonymization of a dataset.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    
    anonymized_dataset = []
    
    ################### Start of my code. ###################
    
    # Create root node (all maximally generalized)
    root = create_root_node(raw_dataset, DGHs)
    
    # Recursively specialize
    specialize_node(root, DGHs, k, raw_dataset)
    
    # Apply generalizations to dataset
    anonymized_dataset = apply_generalizations_to_dataset(raw_dataset, root, DGHs)
    
    ################### End of my code. ###################

    write_dataset(anonymized_dataset, output_file)


# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
    print(f"\tWhere algorithm is one of [clustering, random, topdown]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'topdown']:
    print("Invalid algorithm.")
    sys.exit(2)

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer")
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, topdown]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonymity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")


# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300