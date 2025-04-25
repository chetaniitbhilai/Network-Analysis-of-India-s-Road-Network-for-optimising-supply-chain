import pandas as pd
import numpy as np
import random
import time
import itertools
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from geopy.distance import geodesic
import concurrent.futures
import os

# Function to calculate geodesic distance (straight-line)
def get_geodesic_distance(coord1, coord2):
    """
    Get geodesic distance between two coordinates and convert to estimated travel time.
    coord1, coord2: tuples of (latitude, longitude)
    Returns estimated travel time in seconds
    """
    try:
        # Calculate straight-line distance in meters
        distance = geodesic(coord1, coord2).meters
        
        # Convert distance to estimated travel time (assuming average speed of 30 km/h)
        avg_speed_mps = 30 * 1000 / 3600  # 30 km/h in meters per second
        estimated_time = distance / avg_speed_mps
        
        return estimated_time
    except Exception as e:
        print(f"Error calculating geodesic distance: {e}")
        return float('inf')

# Function to calculate a chunk of the distance matrix
def calculate_distance_chunk(chunk_data, selected_df):
    i_vals, j_vals = zip(*chunk_data)
    results = []
    
    for i, j in zip(i_vals, j_vals):
        # Get coordinates - note y is latitude, x is longitude
        coord1 = (selected_df.iloc[i]['y'], selected_df.iloc[i]['x'])
        coord2 = (selected_df.iloc[j]['y'], selected_df.iloc[j]['x'])
        
        duration = get_geodesic_distance(coord1, coord2)
        results.append((i, j, duration))
    
    return results

# Function to calculate total travel time for a given order of nodes
def calculate_total_time(order, distance_matrix):
    total_time = 0
    for i in range(len(order)-1):
        total_time += distance_matrix[order[i]][order[i+1]]
    # Add return to starting point
    total_time += distance_matrix[order[-1]][order[0]]
    return total_time

# Function to process a batch of permutations and return the best one
def process_permutation_batch(perms, distance_matrix):
    best_perm = None
    best_time = float('inf')
    
    for perm in perms:
        time = calculate_total_time(perm, distance_matrix)
        if time < best_time:
            best_time = time
            best_perm = perm
            
    return best_perm, best_time

# Standalone function for nearest neighbor (to make it picklable)
def process_start_node(args):
    start_node, distance_matrix = args
    n = len(distance_matrix)
    
    # Initialize route with start node
    route = [start_node]
    unvisited = set(range(n)) - {start_node}
    total_time = 0
    
    # Build route by always choosing nearest unvisited node
    current = start_node
    while unvisited:
        next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
        route.append(next_node)
        total_time += distance_matrix[current][next_node]
        unvisited.remove(next_node)
        current = next_node
    
    # Add time to return to start
    total_time += distance_matrix[current][start_node]
    
    return route, total_time

# Main function to find optimal route
def find_optimal_route(csv_file_path, num_runs=10):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get number of CPU cores
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    best_routes = []
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        
        # Group by community and select one random node from each community
        selected_nodes = []
        for community, group in df.groupby('community'):
            # Select random node from this community
            selected_node = group.sample(1).iloc[0]
            selected_nodes.append({
                'community': community,
                'node_index': selected_node['node_index'],
                'x': selected_node['x'],
                'y': selected_node['y']
            })
        
        selected_df = pd.DataFrame(selected_nodes)
        
        # Calculate distance matrix with parallelization
        n = len(selected_df)
        distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        print("Calculating distance matrix in parallel...")
        
        # Create list of all pairs to calculate
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Split pairs into chunks for parallel processing
        chunk_size = max(1, len(pairs) // num_cores)
        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        
        # Calculate distances in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(calculate_distance_chunk, chunk, selected_df) for chunk in chunks]
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(chunks)):
                results = future.result()
                for i, j, duration in results:
                    distance_matrix[i][j] = duration
                    distance_matrix[j][i] = duration  # Matrix is symmetric
        
        # Check if the number of permutations is manageable
        num_communities = len(selected_df)
        try:
            num_perms = math.factorial(num_communities)
        except OverflowError:
            num_perms = float('inf')
        
        if num_perms > 1000000:  # Too many permutations for brute force
            print(f"Number of permutations is too large for brute force. Using nearest neighbor heuristic.")
            
            # Use nearest neighbor heuristic
            start_nodes = list(range(n))
            
            # Create argument list for the workers
            args_list = [(start_node, distance_matrix) for start_node in start_nodes]
            
            # Run in parallel with proper arguments
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
                results = list(executor.map(process_start_node, args_list))
            
            # Find best result
            best_route_order, best_route_time = min(results, key=lambda x: x[1])
            
            best_perm = best_route_order
            best_time = best_route_time
            
        else:
            print(f"Calculating optimal route from {num_perms} permutations using parallel processing...")
            
            # Generate all permutations
            all_perms = list(itertools.permutations(range(n)))
            
            # Split permutations into batches for parallel processing
            batch_size = max(1, len(all_perms) // (num_cores * 10))  # Multiple batches per core
            batches = [all_perms[i:i + batch_size] for i in range(0, len(all_perms), batch_size)]
            
            # Process batches in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = [executor.submit(process_permutation_batch, batch, distance_matrix) for batch in batches]
                
                results = []
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(batches)):
                    results.append(future.result())
            
            # Find the best permutation
            best_perm, best_time = min(results, key=lambda x: x[1])
        
        # Get the nodes in the optimal order
        optimal_route = [selected_df.iloc[i].to_dict() for i in best_perm]
        
        best_routes.append({
            'run': run + 1,
            'total_time': best_time,
            'route': optimal_route
        })
        
        # Plot the optimal route
        plt.figure(figsize=(10, 8))
        x_coords = [node['x'] for node in optimal_route]
        y_coords = [node['y'] for node in optimal_route]
        
        # Add the first node at the end to complete the loop
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        plt.plot(x_coords, y_coords, 'bo-')
        
        # Add node labels
        for i, node in enumerate(optimal_route):
            plt.annotate(f"C{node['community']}", 
                        (node['x'], node['y']),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
        
        time_unit = "seconds"
        time_value = best_time
        if best_time > 3600:
            time_unit = "hours"
            time_value = best_time / 3600
        elif best_time > 60:
            time_unit = "minutes"
            time_value = best_time / 60
            
        plt.title(f"Optimal Route (Run {run+1}) - Total Time: {time_value:.2f} {time_unit}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f"optimal_route_run_{run+1}.png")
        plt.close()
    
    # Find the overall best route
    best_route = min(best_routes, key=lambda x: x['total_time'])
    
    time_unit = "seconds"
    time_value = best_route['total_time']
    if time_value > 3600:
        time_unit = "hours"
        time_value = time_value / 3600
    elif time_value > 60:
        time_unit = "minutes"
        time_value = time_value / 60
        
    print(f"Best route found on run {best_route['run']} with total time: {time_value:.2f} {time_unit}")
    
    return best_routes

if __name__ == "__main__":
    # Set processes to use 'spawn' start method for better compatibility
    # This helps with multiprocessing issues on some systems
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set
    
    # Replace with your actual CSV file path or use command line argument
    import sys
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
    else:
        csv_file_path = "result/top_betweenness_per_community.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File '{csv_file_path}' not found.")
        sys.exit(1)
    
    best_routes = find_optimal_route(csv_file_path, num_runs=10)
    
    # Save results to file
    with open("optimal_routes_results.txt", "w") as f:
        for route_info in best_routes:
            f.write(f"Run {route_info['run']}:\n")
            
            time_unit = "seconds"
            time_value = route_info['total_time']
            if time_value > 3600:
                time_unit = "hours"
                time_value = time_value / 3600
            elif time_value > 60:
                time_unit = "minutes"
                time_value = time_value / 60
                
            f.write(f"Total time: {time_value:.2f} {time_unit}\n")
            f.write("Route: ")
            for node in route_info['route']:
                f.write(f"Community {node['community']} (Node {node['node_index']}) -> ")
            f.write("Community {0} (Node {1})\n\n".format(
                route_info['route'][0]['community'], 
                route_info['route'][0]['node_index']
            ))