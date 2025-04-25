import pandas as pd
import numpy as np
import random
import time
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import math
from geopy.distance import geodesic
import concurrent.futures
import os
import shutil
import json
import igraph as ig
from tqdm import tqdm

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

# Function to load and parse GeoJSON
def load_geojson(file_path):
    with open(file_path, 'r') as f:
        geojson_data = json.load(f)
    return geojson_data

# Function to create igraph from GeoJSON
def create_igraph_from_geojson(geojson_data):
    # Extract nodes and edges
    nodes = {}
    edges = []
    node_attrs = {"x": [], "y": [], "id": [], "community": []}
    
    # Process features to extract nodes and edges
    for feature in geojson_data["features"]:
        if feature["geometry"]["type"] == "LineString":
            coords = feature["geometry"]["coordinates"]
            for i in range(len(coords) - 1):
                start = tuple(coords[i])
                end = tuple(coords[i + 1])
                
                # Create or get node indices
                if start not in nodes:
                    nodes[start] = len(nodes)
                    node_attrs["x"].append(start[0])
                    node_attrs["y"].append(start[1])
                    node_attrs["id"].append(len(nodes) - 1)
                    # Default community if not available
                    node_attrs["community"].append(feature["properties"].get("community", 0))
                
                if end not in nodes:
                    nodes[end] = len(nodes)
                    node_attrs["x"].append(end[0])
                    node_attrs["y"].append(end[1])
                    node_attrs["id"].append(len(nodes) - 1)
                    # Default community if not available
                    node_attrs["community"].append(feature["properties"].get("community", 0))
                
                edges.append((nodes[start], nodes[end]))
    
    # Create igraph
    g = ig.Graph(edges=edges, directed=False)
    
    # Set node attributes
    for attr_name, attr_values in node_attrs.items():
        g.vs[attr_name] = attr_values
    
    return g

# Function to plot the graph with optimal route
def plot_graph_with_route(g, optimal_route, output_path="min_time/route_on_graph.png"):
    plt.figure(figsize=(12, 10))
    
    # Plot edges
    edge_coords = []
    edge_colors = []
    for edge in g.es:
        source = g.vs[edge.source]
        target = g.vs[edge.target]
        edge_coords.append([(source["x"], source["y"]), (target["x"], target["y"])])
        edge_colors.append('grey')
    
    line_collection = LineCollection(edge_coords, colors=edge_colors, alpha=0.6, linewidths=0.7)
    plt.gca().add_collection(line_collection)
    
    # Plot nodes
    communities = np.array(g.vs["community"])
    unique_communities = np.unique(communities)
    cmap = plt.cm.get_cmap('tab20', len(unique_communities))
    
    # Create node color map by community
    node_colors = [cmap(communities[i] % len(unique_communities)) for i in range(len(g.vs))]
    
    plt.scatter(g.vs["x"], g.vs["y"], c=node_colors, s=10, alpha=0.7)
    
    # Plot optimal route
    route_x = [node['x'] for node in optimal_route]
    route_y = [node['y'] for node in optimal_route]
    
    # Add first node at the end to complete the loop
    route_x.append(route_x[0])
    route_y.append(route_y[0])
    
    plt.plot(route_x, route_y, 'r-', linewidth=2.5)
    
    # Add markers for each node in the route
    plt.scatter(route_x[:-1], route_y[:-1], c='red', s=100, marker='o', edgecolors='black', zorder=5)
    
    # Add labels for each node in the route
    for i, node in enumerate(optimal_route):
        plt.annotate(f"C{node['community']}", 
                    (node['x'], node['y']),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=12,
                    fontweight='bold',
                    zorder=6)
    
    # Set plot limits with a small margin
    x_min, x_max = min(g.vs["x"]), max(g.vs["x"])
    y_min, y_max = min(g.vs["y"]), max(g.vs["y"])
    margin = 0.01  # 1% margin
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    plt.title("Optimal Route on Road Network", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

# Main function to find optimal route
def find_optimal_route(csv_file_path, geojson_file_path, num_runs=10, num_random_perms=10):
    # Create min_time folder if it doesn't exist
    if os.path.exists("min_time"):
        shutil.rmtree("min_time")
    os.makedirs("min_time")
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get number of CPU cores
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    # Load GeoJSON and create igraph
    print("Loading GeoJSON and creating graph...")
    geojson_data = load_geojson(geojson_file_path)
    g = create_igraph_from_geojson(geojson_data)
    print(f"Graph created with {len(g.vs)} nodes and {len(g.es)} edges")
    
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
        
        # Generate random permutations
        print(f"Generating {num_random_perms} random permutations...")
        random_perms = []
        
        for _ in range(num_random_perms):
            # Generate a random permutation
            perm = list(range(n))
            random.shuffle(perm)
            random_perms.append(perm)
        
        # Find the best permutation among random ones
        best_perm = None
        best_time = float('inf')
        
        for perm in random_perms:
            time = calculate_total_time(perm, distance_matrix)
            if time < best_time:
                best_time = time
                best_perm = perm
        
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
        plt.savefig(f"min_time/route_run_{run+1}.png")
        plt.close()
        
        # Save this run's data
        with open(f"min_time/route_run_{run+1}_info.txt", "w") as f:
            f.write(f"Run {run+1}:\n")
            f.write(f"Total time: {time_value:.2f} {time_unit}\n")
            f.write("Route: ")
            for node in optimal_route:
                f.write(f"Community {node['community']} (Node {node['node_index']}) -> ")
            f.write("Community {0} (Node {1})\n".format(
                optimal_route[0]['community'], 
                optimal_route[0]['node_index']
            ))
    
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
    
    # Plot the best overall route
    optimal_route = best_route['route']
    plt.figure(figsize=(12, 10))
    x_coords = [node['x'] for node in optimal_route]
    y_coords = [node['y'] for node in optimal_route]
    
    # Add the first node at the end to complete the loop
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    
    plt.plot(x_coords, y_coords, 'ro-', linewidth=2)
    
    # Add node labels
    for i, node in enumerate(optimal_route):
        plt.annotate(f"C{node['community']}", 
                    (node['x'], node['y']),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=12,
                    fontweight='bold')
    
    plt.title(f"BEST OVERALL ROUTE - Total Time: {time_value:.2f} {time_unit}", fontsize=16)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("min_time/best_overall_route.png", dpi=300)
    plt.close()
    
    # Plot the best route on the graph
    plot_graph_with_route(g, optimal_route, "min_time/route_on_graph.png")
    
    # Save summary of all runs to a single file
    with open("min_time/all_results_summary.txt", "w") as f:
        for route_info in sorted(best_routes, key=lambda x: x['total_time']):
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
                f.write(f"C{node['community']}(N{node['node_index']}) → ")
            f.write(f"C{route_info['route'][0]['community']}(N{route_info['route'][0]['node_index']})\n\n")
    
    return best_routes, g

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
    
    # GeoJSON file path
    geojson_file_path = "filtered_output_2_small.geojson"
    
    # Check if files exist
    if not os.path.exists(csv_file_path):
        print(f"Error: File '{csv_file_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(geojson_file_path):
        print(f"Error: File '{geojson_file_path}' not found.")
        sys.exit(1)
    
    # Run the optimization with 10 random permutations for each run
    best_routes, graph = find_optimal_route(csv_file_path, geojson_file_path, num_runs=10, num_random_perms=10)
    
    # Print final message
    print(f"All results saved in the 'min_time' folder.")
    print(f"Check 'min_time/best_overall_route.png' for the visualization of the best route.")
    print(f"Check 'min_time/route_on_graph.png' for the visualization of the route on the road network.")
    print(f"See 'min_time/all_results_summary.txt' for details on all runs.")   