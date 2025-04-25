import pandas as pd
import numpy as np
import random
import requests
import time
import itertools
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to get road distance between two coordinates using OSRM API
def get_road_distance(coord1, coord2):
    """
    Get road distance between two coordinates using OSRM API.
    coord1, coord2: tuples of (longitude, latitude)
    Returns travel time in seconds
    """
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{coord1[0]},{coord1[1]};{coord2[0]},{coord2[1]}?overview=false"
        response = requests.get(url)
        data = response.json()
        
        # Check if route was found
        if data["code"] == "Ok" and len(data["routes"]) > 0:
            duration = data["routes"][0]["duration"]  # in seconds
            return duration
        else:
            # Return a large value if no route found
            return float('inf')
    except Exception as e:
        print(f"Error getting road distance: {e}")
        # Return a large value in case of error
        return float('inf')
    
# Function to calculate total travel time for a given order of nodes
def calculate_total_time(order, distance_matrix):
    total_time = 0
    for i in range(len(order)-1):
        total_time += distance_matrix[order[i]][order[i+1]]
    return total_time

# Function to process a permutation and return its total time
def process_permutation(perm, distance_matrix):
    return perm, calculate_total_time(perm, distance_matrix)

# Main function to find optimal route
def find_optimal_route(csv_file_path, num_runs=10):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
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
        
        # Calculate distance matrix
        n = len(selected_df)
        distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        print("Calculating distance matrix...")
        for i in tqdm(range(n)):
            for j in range(i+1, n):
                # Note: OSRM API expects coordinates as (longitude, latitude)
                coord1 = (selected_df.iloc[i]['x'], selected_df.iloc[i]['y'])
                coord2 = (selected_df.iloc[j]['x'], selected_df.iloc[j]['y'])
                
                # Add some delay to avoid hitting API rate limits
                time.sleep(0.2)
                
                duration = get_road_distance(coord1, coord2)
                distance_matrix[i][j] = duration
                distance_matrix[j][i] = duration
        
        # Find the optimal order using multiprocessing
        all_perms = list(itertools.permutations(range(n)))
        
        # Use multiprocessing to speed up the calculation
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(partial(process_permutation, distance_matrix=distance_matrix), all_perms)
        
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
        
        plt.title(f"Optimal Route (Run {run+1}) - Total Time: {best_time/60:.2f} minutes")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f"optimal_route_run_{run+1}.png")
        plt.close()
    
    # Find the overall best route
    best_route = min(best_routes, key=lambda x: x['total_time'])
    print(f"Best route found on run {best_route['run']} with total time: {best_route['total_time']/60:.2f} minutes")
    
    return best_routes

if __name__ == "__main__":
    # Replace with your actual CSV file path
    csv_file_path = "result/top_betweenness_per_community.csv"
    best_routes = find_optimal_route(csv_file_path)
    
    # Save results to file
    with open("optimal_routes_results.txt", "w") as f:
        for route_info in best_routes:
            f.write(f"Run {route_info['run']}:\n")
            f.write(f"Total time: {route_info['total_time']/60:.2f} minutes\n")
            f.write("Route: ")
            for node in route_info['route']:
                f.write(f"Community {node['community']} (Node {node['node_index']}) -> ")
            f.write("\n\n")