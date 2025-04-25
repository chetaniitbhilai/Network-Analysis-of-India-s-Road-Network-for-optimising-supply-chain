import pandas as pd
import numpy as np
import random
import time
import itertools
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import googlemaps
from geopy.distance import geodesic
from datetime import datetime

# Function to get road distance using Google Maps API
# You'll need to get an API key from Google Cloud Platform
def get_road_distance_google(coord1, coord2, api_key):
    """
    Get road distance between two coordinates using Google Maps API.
    coord1, coord2: tuples of (latitude, longitude)
    Returns travel time in seconds
    """
    try:
        gmaps = googlemaps.Client(key=api_key)
        
        # Request directions
        directions_result = gmaps.directions(
            origin=f"{coord1[0]},{coord1[1]}",
            destination=f"{coord2[0]},{coord2[1]}",
            mode="driving",
            departure_time=datetime.now()
        )
        
        if directions_result:
            # Get duration in seconds
            duration = directions_result[0]['legs'][0]['duration']['value']
            return duration
        else:
            return float('inf')
    except Exception as e:
        print(f"Error with Google Maps API: {e}")
        return float('inf')

# Function to get distance using geopy (no API needed, straight-line distance)
def get_geodesic_distance(coord1, coord2):
    """
    Get geodesic (as-the-crow-flies) distance between two coordinates.
    coord1, coord2: tuples of (latitude, longitude)
    Returns distance in meters
    """
    try:
        # Note: geodesic expects (latitude, longitude)
        distance = geodesic((coord1[0], coord1[1]), (coord2[0], coord2[1])).meters
        
        # Convert distance to estimated travel time (assuming average speed of 30 km/h)
        # This is a rough estimation; for better results use an actual routing API
        avg_speed_mps = 30 * 1000 / 3600  # 30 km/h in meters per second
        estimated_time = distance / avg_speed_mps
        
        return estimated_time
    except Exception as e:
        print(f"Error calculating geodesic distance: {e}")
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
def find_optimal_route(csv_file_path, num_runs=10, use_google_maps=False, api_key=None):
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
                # Get coordinates - note y is latitude, x is longitude
                coord1 = (selected_df.iloc[i]['y'], selected_df.iloc[i]['x'])
                coord2 = (selected_df.iloc[j]['y'], selected_df.iloc[j]['x'])
                
                if use_google_maps and api_key:
                    # Add delay to respect API rate limits
                    time.sleep(0.2)
                    duration = get_road_distance_google(coord1, coord2, api_key)
                else:
                    # Use geodesic distance (no API needed)
                    duration = get_geodesic_distance(coord1, coord2)
                
                distance_matrix[i][j] = duration
                distance_matrix[j][i] = duration
        
        # Find the optimal order using multiprocessing to solve TSP
        all_perms = list(itertools.permutations(range(n)))
        
        print(f"Calculating optimal route from {len(all_perms)} permutations...")
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

# For large number of communities (more than 10), we should use a heuristic approach instead of brute force
def find_optimal_route_large(csv_file_path, num_runs=10, use_google_maps=False, api_key=None):
    """
    Use a nearest neighbor heuristic for larger problems where brute force isn't feasible
    """
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
                # Get coordinates - note y is latitude, x is longitude
                coord1 = (selected_df.iloc[i]['y'], selected_df.iloc[i]['x'])
                coord2 = (selected_df.iloc[j]['y'], selected_df.iloc[j]['x'])
                
                if use_google_maps and api_key:
                    # Add delay to respect API rate limits
                    time.sleep(0.2)
                    duration = get_road_distance_google(coord1, coord2, api_key)
                else:
                    # Use geodesic distance (no API needed)
                    duration = get_geodesic_distance(coord1, coord2)
                
                distance_matrix[i][j] = duration
                distance_matrix[j][i] = duration
        
        # Use nearest neighbor heuristic to find a good route
        best_route_time = float('inf')
        best_route_order = None
        
        # Try starting from each node
        for start_node in range(n):
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
            
            # Check if this is the best route so far
            if total_time < best_route_time:
                best_route_time = total_time
                best_route_order = route
        
        # Get the nodes in the optimal order
        optimal_route = [selected_df.iloc[i].to_dict() for i in best_route_order]
        
        best_routes.append({
            'run': run + 1,
            'total_time': best_route_time,
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
        time_value = best_route_time
        if best_route_time > 3600:
            time_unit = "hours"
            time_value = best_route_time / 3600
        elif best_route_time > 60:
            time_unit = "minutes"
            time_value = best_route_time / 60
            
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
    # Replace with your actual CSV file path
    csv_file_path = "result/top_betweenness_per_community.csv"
    
    # Choose which method to use based on number of communities
    df = pd.read_csv(csv_file_path)
    num_communities = df['community'].nunique()
    
    # For Google Maps API (optional)
    use_google_maps = False  # Set to True if you have an API key
    google_maps_api_key = "YOUR_API_KEY_HERE"  # Replace with your API key if using Google Maps
    
    if num_communities <= 10:
        # Brute force approach is feasible
        best_routes = find_optimal_route(
            csv_file_path, 
            num_runs=10, 
            use_google_maps=use_google_maps, 
            api_key=google_maps_api_key if use_google_maps else None
        )
    else:
        # Use heuristic approach for larger problems
        best_routes = find_optimal_route_large(
            csv_file_path, 
            num_runs=10, 
            use_google_maps=use_google_maps, 
            api_key=google_maps_api_key if use_google_maps else None
        )
    
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
            f.write("\n\n")