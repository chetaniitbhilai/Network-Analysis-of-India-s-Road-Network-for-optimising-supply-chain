import json
import csv

def geojson_to_csv(geojson_path, nodes_csv, edges_csv):
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    nodes = []
    edges = []
    node_map = {}
    node_id = 1
    
    for feature in data["features"]:
        coordinates = feature["geometry"]["coordinates"]
        prev_id = None
        
        for lon, lat in coordinates:
            if (lat, lon) not in node_map:
                node_map[(lat, lon)] = node_id
                nodes.append([node_id, lat, lon])
                node_id += 1
            
            curr_id = node_map[(lat, lon)]
            if prev_id is not None:
                edges.append([prev_id, curr_id])
            prev_id = curr_id
    
    with open(nodes_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "latitude", "longitude"])
        writer.writerows(nodes)
    
    with open(edges_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows(edges)

# Example usage
geojson_to_csv("filtered_output_2.geojson", "nodes.csv", "edges.csv")
