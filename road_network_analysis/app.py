from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import os
import uuid
import json
import pickle
import time
import networkx as nx
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap, Fullscreen
from collections import defaultdict
import warnings
import shutil
from werkzeug.utils import secure_filename
import threading
from datetime import datetime

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'road_network_analysis_key'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'geojson', 'json', 'pickle'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = (1024 * 1024) * 50  # 50MB limit

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global variables to store job status
processing_jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_id():
    return str(uuid.uuid4())

def get_job_status(job_id):
    if job_id in processing_jobs:
        return processing_jobs[job_id]
    return {'status': 'not_found', 'progress': 0, 'message': 'Job not found'}

def update_job_status(job_id, status, progress, message):
    processing_jobs[job_id] = {
        'status': status,
        'progress': progress,
        'message': message,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def clean_old_jobs():
    """Remove jobs older than 24 hours"""
    current_time = datetime.now()
    jobs_to_remove = []
    
    for job_id, job_info in processing_jobs.items():
        if 'timestamp' in job_info:
            job_time = datetime.strptime(job_info['timestamp'], '%Y-%m-%d %H:%M:%S')
            if (current_time - job_time).total_seconds() > 86400:  # 24 hours
                jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        del processing_jobs[job_id]
        # Clean up results directory
        job_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)

def load_or_create_graph(graph_path, representation_type='road_as_edge'):
    """
    Load graph from GeoJSON with different node representation options
    representation_type: 'road_as_edge' or 'coordinate_as_node'
    """
    update_job_status(os.path.basename(os.path.dirname(graph_path)), 
                     'processing', 10, 'Loading GeoJSON data...')
    
    try:
        # Check if file exists
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"File not found: {graph_path}")
        
        # Load GeoJSON file
        with open(graph_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        # Create a new graph
        G = nx.Graph()
        
        update_job_status(os.path.basename(os.path.dirname(graph_path)), 
                         'processing', 20, f'Creating graph with {len(geojson_data["features"])} features...')
        
        if representation_type == 'road_as_edge':
            # Each road is an edge with endpoints as nodes
            for i, feature in enumerate(geojson_data['features']):
                if feature['geometry']['type'] == 'LineString':
                    coordinates = feature['geometry']['coordinates']
                    
                    # Get road properties
                    road_type = feature['properties'].get('highway', 'unknown')
                    road_id = feature['properties'].get('osm_id', i)
                    
                    # Use the first and last point as nodes
                    if len(coordinates) >= 2:
                        node1_id = f"start_{road_id}"
                        node2_id = f"end_{road_id}"
                        
                        # Add nodes with their coordinates
                        G.add_node(node1_id, 
                                  x=coordinates[0][0], 
                                  y=coordinates[0][1])
                        G.add_node(node2_id, 
                                  x=coordinates[-1][0], 
                                  y=coordinates[-1][1])
                        
                        # Calculate edge length
                        length = np.sqrt((coordinates[-1][0] - coordinates[0][0])**2 + 
                                         (coordinates[-1][1] - coordinates[0][1])**2)
                        
                        # Add edge with attributes
                        G.add_edge(node1_id, node2_id, 
                                  road_type=road_type,
                                  length=length,
                                  weight=length)
        else:  # coordinate_as_node
            # Each coordinate point is a node
            for i, feature in enumerate(geojson_data['features']):
                if feature['geometry']['type'] == 'LineString':
                    coordinates = feature['geometry']['coordinates']
                    
                    # Get road properties
                    road_type = feature['properties'].get('highway', 'unknown')
                    road_id = feature['properties'].get('osm_id', i)
                    
                    # Add nodes and edges for this road segment
                    for j in range(len(coordinates) - 1):
                        # Create node IDs based on coordinates
                        node1_id = f"{road_id}_{j}"
                        node2_id = f"{road_id}_{j+1}"
                        
                        # Add nodes with their coordinates
                        G.add_node(node1_id, 
                                  x=coordinates[j][0], 
                                  y=coordinates[j][1])
                        G.add_node(node2_id, 
                                  x=coordinates[j+1][0], 
                                  y=coordinates[j+1][1])
                        
                        # Calculate edge length
                        length = np.sqrt((coordinates[j+1][0] - coordinates[j][0])**2 + 
                                         (coordinates[j+1][1] - coordinates[j][1])**2)
                        
                        # Add edge with attributes
                        G.add_edge(node1_id, node2_id, 
                                  road_type=road_type,
                                  length=length,
                                  weight=length)
        
        update_job_status(os.path.basename(os.path.dirname(graph_path)), 
                         'processing', 30, 
                         f'Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
        
        return G
    
    except Exception as e:
        update_job_status(os.path.basename(os.path.dirname(graph_path)), 
                         'failed', 0, f'Error creating graph: {str(e)}')
        raise e

def analyze_network(job_id, file_path, representation_type='road_as_edge', min_warehouse_distance=500, 
                   min_emergency_distance=300):
    """
    Process the uploaded file and analyze the road network
    """
    try:
        # Create job directory
        job_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Create subdirectories
        output_dirs = [
            os.path.join(job_dir, "centrality_results"),
            os.path.join(job_dir, "warehouse_analysis"),
            os.path.join(job_dir, "network_applications"),
            os.path.join(job_dir, "visualizations")
        ]
        
        for directory in output_dirs:
            os.makedirs(directory, exist_ok=True)
        
        update_job_status(job_id, 'processing', 5, 'Starting analysis...')
        
        # Load graph
        G = load_or_create_graph(file_path, representation_type)
        
        update_job_status(job_id, 'processing', 35, 'Calculating centrality measures...')
        
        # Centrality calculations
        # Degree Centrality
        start_time = time.time()
        degree_centrality = nx.degree_centrality(G)
        end_time = time.time()
        sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        
        # Betweenness Centrality
        start_time = time.time()
        betweenness_centrality = nx.betweenness_centrality(G, k=100, weight='length', normalized=True)
        end_time = time.time()
        sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
        
        # Closeness Centrality
        start_time = time.time()
        sample_size = min(1000, len(G))
        sampled_nodes = np.random.choice(list(G.nodes()), sample_size, replace=False)
        closeness_centrality = {}
        
        for node in sampled_nodes:
            try:
                length = nx.single_source_dijkstra_path_length(G, node, weight='length')
                if sum(length.values()) > 0:
                    closeness_centrality[node] = 1.0 / (sum(length.values()) / (len(length) - 1))
                else:
                    closeness_centrality[node] = 0.0
            except:
                closeness_centrality[node] = 0.0
        
        end_time = time.time()
        sorted_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
        
        # PageRank
        start_time = time.time()
        pagerank = nx.pagerank(G, weight='weight', max_iter=100)
        end_time = time.time()
        sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        
        # Store centrality results
        centrality_results = {
            'degree': {
                'top_nodes': sorted_degree[:100],
                'calculation_time': end_time - start_time
            },
            'betweenness': {
                'top_nodes': sorted_betweenness[:100],
                'calculation_time': end_time - start_time
            },
            'closeness': {
                'top_nodes': sorted_closeness[:100],
                'calculation_time': end_time - start_time,
                'note': f'Approximated using {sample_size} sampled nodes'
            },
            'pagerank': {
                'top_nodes': sorted_pagerank[:100],
                'calculation_time': end_time - start_time
            }
        }
        
        # Save centrality results
        centrality_json_path = os.path.join(job_dir, "centrality_results/all_centrality_measures.json")
        with open(centrality_json_path, "w") as f:
            serializable_results = {}
            for measure, data in centrality_results.items():
                if 'top_nodes' in data:
                    serializable_results[measure] = {
                        'top_nodes': [[str(node), score] for node, score in data['top_nodes']],
                        'calculation_time': data['calculation_time']
                    }
                    if 'note' in data:
                        serializable_results[measure]['note'] = data['note']
                else:
                    serializable_results[measure] = data
            
            json.dump(serializable_results, f, indent=2)
        
        update_job_status(job_id, 'processing', 50, 'Analyzing optimal warehouse locations...')
        
        # Get node coordinates
        node_coordinates = {}
        for node in G.nodes():
            if 'x' in G.nodes[node] and 'y' in G.nodes[node]:
                node_coordinates[node] = (G.nodes[node]['x'], G.nodes[node]['y'])
            else:
                node_coordinates[node] = (0, 0)
        
        # Find nodes that appear in multiple centrality measures
        summary = {
            'degree': [node for node, _ in sorted_degree[:10]],
            'betweenness': [node for node, _ in sorted_betweenness[:10]],
            'closeness': [node for node, _ in sorted_closeness[:10]],
            'pagerank': [node for node, _ in sorted_pagerank[:10]]
        }
        
        # Calculate node frequency across centrality measures
        node_frequency = defaultdict(int)
        for measure, nodes in summary.items():
            for node in nodes:
                node_frequency[node] += 1
        
        # Get road type diversity for each node
        node_road_diversity = {}
        for node in G.nodes():
            connected_roads = set()
            for edge in G.edges(node, data=True):
                road_type = edge[2].get('road_type', 'unknown')
                connected_roads.add(road_type)
            node_road_diversity[node] = len(connected_roads)
        
        # Calculate accessibility to major roads
        major_road_accessibility = {}
        for node in G.nodes():
            for edge in G.edges(node, data=True):
                road_type = edge[2].get('road_type', 'unknown')
                if road_type in ['primary', 'secondary', 'trunk']:
                    if node in major_road_accessibility:
                        major_road_accessibility[node] += 1
                    else:
                        major_road_accessibility[node] = 1
            
            # If not found, set to 0
            if node not in major_road_accessibility:
                major_road_accessibility[node] = 0
        
        # Combine scores for warehouse ranking with spatial distribution
        warehouse_rankings = []
        selected_warehouses = []
        
        # First calculate scores for all nodes
        for node in G.nodes():
            cent_score = node_frequency.get(node, 0)
            road_div = node_road_diversity.get(node, 0)
            major_access = major_road_accessibility.get(node, 0)
            
            # Combined score with weights
            combined_score = (cent_score * 0.5) + (road_div * 0.3) + (major_access * 0.2)
            
            lon, lat = node_coordinates.get(node, (0, 0))
            
            warehouse_rankings.append({
                'node': str(node),
                'longitude': lon,
                'latitude': lat,
                'centrality_score': cent_score,
                'road_diversity': road_div,
                'major_road_access': major_access,
                'combined_score': combined_score
            })
        
        # Sort by combined score
        warehouse_rankings.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Apply spatial distribution constraint
        final_warehouses = []
        
        for warehouse in warehouse_rankings:
            # Check if this warehouse is far enough from all selected warehouses
            is_valid = True
            warehouse_coords = (warehouse['latitude'], warehouse['longitude'])
            
            for selected in final_warehouses:
                selected_coords = (selected['latitude'], selected['longitude'])
                # Calculate Haversine distance in meters
                lat1, lon1 = np.radians(warehouse_coords)
                lat2, lon2 = np.radians(selected_coords)
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance = 6371000 * c  # Earth radius in meters
                
                if distance < min_warehouse_distance:
                    is_valid = False
                    break
            
            if is_valid:
                final_warehouses.append(warehouse)
                if len(final_warehouses) >= 20:  # Limit to top 20
                    break
        
        # Save warehouse results
        warehouse_df = pd.DataFrame(final_warehouses)
        warehouse_csv_path = os.path.join(job_dir, "warehouse_analysis/warehouse_rankings.csv")
        warehouse_df.to_csv(warehouse_csv_path, index=False)
        
        update_job_status(job_id, 'processing', 70, 'Analyzing optimal emergency service locations...')
        
        # Calculate emergency service locations with spatial distribution
        emergency_nodes = {}
        for measure in ['betweenness', 'closeness', 'pagerank']:
            if measure in centrality_results and 'top_nodes' in centrality_results[measure]:
                top_nodes = centrality_results[measure]['top_nodes']
                for node_data in top_nodes[:50]:
                    node, score = node_data
                    if node not in emergency_nodes:
                        emergency_nodes[node] = {}
                    emergency_nodes[node][measure] = score
        
        # Calculate combined score for emergency service locations
        emergency_scores = []
        for node, scores in emergency_nodes.items():
            combined_score = (
                scores.get('betweenness', 0) * 0.5 + 
                scores.get('closeness', 0) * 0.4 + 
                scores.get('pagerank', 0) * 0.1
            )
            
            lon, lat = node_coordinates.get(node, (0, 0))
            
            emergency_scores.append({
                'node': str(node),
                'longitude': lon,
                'latitude': lat,
                'score': combined_score
            })
        
        # Sort by combined score
        emergency_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply spatial distribution constraint
        final_emergency = []
        
        for emergency in emergency_scores:
            # Check if this emergency location is far enough from all selected locations
            is_valid = True
            emergency_coords = (emergency['latitude'], emergency['longitude'])
            
            for selected in final_emergency:
                selected_coords = (selected['latitude'], selected['longitude'])
                # Calculate Haversine distance
                lat1, lon1 = np.radians(emergency_coords)
                lat2, lon2 = np.radians(selected_coords)
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance = 6371000 * c  # Earth radius in meters
                
                if distance < min_emergency_distance:
                    is_valid = False
                    break
            
            if is_valid:
                final_emergency.append(emergency)
                if len(final_emergency) >= 20:  # Limit to top 20
                    break
        
        # Save emergency service location recommendations
        emergency_df = pd.DataFrame(final_emergency)
        emergency_csv_path = os.path.join(job_dir, "network_applications/emergency_service_locations.csv")
        emergency_df.to_csv(emergency_csv_path, index=False)
        
        update_job_status(job_id, 'processing', 80, 'Analyzing traffic congestion points...')
        
        # Traffic Congestion Analysis
        edge_betweenness = nx.edge_betweenness_centrality(G, k=100, weight='length', normalized=True)
        sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top congestion points
        congestion_points = []
        for (u, v), score in sorted_edges[:100]:
            if 'x' in G.nodes[u] and 'y' in G.nodes[u] and 'x' in G.nodes[v] and 'y' in G.nodes[v]:
                u_lon, u_lat = G.nodes[u]['x'], G.nodes[u]['y']
                v_lon, v_lat = G.nodes[v]['x'], G.nodes[v]['y']
            else:
                u_lon, u_lat = 0, 0
                v_lon, v_lat = 0, 0
            
            # Use midpoint of edge as congestion point
            mid_lon = (u_lon + v_lon) / 2
            mid_lat = (u_lat + v_lat) / 2
            
            road_type = G.edges[u, v].get('road_type', 'unknown')
            
            congestion_points.append({
                'edge': f"{u} - {v}",
                'longitude': mid_lon,
                'latitude': mid_lat,
                'betweenness': score,
                'road_type': road_type
            })
        
        # Save congestion analysis
        congestion_df = pd.DataFrame(congestion_points)
        congestion_csv_path = os.path.join(job_dir, "network_applications/traffic_congestion_points.csv")
        congestion_df.to_csv(congestion_csv_path, index=False)
        
        update_job_status(job_id, 'processing', 90, 'Creating visualizations...')
        
        # Calculate center of the map
        latitudes = []
        longitudes = []
        for node in list(G.nodes())[:1000]:
            if 'x' in G.nodes[node] and 'y' in G.nodes[node]:
                lon = float(G.nodes[node]['x'])
                lat = float(G.nodes[node]['y'])
                if -180 <= lon <= 180 and -90 <= lat <= 90:  # Filter valid coordinates
                    longitudes.append(lon)
                    latitudes.append(lat)
        
        # Set default center if no valid coordinates
        center_lat = np.mean(latitudes) if latitudes else 28.26
        center_lon = np.mean(longitudes) if longitudes else 76.85
        
        # Create combined applications map
        app_map = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                            tiles='cartodbpositron')
        
        # Add a title
        title_html = '''
        <h3 align="center" style="font-size:16px"><b>Road Network Analysis - Multiple Applications</b></h3>
        '''
        app_map.get_root().html.add_child(folium.Element(title_html))
        
        # Add mini-map and fullscreen control
        MiniMap(position="bottomright", width=150, height=150).add_to(app_map)
        Fullscreen(position="topright").add_to(app_map)
        
        # Add top warehouse locations
        warehouse_group = folium.FeatureGroup(name="Top Warehouse Locations").add_to(app_map)
        for i, row in warehouse_df.iterrows():
            popup_text = f"""
            <b>Top Warehouse #{i+1}</b><br>
            Combined Score: {row['combined_score']:.4f}<br>
            Centrality Score: {row['centrality_score']}<br>
            Road Diversity: {row['road_diversity']}<br>
            Major Road Access: {row['major_road_access']}
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_text,
                tooltip=f"Warehouse #{i+1}",
                icon=folium.Icon(color='green', icon='industry', prefix='fa'),
            ).add_to(warehouse_group)
        
        # Add emergency service locations
        emergency_group = folium.FeatureGroup(name="Emergency Service Locations").add_to(app_map)
        for i, row in emergency_df.iterrows():
            popup_text = f"""
            <b>Emergency Service #{i+1}</b><br>
            Score: {row['score']:.4f}
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_text,
                tooltip=f"Emergency #{i+1}",
                icon=folium.Icon(color='red', icon='plus', prefix='fa'),
            ).add_to(emergency_group)
        
        # Add traffic congestion points
        congestion_group = folium.FeatureGroup(name="Traffic Congestion Points").add_to(app_map)
        for i, row in congestion_df.iterrows():
            popup_text = f"""
            <b>Congestion Point #{i+1}</b><br>
            Betweenness: {row['betweenness']:.6f}<br>
            Road Type: {row['road_type']}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=popup_text,
                tooltip=f"Congestion #{i+1}",
                color='orange',
                fill=True,
                fill_color='orange',
                fill_opacity=0.7
            ).add_to(congestion_group)
        
        # Add layer control
        folium.LayerControl().add_to(app_map)
        
        # Save the map
        app_map_path = os.path.join(job_dir, "visualizations/network_applications.html")
        app_map.save(app_map_path)
        
        # Save graph to pickle for future use
        graph_pickle_path = os.path.join(job_dir, "processed_graph.pickle")
        with open(graph_pickle_path, "wb") as f:
            pickle.dump(G, f)
        
        update_job_status(job_id, 'completed', 100, 'Analysis completed successfully!')
        return job_id
    
    except Exception as e:
        update_job_status(job_id, 'failed', 0, f'Error during analysis: {str(e)}')
        return None

@app.route('/')
def index():
    clean_old_jobs()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique job ID
        job_id = generate_unique_id()
        
        # Create job directory
        job_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(job_dir, filename)
        file.save(file_path)
        
        # Get options from form
        representation_type = request.form.get('representation_type', 'road_as_edge')
        min_warehouse_distance = int(request.form.get('min_warehouse_distance', 500))
        min_emergency_distance = int(request.form.get('min_emergency_distance', 300))
        
        # Initialize job status
        update_job_status(job_id, 'queued', 0, 'Job queued, waiting to start...')
        
        # Start processing in a separate thread
        thread = threading.Thread(
            target=analyze_network,
            args=(job_id, file_path, representation_type, min_warehouse_distance, min_emergency_distance)
        )
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('job_status', job_id=job_id))
    
    flash('Invalid file type. Please upload a GeoJSON or pickle file.')
    return redirect(url_for('index'))

@app.route('/status/<job_id>')
def job_status(job_id):
    status = get_job_status(job_id)
    return render_template('status.html', job_id=job_id, status=status)

@app.route('/api/status/<job_id>')
def api_job_status(job_id):
    status = get_job_status(job_id)
    return jsonify(status)

@app.route('/results/<job_id>')
def results(job_id):
    status = get_job_status(job_id)
    
    if status['status'] != 'completed':
        flash('Job is not completed yet')
        return redirect(url_for('job_status', job_id=job_id))
    
    # Load results
    job_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    
    warehouse_csv = os.path.join(job_dir, "warehouse_analysis/warehouse_rankings.csv")
    emergency_csv = os.path.join(job_dir, "network_applications/emergency_service_locations.csv")
    congestion_csv = os.path.join(job_dir, "network_applications/traffic_congestion_points.csv")
    
    warehouses = []
    emergency_services = []
    congestion_points = []
    
    if os.path.exists(warehouse_csv):
        warehouses = pd.read_csv(warehouse_csv).to_dict('records')
    
    if os.path.exists(emergency_csv):
        emergency_services = pd.read_csv(emergency_csv).to_dict('records')
    
    if os.path.exists(congestion_csv):
        congestion_points = pd.read_csv(congestion_csv).to_dict('records')
    
    # Interactive map path
    map_path = f"/results/{job_id}/visualizations/network_applications.html"
    
    return render_template('results.html', 
                          job_id=job_id,
                          map_path=map_path,
                          warehouses=warehouses[:10],
                          status= status,
                          emergency_services=emergency_services[:10],
                          congestion_points=congestion_points[:10])

@app.route('/results/<job_id>/<path:filename>')
def download_result(job_id, filename):
    return send_from_directory(os.path.join(app.config['RESULTS_FOLDER'], job_id), filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)