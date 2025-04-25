# Road Network Analysis Application

## Overview

This application is a Flask-based web service for analyzing road networks using graph theory. It allows users to upload geospatial data (GeoJSON format) of road networks and performs various analyses to identify optimal locations for warehouses, emergency services, and potential traffic congestion points.

## Features

- **Road Network Graph Creation**: Converts GeoJSON road data into a mathematical graph representation
- **Centrality Analysis**: Calculates various centrality measures to identify key nodes in the network
- **Optimal Location Identification**: Suggests optimal locations for:
  - Warehouses based on centrality, road diversity, and major road access
  - Emergency service stations based on accessibility and network coverage
- **Traffic Congestion Analysis**: Identifies potential traffic bottlenecks
- **Interactive Visualization**: Creates interactive maps with Folium to visualize results
- **Asynchronous Processing**: Handles data processing asynchronously with job status tracking

## Technical Architecture

### Core Components

1. **Flask Web Server**: Provides the user interface and API endpoints
2. **NetworkX**: Powers the graph-based network analysis
3. **Folium**: Generates interactive maps for visualization
4. **Threading**: Enables asynchronous processing of network analysis jobs

### Directory Structure

- `/uploads`: Temporary storage for uploaded files
- `/results/{job_id}/`: Contains results organized by job ID:
  - `/centrality_results/`: Contains centrality measure data
  - `/warehouse_analysis/`: Contains optimal warehouse location data
  - `/network_applications/`: Contains emergency service locations and congestion points
  - `/visualizations/`: Contains generated interactive maps

## Detailed Function Descriptions

### 1. Core Utility Functions

#### `allowed_file(filename)`
- Validates if the uploaded file extension is in the allowed extensions list (geojson, json, pickle)

#### `generate_unique_id()`
- Creates a unique UUID for each analysis job

#### `get_job_status(job_id)` / `update_job_status(job_id, status, progress, message)`
- Maintains and retrieves the status of processing jobs

#### `clean_old_jobs()`
- Removes job data older than 24 hours to free up storage space

### 2. Graph Construction

#### `load_or_create_graph(graph_path, representation_type='road_as_edge')`
- Creates a NetworkX graph from GeoJSON road data
- Supports two representation methods:
  - `road_as_edge`: Each road segment becomes an edge with endpoints as nodes
  - `coordinate_as_node`: Each coordinate becomes a node, with edges connecting consecutive coordinates

### 3. Network Analysis

#### `analyze_network(job_id, file_path, representation_type, min_warehouse_distance, min_emergency_distance)`
- Main analysis function that processes the uploaded file and performs all analyses
- Calculates centrality measures:
  - Degree Centrality: Measures the number of connections a node has
  - Betweenness Centrality: Identifies nodes that act as bridges between different parts of the network
  - Closeness Centrality: Measures how close a node is to all other nodes
  - PageRank: Identifies globally important nodes
- Performs warehouse location analysis based on:
  - Centrality score
  - Road type diversity
  - Major road accessibility
  - Spatial distribution (ensuring warehouses aren't too close together)
- Identifies optimal emergency service locations based on:
  - Weighted combination of centrality measures
  - Spatial distribution constraints
- Detects traffic congestion points using edge betweenness centrality
- Creates visualizations using Folium

### 4. Flask Routes

#### `@app.route('/')`
- Renders the main upload page

#### `@app.route('/upload', methods=['POST'])`
- Processes file uploads
- Initiates asynchronous analysis
- Redirects to job status page

#### `@app.route('/status/<job_id>')` / `@app.route('/api/status/<job_id>')`
- Displays/returns job status information

#### `@app.route('/results/<job_id>')`
- Displays analysis results once the job is complete
- Shows top warehouse locations, emergency service locations, and congestion points
- Embeds interactive map visualization

#### `@app.route('/results/<job_id>/<path:filename>')`
- Serves result files for download

## Algorithm Details

### Warehouse Location Ranking Algorithm

The application uses a multi-criteria approach for warehouse location ranking:

1. **Base Scoring**:
   - Centrality Score (50%): Frequency of node appearance in top centrality measures
   - Road Diversity (30%): Number of different road types connected to the node
   - Major Road Access (20%): Number of major roads (primary, secondary, trunk) connected

2. **Spatial Distribution**:
   - Enforces minimum distance constraint between selected warehouses
   - Uses Haversine formula to calculate distances between geographic coordinates

### Emergency Service Location Algorithm

1. **Combined Centrality Score**:
   - Betweenness Centrality (50%): Prioritizes locations with high traffic flow
   - Closeness Centrality (40%): Ensures quick access to different parts of the network
   - PageRank (10%): Considers global importance of locations

2. **Spatial Distribution**:
   - Similar to warehouse algorithm, enforces minimum distance between emergency service locations

### Traffic Congestion Analysis

- Utilizes edge betweenness centrality to identify road segments likely to experience high traffic
- Higher edge betweenness indicates roads that participate in many shortest paths
- These roads are potential bottlenecks where traffic congestion might occur

## Usage

1. **Data Preparation**:
   - Prepare GeoJSON files containing road network data
   - Each feature should be a LineString with coordinates and properties

2. **Upload and Configuration**:
   - Upload the GeoJSON file via the web interface
   - Configure analysis parameters:
     - Network representation type
     - Minimum warehouse distance
     - Minimum emergency service distance

3. **Monitor Job Status**:
   - Track processing progress on the status page
   - Processing time depends on network size and complexity

4. **Explore Results**:
   - View interactive map with warehouse, emergency service, and congestion points
   - Download detailed CSV reports for further analysis

## Limitations and Considerations

- Processing large networks may require significant computational resources
- Centrality calculations are approximated for large networks to improve performance
- The application assumes that the road network is connected; disconnected components may affect analysis results
- Geographic coordinates must be in a format where Euclidean distance is meaningful (e.g., WGS84)

## Dependencies

- Flask: Web framework
- NetworkX: Graph analysis library
- Folium: Interactive map generation
- NumPy/Pandas: Data manipulation
- GeoJSON: Spatial data handling
