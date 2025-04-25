import json
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
import igraph as ig
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from scipy.stats import pearsonr, spearmanr
import networkx as nx
from tqdm import tqdm
import contextily as ctx
from pyproj import CRS

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load GeoJSON
print("Loading GeoJSON data...")
with open("filtered_output_2_small.geojson", "r") as f:
    data = json.load(f)

# Step 2: Filter valid LineStrings
print("Filtering valid LineStrings...")
valid_features = []
for feature in data['features']:
    coords = feature['geometry']['coordinates']
    if feature['geometry']['type'] == "LineString" and len(coords) >= 2:
        valid_features.append(feature)

# Step 3: Create GeoDataFrame
print("Creating GeoDataFrame...")
geometries = [LineString(f["geometry"]["coordinates"]) for f in valid_features]
gdf = gpd.GeoDataFrame(valid_features, geometry=geometries)
gdf.reset_index(drop=True, inplace=True)
geom_list = gdf.geometry.values

# Step 4: Parallel pairwise intersection checking
def check_chunk(start, end):
    local_edges = []
    for i in range(start, end):
        g1 = geom_list[i]
        for j in range(i + 1, len(geom_list)):
            g2 = geom_list[j]
            try:
                if g1.intersects(g2):
                    local_edges.append((i, j))
            except:
                continue
    return local_edges

def parallel_intersection(n_workers=mp.cpu_count()):
    chunk_size = max(1, len(geom_list) // n_workers)
    ranges = [(i * chunk_size, min((i + 1) * chunk_size, len(geom_list))) for i in range(n_workers)]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(check_chunk, ranges)
    
    edges = [e for sub in results for e in sub]
    return edges

# Main analysis
print("Building graph and performing analysis...")
edges = parallel_intersection()

# Step 5: Build graph
g = ig.Graph()
g.add_vertices(len(gdf))
g.add_edges(edges)
print(f"Original graph: {g.summary()}")

# Step 6: Drop nodes with degree 0 and degree 1
degrees = g.degree()
to_keep = [i for i, d in enumerate(degrees) if d > 1]
g_filtered = g.subgraph(to_keep)
print(f"Filtered graph: {g_filtered.summary()}")

# Map old vertex IDs to new vertex IDs
old_to_new = {old_id: new_id for new_id, old_id in enumerate(to_keep)}
new_to_old = {new_id: old_id for new_id, old_id in enumerate(to_keep)}

# Create a filtered geodataframe with only the kept nodes
gdf_filtered = gdf.iloc[[new_to_old[i] for i in range(g_filtered.vcount())]]
gdf_filtered.reset_index(drop=True, inplace=True)

# Step 7: Centrality measures
print("Calculating centrality measures...")
# Betweenness centrality
betweenness = g_filtered.betweenness()
# Closeness centrality
closeness = g_filtered.closeness()
# Degree centrality
degree_centrality = [d / (g_filtered.vcount() - 1) for d in g_filtered.degree()]
# Eigenvector centrality
eigenvector = g_filtered.eigenvector_centrality()

# Add centrality measures to the graph as vertex attributes
g_filtered.vs["betweenness"] = betweenness
g_filtered.vs["closeness"] = closeness
g_filtered.vs["degree_centrality"] = degree_centrality
g_filtered.vs["eigenvector"] = eigenvector

# Step 8: Degree distribution analysis
print("Analyzing degree distribution...")
degrees = g_filtered.degree()
degree_counts = pd.Series(degrees).value_counts().sort_index()

# Create square plot for degree distribution
plt.figure(figsize=(10, 10))  # Square figure
plt.style.use('dark_background')
plt.bar(degree_counts.index, degree_counts.values, color='purple', alpha=0.7)
plt.yscale('log')
plt.xlabel('Degree', color='white', fontsize=14)
plt.ylabel('Count (log scale)', color='white', fontsize=14)
plt.title('Degree Distribution (log scale)', color='white', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tick_params(colors='white')
# Make the plot more square-like
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'degree_distribution.png'), dpi=300, bbox_inches='tight', facecolor='black')

# Step 9: Betweenness centrality analysis
print("Analyzing betweenness centrality...")
plt.figure(figsize=(10, 10))  # Square figure
plt.hist(betweenness, bins=50, color='purple', alpha=0.7)
plt.xlabel('Betweenness Centrality', color='white', fontsize=14)
plt.ylabel('Count', color='white', fontsize=14)
plt.title('Betweenness Centrality Distribution', color='white', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tick_params(colors='white')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'betweenness_histogram.png'), dpi=300, bbox_inches='tight', facecolor='black')

# Step 10: Centrality correlation analysis
print("Analyzing centrality correlations...")
centrality_df = pd.DataFrame({
    'betweenness': betweenness,
    'closeness': closeness,
    'degree_centrality': degree_centrality,
    'eigenvector': eigenvector
})

# Calculate correlations
corr_matrix = centrality_df.corr(method='spearman')

# Plot correlation heatmap - already square by nature
plt.figure(figsize=(10, 10))  # Square figure
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, 
            annot_kws={"size": 12})
plt.tick_params(colors='black', labelsize=12)
plt.title('Spearman Correlation Between Centrality Measures', fontsize=16, color='black')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'centrality_correlation.png'), dpi=300, bbox_inches='tight')

# Step 11: Community detection using the Louvain method
print("Detecting communities...")
communities = g_filtered.community_multilevel()
membership = communities.membership

# Add community membership to the graph
g_filtered.vs["community"] = membership

# Get community sizes
community_sizes = pd.Series(membership).value_counts().sort_values(ascending=False)
print(f"Number of communities: {len(communities)}")
print(f"Top 10 community sizes: {community_sizes.head(10).tolist()}")

# Create a dataframe for communities
community_data = pd.DataFrame({
    'Community': range(len(community_sizes)),
    'Size': community_sizes.tolist()
})
community_data.to_csv(os.path.join(output_dir, 'community_sizes.csv'), index=False)

# Step 12: Find top nodes in each community based on betweenness and closeness
print("Finding top nodes in each community...")
top_nodes = {}
unique_communities = set(membership)

for comm in unique_communities:
    comm_nodes = [i for i, m in enumerate(membership) if m == comm]
    if not comm_nodes:
        continue
    
    # Get betweenness for community nodes
    comm_betweenness = [betweenness[i] for i in comm_nodes]
    top_betweenness_idx = comm_nodes[np.argmax(comm_betweenness)]
    
    # Get closeness for community nodes
    comm_closeness = [closeness[i] for i in comm_nodes]
    top_closeness_idx = comm_nodes[np.argmax(comm_closeness)]
    
    top_nodes[comm] = {
        'top_betweenness_node': top_betweenness_idx,
        'top_closeness_node': top_closeness_idx,
        'betweenness_value': betweenness[top_betweenness_idx],
        'closeness_value': closeness[top_closeness_idx],
        'size': len(comm_nodes)
    }

# Save top nodes data
top_nodes_df = pd.DataFrame([
    {
        'Community': comm,
        'Size': data['size'],
        'Top_Betweenness_Node': data['top_betweenness_node'],
        'Betweenness_Value': data['betweenness_value'],
        'Top_Closeness_Node': data['top_closeness_node'],
        'Closeness_Value': data['closeness_value']
    }
    for comm, data in top_nodes.items()
])
top_nodes_df.sort_values('Size', ascending=False, inplace=True)
top_nodes_df.to_csv(os.path.join(output_dir, 'top_community_nodes.csv'), index=False)

# Step 13: Visualize network as a standard graph (not geographic)
# Get top communities (by size)
top_communities = community_sizes.head(10).index.tolist()

# Create a NetworkX graph for visualization
print("Creating abstract network visualization...")
G = nx.Graph()

# Add all nodes first with their attributes
for i in range(g_filtered.vcount()):
    G.add_node(i, 
               betweenness=betweenness[i],
               closeness=closeness[i],
               community=membership[i])

# Then add edges
for edge in g_filtered.get_edgelist():
    G.add_edge(edge[0], edge[1])

# Set up the dark theme figure with square dimensions
plt.figure(figsize=(16, 16))  # Square figure
plt.style.use('dark_background')

# Color map for communities
num_communities = len(set(membership))
cmap = plt.cm.viridis
community_colors = {comm: cmap(i/num_communities) for i, comm in enumerate(set(membership))}

# Use a different layout that's not geographic - Fruchterman-Reingold
pos = nx.spring_layout(G, seed=42)

# Draw the graph with community colors
node_colors = [community_colors[G.nodes[i]['community']] for i in G.nodes()]
nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.3, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=5, node_color=node_colors, alpha=0.8)

# Highlight top nodes in each community
for comm in top_communities:
    if comm in top_nodes:
        betw_node = top_nodes[comm]['top_betweenness_node']
        if betw_node in G.nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[betw_node], node_size=50, 
                                  node_color='red', label=f'Top Betweenness in Comm {comm}')
        
        close_node = top_nodes[comm]['top_closeness_node']
        if close_node in G.nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[close_node], node_size=50, 
                                  node_color='yellow', label=f'Top Closeness in Comm {comm}')

plt.title('Abstract Network Visualization with Community Structure', fontsize=20, color='white')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'network_abstract.png'), dpi=300, bbox_inches='tight', facecolor='black')

# Step 14: Create a traditional heatmap of betweenness centrality (not geographic)
print("Creating betweenness centrality heatmap (non-geographic)...")

# Create a sample points grid for the heatmap
grid_size = int(np.sqrt(g_filtered.vcount()))  # square grid
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
xx, yy = np.meshgrid(x, y)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

# Assign node positions from the spring layout to grid points
sorted_betweenness = sorted(enumerate(betweenness), key=lambda x: x[1], reverse=True)
betweenness_grid = np.zeros((grid_size, grid_size))

# Fill the grid with betweenness values
for idx, (node_id, btw_value) in enumerate(sorted_betweenness):
    if idx >= len(grid_points):
        break
    grid_x, grid_y = grid_points[idx]
    i, j = int(grid_y * (grid_size-1)), int(grid_x * (grid_size-1))
    betweenness_grid[i, j] = btw_value

# Create a heatmap plot
plt.figure(figsize=(14, 14))  # Square figure
plt.style.use('dark_background')
plt.imshow(betweenness_grid, cmap='Purples', interpolation='bilinear')
plt.colorbar(label='Betweenness Centrality')
plt.title('Betweenness Centrality Heatmap (Non-Geographic)', fontsize=20, color='white')
plt.tick_params(colors='white')
plt.xlabel('Node Grid X', color='white', fontsize=14)
plt.ylabel('Node Grid Y', color='white', fontsize=14)
plt.tight_layout() 
plt.savefig(os.path.join(output_dir, 'betweenness_abstract_heatmap.png'), dpi=300, bbox_inches='tight', facecolor='black')

# Step 15: Generate summary statistics
print("Generating summary statistics...")
summary_stats = {
    'Number of Nodes': g_filtered.vcount(),
    'Number of Edges': g_filtered.ecount(),
    'Density': g_filtered.density(),
    'Average Degree': np.mean(g_filtered.degree()),
    'Diameter': g_filtered.diameter(),
    'Average Path Length': g_filtered.average_path_length(),
    'Number of Communities': len(communities),
    'Modularity': communities.modularity,
    'Largest Community Size': community_sizes.iloc[0],
    'Max Betweenness': max(betweenness),
    'Max Closeness': max(closeness),
    'Max Degree': max(g_filtered.degree())
}

with open(os.path.join(output_dir, 'network_summary_stats.txt'), 'w') as f:
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n")

# Bonus: Add a new visualization - square community size distribution plot
plt.figure(figsize=(12, 12))  # Square figure
plt.style.use('dark_background')
community_sizes_top20 = community_sizes.head(20)
plt.bar(range(len(community_sizes_top20)), community_sizes_top20.values, color='purple', alpha=0.7)
plt.xticks(range(len(community_sizes_top20)), community_sizes_top20.index, rotation=45)
plt.xlabel('Community ID', color='white', fontsize=14) 
plt.ylabel('Number of Nodes', color='white', fontsize=14)
plt.title('Top 20 Communities by Size', color='white', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tick_params(colors='white') 
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'community_size_distribution.png'), dpi=300, bbox_inches='tight', facecolor='black')

print("Analysis complete! All results saved to the 'output' directory.")