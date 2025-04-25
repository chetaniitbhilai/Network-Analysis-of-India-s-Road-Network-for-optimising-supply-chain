import json
import geopandas as gpd
from shapely.geometry import LineString
import multiprocessing as mp
import json
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import LineString
import igraph as ig
from collections import Counter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import pandas as pd
import os

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Step 1: Load GeoJSON
with open("filtered_output_2_small.geojson", "r") as f:
    data = json.load(f)

# Step 2: Filter valid LineStrings
valid_features = []
for feature in data['features']:
    coords = feature['geometry']['coordinates']
    if feature['geometry']['type'] == "LineString" and len(coords) >= 2:
        valid_features.append(feature)

# Step 3: Create GeoDataFrame
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
    chunk_size = len(geom_list) // n_workers
    ranges = [(i * chunk_size, (i + 1) * chunk_size if i != n_workers - 1 else len(geom_list)) for i in range(n_workers)]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(check_chunk, ranges)

    edges = [e for sub in results for e in sub]
    return edges

if __name__ == "__main__":
    edges = parallel_intersection()

    # Step 5: Build graph
    graph = ig.Graph()
    graph.add_vertices(len(gdf))
    graph.add_edges(edges)
    print("Before filtering:", graph.summary())

    # Step 6: Remove nodes with degree 1 or 2
    degrees = graph.degree()
    to_remove = [i for i, d in enumerate(degrees) if d == 1 or d == 0]
    graph.delete_vertices(to_remove)

    print("After filtering:", graph.summary())


# Set dark style for plots
plt.style.use('dark_background')
custom_params = {
    "axes.facecolor": "#1a1a1a",
    "figure.facecolor": "#121212",
    "grid.color": "#444444",
    "text.color": "#ffffff",
    "axes.labelcolor": "#ffffff",
    "xtick.color": "#ffffff",
    "ytick.color": "#ffffff",
    "axes.grid": False
}
plt.rcParams.update(custom_params)

# 1. Degree Distribution Analysis
def analyze_degree_distribution(graph):
    degrees = graph.degree()
    degree_counts = Counter(degrees)
    
    # Sort by degree
    x = list(sorted(degree_counts.keys()))
    y = [degree_counts[k] for k in x]
    
    # Create degree distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot degree distribution - simplified
    ax1.bar(x, y, color='#41d6c3')
    ax1.set_title('Degree Distribution', fontsize=14)
    ax1.set_xlabel('Degree (k)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    
    # Plot log-log degree distribution to check for power law
    nonzero_x = [d for d in x if d > 0]
    nonzero_y = [degree_counts[d] for d in nonzero_x]
    
    if nonzero_x and nonzero_y:
        ax2.loglog(nonzero_x, nonzero_y, 'o-', color='#41d6c3', alpha=0.7, markersize=8)
        ax2.set_title('Log-Log Degree Distribution', fontsize=14)
        ax2.set_xlabel('Log(Degree)', fontsize=12)
        ax2.set_ylabel('Log(Count)', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/degree_distribution.png', dpi=300, bbox_inches='tight', facecolor="#121212")
    plt.close()
    
    # Print degree statistics
    print(f"Network Degree Statistics:")
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Maximum degree: {max(degrees)}")
    print(f"Minimum degree: {min(degrees)}")
    
    return degrees

# 2. Centrality Analysis - SIMPLIFIED FOR COMPATIBILITY
def calculate_centralities(graph):
    print("Calculating centrality measures...")
    
    # Calculate centrality measures
    degree_centrality = graph.degree()
    
    # Fix: Remove parameters that could cause errors
    betweenness_centrality = graph.betweenness(directed=False)
    
    # Try closeness centrality with error handling
    try:
        closeness_centrality = graph.closeness(normalized=True)
    except TypeError:
        # If normalized parameter causes an error, try without it
        closeness_centrality = graph.closeness()
    
    # Calculate eigenvector centrality with error handling
    try:
        eigenvector_centrality = graph.eigenvector_centrality(directed=False)
    except Exception as e:
        print(f"Eigenvector centrality calculation failed: {e}")
        print("Using degree centrality as a fallback")
        eigenvector_centrality = degree_centrality
    
    # Create centrality dataframe
    centrality_measures = {
        'degree': degree_centrality,
        'betweenness': betweenness_centrality,
        'closeness': closeness_centrality,
        'eigenvector': eigenvector_centrality
    }
    
    # SIMPLIFIED: Use direct matplotlib plotting instead of seaborn's kde
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Custom colors for each measure - simplified
    colors = ['#41d6c3', '#c341d6', '#d6c341', '#d64141']
    
    for i, (name, values) in enumerate(centrality_measures.items()):
        ax = axes[i]
        # Simple histogram instead of complex kde
        ax.hist(values, bins=30, color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.set_title(f'{name.capitalize()} Centrality Distribution', fontsize=14)
        ax.set_xlabel(f'{name.capitalize()} Centrality', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/centrality_distributions.png', dpi=300, bbox_inches='tight', facecolor="#121212")
    plt.close()
    
    # Calculate correlations between centrality measures
    centrality_df = pd.DataFrame(centrality_measures)
    correlation = centrality_df.corr()
    
    # Simple heatmap without seaborn
    plt.figure(figsize=(10, 8))
    im = plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add correlation values
    for i in range(correlation.shape[0]):
        for j in range(correlation.shape[1]):
            text = plt.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                           ha="center", va="center", color="white" if abs(correlation.iloc[i, j]) < 0.5 else "black")
    
    plt.colorbar(im, label='Correlation Coefficient')
    plt.xticks(range(len(correlation)), correlation.columns, rotation=45)
    plt.yticks(range(len(correlation)), correlation.index)
    plt.title('Correlation Between Centrality Measures', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/centrality_correlation.png', dpi=300, bbox_inches='tight', facecolor="#121212")
    plt.close()
    
    return centrality_measures

# 3. Community Detection - SIMPLIFIED
def detect_communities(graph):
    print("Detecting communities...")
    
    # Apply multiple community detection algorithms with error handling
    communities_methods = []
    method_names = []
    modularities = []
    
    # Try Louvain method (modularity optimization)
    try:
        louvain_communities = graph.community_multilevel()
        louvain_modularity = graph.modularity(louvain_communities)
        communities_methods.append(louvain_communities)
        method_names.append("Louvain")
        modularities.append(louvain_modularity)
        print(f"Louvain method: {len(louvain_communities)} communities, modularity: {louvain_modularity:.4f}")
    except Exception as e:
        print(f"Louvain method failed: {e}")
    
    # Try Label propagation
    try:
        label_prop_communities = graph.community_label_propagation()
        label_prop_modularity = graph.modularity(label_prop_communities)
        communities_methods.append(label_prop_communities)
        method_names.append("Label Propagation")
        modularities.append(label_prop_modularity)
        print(f"Label propagation: {len(label_prop_communities)} communities, modularity: {label_prop_modularity:.4f}")
    except Exception as e:
        print(f"Label propagation method failed: {e}")
    
    # If all methods failed, use a fallback method
    if not communities_methods:
        try:
            # Fallback to connected components
            fallback_communities = graph.components()
            fallback_modularity = graph.modularity(fallback_communities)
            communities_methods.append(fallback_communities)
            method_names.append("Connected Components")
            modularities.append(fallback_modularity)
            print(f"Connected Components: {len(fallback_communities)} communities, modularity: {fallback_modularity:.4f}")
        except Exception as e:
            print(f"All community detection methods failed: {e}")
            return None, "None"
    
    # Select the method with highest modularity
    if modularities:
        best_idx = modularities.index(max(modularities))
        best_method = communities_methods[best_idx]
        best_method_name = method_names[best_idx]
        
        print(f"Using {best_method_name} method with highest modularity ({modularities[best_idx]:.4f})")
    else:
        print("No community detection method worked. Cannot perform community analysis.")
        return None, "None"
    
    # Plot community size distribution - simplified
    community_sizes = [len(comm) for comm in best_method]
    
    plt.figure(figsize=(10, 6))
    plt.hist(community_sizes, bins=20, color="#5cbae6", edgecolor="#2980b9")
    plt.title(f'Community Size Distribution ({best_method_name} method)', fontsize=14)
    plt.xlabel('Community Size', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add summary statistics
    textstr = '\n'.join((
        f'Communities: {len(best_method)}',
        f'Modularity: {modularities[best_idx]:.4f}',
        f'Avg Size: {np.mean(community_sizes):.1f}',
        f'Max Size: {max(community_sizes)}',
        f'Min Size: {min(community_sizes)}'
    ))
    
    props = dict(boxstyle='round', facecolor='#333333', alpha=0.8)
    plt.text(0.75, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.savefig('results/community_size_distribution.png', dpi=300, bbox_inches='tight', facecolor="#121212")
    plt.close()
    
    return best_method, best_method_name

# 4. Network Visualization with spatial data - SIMPLIFIED
def visualize_network(graph, gdf, centrality_measures, communities, community_method_name):
    print("Creating network visualizations...")
    
    # Skip if communities detection failed
    if communities is None:
        print("Skipping community visualization as community detection failed.")
        community_vis = False
    else:
        community_vis = True
        # 4.1 Create a complex visualization combining communities with centrality
        membership = []
        for i in range(graph.vcount()):
            for j, comm in enumerate(communities):
                if i in comm:
                    membership.append(j)
                    break
        
        # Create a color map for communities - simplified to use standard colormap
        unique_communities = len(set(membership))
        colormap = plt.cm.tab20 if unique_communities <= 20 else plt.cm.viridis
        
        # Plot the spatial network with communities
        plt.figure(figsize=(16, 12))
        
        # Plot the lines with community colors
        for i, feature in enumerate(gdf.geometry):
            if i < len(membership):  # Safety check
                comm_idx = membership[i]
                color_val = comm_idx % unique_communities / unique_communities
                plt.plot(*feature.xy, color=colormap(color_val), linewidth=1.5, alpha=0.7)
        
        # Add legend for communities (limit to top 10 by size if there are many)
        comm_sizes = [len(comm) for comm in communities]
        top_communities = sorted(range(len(comm_sizes)), key=lambda i: comm_sizes[i], reverse=True)[:min(10, len(comm_sizes))]
        
        legend_patches = []
        for i, comm_idx in enumerate(top_communities):
            color_val = comm_idx % unique_communities / unique_communities
            patch = mpatches.Patch(color=colormap(color_val), 
                                label=f'Community {comm_idx+1} (size: {comm_sizes[comm_idx]})')
            legend_patches.append(patch)
        
        plt.legend(handles=legend_patches, loc='upper right', title=f"{community_method_name} Communities")
        plt.title(f'Geospatial Network Visualization with Communities', fontsize=16)
        plt.grid(False)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/geospatial_network_communities.png', dpi=300, bbox_inches='tight', facecolor="#121212")
        plt.close()
    
    # 4.2 Create a betweenness centrality visualization
    plt.figure(figsize=(16, 12))
    
    # Normalize betweenness for coloring
    betweenness = centrality_measures['betweenness']
    max_betweenness = max(betweenness) if betweenness else 1
    
    # Create a colormap
    cmap = plt.cm.viridis
    
    # Plot lines with color based on betweenness
    for i, feature in enumerate(gdf.geometry):
        if i < len(betweenness):  # Safety check
            if max_betweenness > 0:  # Avoid division by zero
                norm_value = betweenness[i] / max_betweenness
            else:
                norm_value = 0
                
            line_color = cmap(norm_value)
            line_width = 1 + 3 * norm_value  # Scale line width based on betweenness
            
            plt.plot(*feature.xy, color=line_color, linewidth=line_width, alpha=0.7)
    
    # Add a colorbar
    norm = Normalize(vmin=0, vmax=max_betweenness)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', pad=0.01)
    cbar.set_label('Betweenness Centrality', size=12)
    
    plt.title('Geospatial Network - Betweenness Centrality', fontsize=16)
    plt.grid(False)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/geospatial_betweenness.png', dpi=300, bbox_inches='tight', facecolor="#121212")
    plt.close()
    
    # 4.3 Create a degree centrality visualization
    plt.figure(figsize=(16, 12))
    
    # Normalize degree for coloring
    degrees = centrality_measures['degree']
    max_degree = max(degrees) if degrees else 1
    
    # Create a colormap - use a different one for variety
    cmap = plt.cm.plasma
    
    # Plot lines with color based on degree
    for i, feature in enumerate(gdf.geometry):
        if i < len(degrees):  # Safety check
            norm_value = degrees[i] / max_degree if max_degree > 0 else 0
            line_color = cmap(norm_value)
            line_width = 1 + 3 * norm_value  # Scale line width based on degree
            
            plt.plot(*feature.xy, color=line_color, linewidth=line_width, alpha=0.7)
    
    # Add a colorbar
    norm = Normalize(vmin=0, vmax=max_degree)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', pad=0.01)
    cbar.set_label('Degree Centrality', size=12)
    
    plt.title('Geospatial Network - Degree Centrality', fontsize=16)
    plt.grid(False)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/geospatial_degree.png', dpi=300, bbox_inches='tight', facecolor="#121212")
    plt.close()

# Network metrics summary
def network_metrics_summary(graph):
    # Calculate overall network metrics with error handling
    try:
        avg_path_length = graph.average_path_length(directed=False, unconn=True)
    except:
        try:
            # Try without unconn parameter
            avg_path_length = graph.average_path_length(directed=False)
        except:
            avg_path_length = "N/A"
    
    try:
        diameter = graph.diameter(directed=False, unconn=True)
    except:
        try:
            # Try without unconn parameter
            diameter = graph.diameter(directed=False)
        except:
            diameter = "N/A"
    
    density = graph.density()
    
    try:
        clustering_coef = graph.transitivity_avglocal_undirected()
    except:
        try:
            # Try global transitivity as fallback
            clustering_coef = graph.transitivity_undirected()
        except:
            clustering_coef = "N/A"
    
    # Identify articulation points (cut vertices)
    try:
        cut_vertices = graph.cut_vertices()
    except:
        cut_vertices = []
    
    # Identify connected components
    components = graph.components()
    component_sizes = [len(comp) for comp in components]
    
    # Create a summary plot - simplified to avoid rendering issues
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    
    # Create a text summary with dark background styling
    textstr = '\n'.join((
        f'NETWORK SUMMARY METRICS',
        f'------------------------',
        f'Nodes: {graph.vcount()}',
        f'Edges: {graph.ecount()}',
        f'Density: {density:.4f}',
        f'Average Path Length: {avg_path_length if isinstance(avg_path_length, str) else avg_path_length:.4f}',
        f'Diameter: {diameter}',
        f'Clustering Coefficient: {clustering_coef if isinstance(clustering_coef, str) else clustering_coef:.4f}',
        f'Components: {len(components)}',
        f'Largest Component Size: {max(component_sizes) if component_sizes else 0}',
        f'Cut Vertices: {len(cut_vertices)}',
    ))
    
    props = dict(boxstyle='round', facecolor='#333333', alpha=0.9)
    plt.text(0.5, 0.5, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='center', bbox=props,
            family='monospace')
    
    plt.tight_layout()
    plt.savefig('results/network_metrics_summary.png', dpi=300, bbox_inches='tight', facecolor="#121212")
    plt.close()
    
    # Return key metrics
    return {
        'avg_path_length': avg_path_length,
        'diameter': diameter, 
        'density': density,
        'clustering': clustering_coef,
        'components': len(components),
        'cut_vertices': len(cut_vertices)
    }

# Now run all analyses in sequence
# Make sure this code runs after your graph and GeoDataFrame are created
if 'graph' in locals() and 'gdf' in locals():
    print("\n" + "="*50)
    print("Starting comprehensive network analysis...")
    print("="*50)
    
    # Step 1: Analyze degree distribution
    print("\n1. DEGREE DISTRIBUTION ANALYSIS")
    degrees = analyze_degree_distribution(graph)
    
    # Step 2: Calculate centrality measures
    print("\n2. CENTRALITY ANALYSIS")
    centrality_measures = calculate_centralities(graph)
    
    # Step 3: Detect communities
    print("\n3. COMMUNITY DETECTION")
    communities, community_method_name = detect_communities(graph)
    
    # Step 4: Network metrics summary
    print("\n4. NETWORK METRICS SUMMARY")
    metrics = network_metrics_summary(graph)
    
    # Step 5: Visualize the network
    print("\n5. NETWORK VISUALIZATION")
    visualize_network(graph, gdf, centrality_measures, communities, community_method_name)
    
    print("\n" + "="*50)
    print("Network analysis complete. All visualizations have been saved to the 'results' directory.")
    print("="*50)
else:
    print("Error: Graph or GeoDataFrame not found. Make sure to run the initial data loading code first.")