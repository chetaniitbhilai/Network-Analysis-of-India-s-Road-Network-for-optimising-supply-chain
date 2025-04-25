import json
import geopandas as gpd
from shapely.geometry import LineString
import igraph as ig
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
# Load GeoJSON
with open("filtered_output_delhi.geojson", "r") as f:
    data = json.load(f)

# Filter valid LineStrings
valid_features = []
for feature in data['features']:
    coords = feature['geometry']['coordinates']
    if feature['geometry']['type'] == "LineString" and len(coords) >= 2:
        valid_features.append(feature)

# Create GeoDataFrame
geometries = [LineString(f["geometry"]["coordinates"]) for f in valid_features]
gdf = gpd.GeoDataFrame(valid_features, geometry=geometries)
gdf.reset_index(drop=True, inplace=True)
geom_list = gdf.geometry.values

# Parallel pairwise intersection checking
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
    g = ig.Graph()
    g.add_vertices(len(gdf))
    g.add_edges(edges)
    print(g.summary())
    
    # Drop nodes with degree <= 1
    deg = g.degree()
    keep = [i for i, d in enumerate(deg) if d > 1]
    g = g.induced_subgraph(keep)
    gdf = gdf.iloc[keep].reset_index(drop=True)
    geom_list = gdf.geometry.values

# Plot metric without normalization (for closeness)
def plot_metric(metric, name, normalize=True):
    if normalize:
        norm = mcolors.Normalize(vmin=min(metric), vmax=max(metric))
    else:
        norm = mcolors.Normalize(vmin=0, vmax=max(metric))

    cmap = plt.cm.plasma
    gdf[name] = metric

    fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
    ax.set_facecolor('black')

    lines = []
    colors = []
    for i, geom in enumerate(gdf.geometry):
        if isinstance(geom, LineString):
            lines.append(np.array(geom.coords))
            colors.append(cmap(norm(gdf.loc[i, name])))

    lc = LineCollection(lines, colors=colors, linewidths=0.8)
    ax.add_collection(lc)
    ax.autoscale()
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label(name.replace("_", " ").title(), color="white")
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    os.makedirs("result", exist_ok=True)
    plt.savefig(f"result/{name}_map.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {name} map to result/{name}_map.png")

# Plot degree distribution and log-log plot (moved out of plot_metric)
def plot_degree_distribution(graph):
    degrees = graph.degree()
    os.makedirs("result", exist_ok=True)

    # Linear Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), color='skyblue', edgecolor='black')
    plt.title("Degree Distribution Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("result/degree_histogram.png", dpi=300, bbox_inches="tight")
    print("Saved degree histogram to result/degree_histogram.png")

    # Log-log plot
    degree_counts = np.bincount(degrees)
    degrees_nonzero = np.nonzero(degree_counts)[0]
    counts_nonzero = degree_counts[degrees_nonzero]

    plt.figure(figsize=(10, 6))
    plt.loglog(degrees_nonzero, counts_nonzero, marker='o', linestyle='None', color='orange')
    plt.title("Log-Log Degree Distribution")
    plt.xlabel("Degree (log)")
    plt.ylabel("Frequency (log)")
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.savefig("result/degree_loglog.png", dpi=300, bbox_inches="tight")
    print("Saved log-log degree distribution to result/degree_loglog.png")

# Centrality Metrics
plot_metric(g.betweenness(), "betweenness")
plot_metric(g.closeness(normalized=False), "closeness", normalize=False)
plot_metric(g.eigenvector_centrality(), "eigenvector")
plot_metric(g.pagerank(), "pagerank")
plot_degree_distribution(g)

# Community Detection using Louvain method
def plot_communities(graph, gdf):
    communities = graph.community_multilevel()
    membership = communities.membership
    gdf["community"] = membership

    # Assign a color to each community
    num_communities = max(membership) + 1
    cmap = plt.cm.get_cmap('tab20', num_communities)
    colors = [cmap(membership[i]) for i in range(len(membership))]

    fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
    ax.set_facecolor('black')

    lines = [np.array(geom.coords) for geom in gdf.geometry]
    lc = LineCollection(lines, colors=colors, linewidths=0.8)
    ax.add_collection(lc)
    ax.autoscale()
    ax.axis("off")

    os.makedirs("result", exist_ok=True)
    plt.savefig("result/community_map.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved community map to result/community_map.png")

    return communities

# Run and plot communities
communities = plot_communities(g, gdf)

# Community Detection with highlight on top-betweenness nodes per community
def plot_communities(graph, gdf):
    communities = graph.community_multilevel()
    membership = communities.membership
    gdf["community"] = membership

    # Assign a color to each community
    num_communities = max(membership) + 1
    cmap = plt.cm.get_cmap('tab20', num_communities)
    colors = [cmap(membership[i]) for i in range(len(membership))]

    # Get betweenness
    betweenness = graph.betweenness()

    # Find highest-betweenness node in each community
    top_nodes = []
    for comm_id in range(num_communities):
        nodes_in_comm = [i for i, m in enumerate(membership) if m == comm_id]
        if nodes_in_comm:
            top_node = max(nodes_in_comm, key=lambda n: betweenness[n])
            top_nodes.append(top_node)

    fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
    ax.set_facecolor('black')

    lines = [np.array(geom.coords) for geom in gdf.geometry]
    lc = LineCollection(lines, colors=colors, linewidths=0.8)
    ax.add_collection(lc)

    # Plot top-betweenness nodes
    for idx in top_nodes:
        geom = gdf.geometry.iloc[idx]
        if isinstance(geom, LineString):
            x, y = np.array(geom.coords)[len(geom.coords) // 2]  # mid-point of line
            ax.plot(x, y, 'ro', markersize=6, label='Top Betweenness' if idx == top_nodes[0] else "")

    ax.autoscale()
    ax.axis("off")

    os.makedirs("result", exist_ok=True)
    plt.legend(loc='lower right', frameon=False, facecolor='black', labelcolor='white')
    plt.savefig("result/community_map_betw.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Saved community map to result/community_map_with_bet.png")

    return communities

communities = plot_communities(g, gdf)



def save_top_betweenness_nodes(graph, gdf, communities, k=3, filename="result/top_betweenness_per_community.csv"):
    betweenness = graph.betweenness()
    membership = communities.membership
    output_rows = []

    for comm_id in range(max(membership) + 1):
        nodes_in_comm = [i for i, m in enumerate(membership) if m == comm_id]
        if not nodes_in_comm:
            continue

        top_k = sorted(nodes_in_comm, key=lambda n: betweenness[n], reverse=True)[:k]
        for rank, node_idx in enumerate(top_k):
            geom = gdf.geometry.iloc[node_idx]
            if isinstance(geom, LineString):
                midpoint = np.array(geom.coords)[len(geom.coords) // 2]
                output_rows.append({
                    "community": comm_id,
                    "node_index": node_idx,
                    "rank_in_community": rank + 1,
                    "x": midpoint[0],
                    "y": midpoint[1]
                })

    df = pd.DataFrame(output_rows)
    os.makedirs("result", exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Saved top {k} betweenness nodes per community to {filename}")

save_top_betweenness_nodes(g, gdf, communities)
