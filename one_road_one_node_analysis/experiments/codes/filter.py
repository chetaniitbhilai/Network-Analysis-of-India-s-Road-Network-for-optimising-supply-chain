import json
import geopandas as gpd
from shapely.geometry import LineString
import igraph as ig
import multiprocessing as mp

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
    g = ig.Graph()
    g.add_vertices(len(gdf))
    g.add_edges(edges)
    print(g.summary())
