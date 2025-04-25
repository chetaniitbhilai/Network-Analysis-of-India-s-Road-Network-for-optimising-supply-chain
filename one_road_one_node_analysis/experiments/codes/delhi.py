import geopandas as gpd
import matplotlib.pyplot as plt

# Load Delhi boundary GeoJSON file
delhi = gpd.read_file("Delhi_Boundary.geojson")

# Plot the boundary
delhi.plot(edgecolor='black', facecolor='none')
plt.title("Delhi Boundary")
plt.show()

# Extract coordinates
coords = delhi.geometry.iloc[0].exterior.coords[:]
print("Boundary Coordinates:")
print(coords)
