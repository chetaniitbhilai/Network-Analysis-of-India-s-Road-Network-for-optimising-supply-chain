import geopandas as gpd
import json
import ijson
from decimal import Decimal
from shapely.geometry import Polygon, Point

# Load Delhi boundary GeoJSON and create polygon
delhi = gpd.read_file("Delhi_Boundary.geojson")
delhi_coords = list(delhi.geometry.iloc[0].exterior.coords)
delhi_polygon = Polygon(delhi_coords)

input_file = "try.geojson"
output_file = "filtered_output_delhi.geojson"
chunk_size = 10000

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

def filter_geojson_by_polygon():
    print(f"Processing {input_file} for Delhi boundary...")

    features_processed = 0
    features_kept = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        with open(input_file, 'r', encoding='utf-8') as in_f:
            buffer = ""
            for line in in_f:
                buffer += line
                if '"features": [' in buffer:
                    break
        header_end = buffer.find('"features": [') + len('"features": [')
        out_f.write(buffer[:header_end])

        is_first = True
        with open(input_file, 'r', encoding='utf-8') as in_f:
            features = ijson.items(in_f, 'features.item')

            for feature in features:
                features_processed += 1

                if feature["geometry"]["type"] != "LineString":
                    continue

                coordinates = feature["geometry"]["coordinates"]
                filtered_coords = []

                for coord in coordinates:
                    lon = float(coord[0])
                    lat = float(coord[1])
                    if delhi_polygon.contains(Point(lon, lat)):
                        filtered_coords.append([lon, lat])

                if not filtered_coords:
                    continue

                feature["geometry"]["coordinates"] = filtered_coords
                features_kept += 1

                if not is_first:
                    out_f.write(',\n')
                else:
                    is_first = False

                json.dump(feature, out_f, cls=DecimalEncoder)

                if features_processed % chunk_size == 0:
                    print(f"Processed {features_processed}, kept {features_kept}")

        out_f.write(']}\n')

    print(f"✅ Done! Processed {features_processed}, Kept {features_kept}")

filter_geojson_by_polygon()
