import json
import ijson
from decimal import Decimal


# Define bounding box limits
LAT_MIN, LAT_MAX = 28.24, 28.28  # 23°52'N to 31°28'N
LON_MIN, LON_MAX = 76.5, 76.9      # 77°3'E to 84°39'E

input_file = "try.geojson"
output_file = "filtered_output_2_small.geojson"
chunk_size = 10000  # Process this many features at a time

# Custom JSON encoder to handle Decimal objects
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def filter_geojson_without_dependencies():
    """Process GeoJSON file without pandas or numpy dependencies"""
    
    print(f"Starting to process {input_file}...")
    
    features_processed = 0
    features_kept = 0
    
    # Create output file and write header
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Get header information from the input file
        with open(input_file, 'r', encoding='utf-8') as in_f:
            # Read the first part to get the header structure
            buffer = ""
            for line in in_f:
                buffer += line
                if '"features": [' in buffer:
                    break
        
        # Extract and write the header (everything before features array)
        header_end = buffer.find('"features": [') + len('"features": [')
        out_f.write(buffer[:header_end])
        
        # Process features one by one
        is_first = True
        with open(input_file, 'r', encoding='utf-8') as in_f:
            features = ijson.items(in_f, 'features.item')
            
            for feature in features:
                features_processed += 1
                
                # Only process LineString features
                if feature["geometry"]["type"] != "LineString":
                    continue
                
                # Get coordinates and filter them
                coordinates = feature["geometry"]["coordinates"]
                filtered_coords = []
                
                for coord in coordinates:
                    # Convert Decimal to float if needed
                    lon = float(coord[0]) if isinstance(coord[0], Decimal) else coord[0]
                    lat = float(coord[1]) if isinstance(coord[1], Decimal) else coord[1]
                    
                    # Check if outside the bounding box
                    if (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
                        filtered_coords.append([lon, lat])
                
                # Skip if all coordinates were filtered out
                if not filtered_coords:
                    continue
                
                # Update feature with filtered coordinates
                feature["geometry"]["coordinates"] = filtered_coords
                features_kept += 1
                
                # Write feature to output
                if not is_first:
                    out_f.write(', \n')
                else:
                    is_first = False
                
                # Use custom encoder to handle Decimal objects
                json.dump(feature, out_f, cls=DecimalEncoder)
                
                # Display progress
                if features_processed % chunk_size == 0:
                    print(f"Processed {features_processed} features, kept {features_kept}")
        
        # Close the features array and the GeoJSON object
        out_f.write(']}\n')
    
    print(f"Completed! Processed {features_processed} features, kept {features_kept}")

# Run the function
filter_geojson_without_dependencies()