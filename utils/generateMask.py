import json
import numpy as np
import cv2
import os

# Function to process each JSON file and create a mask
def process_json_to_mask(input_folder, output_folder, image_shape):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all JSON files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(input_folder, filename)
            output_mask_path = os.path.join(output_folder, filename.replace('.json', '_mask.png'))
            
            # Load the JSON data
            with open(json_path, 'r') as file:
                data = json.load(file)

            # Create a blank mask
            mask = np.zeros(image_shape, dtype=np.uint8)

            # Process each shape and draw polygons
            for shape in data.get('shapes', []):
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], color=255)

            # Save the mask to the output folder
            cv2.imwrite(output_mask_path, mask)
            print(f"Mask saved: {output_mask_path}")

# Example usage
input_folder = '/Users/liumengyuan/Downloads/muscle_data/06total'  # Replace with the path to your folder containing JSON files
output_folder = '/Users/liumengyuan/Downloads/muscle_data/06totalmasks'  # Replace with the path to save the masks
image_shape = (144, 544)  # Replace with your image dimensions

process_json_to_mask(input_folder, output_folder, image_shape)