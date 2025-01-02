import cv2
import json
import numpy as np
import os

input_folder = "/Users/liumengyuan/Desktop/muscle_seg/sample_mask_json"
output_folder = "/Users/liumengyuan/Desktop/muscle_seg/sample_mask"

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".png"):
        base_name = os.path.splitext(file_name)[0]

        image_path = os.path.join(input_folder, file_name)
        json_path = os.path.join(input_folder, f"{base_name}.json")

        if not os.path.exists(json_path):
            print(f"Cannot find corresponding json file: {json_path}, skip {file_name}")
            continue

        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Cannot read original image: {image_path}, skip")
            continue

        # Create Transparent Overlay
        overlay = original_image.copy()

        with open(json_path, "r") as f:
            mask_data = json.load(f)

        # Draw mask outline to transparent overlay
        for shape in mask_data["shapes"]:
            points = np.array(shape["points"], dtype=np.int32)
            cv2.polylines(overlay, [points], isClosed=True, color=(0, 0, 255), thickness=1)

        # add transparent overlay to original img
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0, original_image)

        output_path = os.path.join(output_folder, f"{base_name}_contours.png")
        cv2.imwrite(output_path, original_image)

        print(f"Complete processing and Save to: {output_path}")

print("All files completed!")