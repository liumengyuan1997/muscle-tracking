import numpy as np
import os
import cv2 as cv

def generate_tracking_mask(points_to_mask, frame_size, output_dir):
    height, width = frame_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 233
    for i, points in enumerate(points_to_mask):
        mask = np.zeros((height, width), dtype=np.uint8)

        if len(points) < 3:
            print(f"Frame {i}: points not enough to create closed area, skip")
            continue

        # Flatten points to (N, 2)
        flat_pts = np.array(points, dtype=np.int32).reshape(-1, 2)

        # Step 1: Compute Convex Hull to make it a closed contour
        hull = cv.convexHull(flat_pts)

        # Step 2: Fill mask using the convex hull
        cv.fillPoly(mask, [hull], 255)

        # Save
        filename = os.path.join(output_dir, f"axial_{counter:03d}.png")
        cv.imwrite(filename, mask)
        counter += 1