import numpy as np
import cv2 as cv
import json
import os

video_path = './data/06_lower.mp4'
json_path = '/Users/liumengyuan/Downloads/muscle_data/06lower_mask/axial_233.json'

# Get points_to_track array from json file
with open(json_path, "r") as f:
    mask_data = json.load(f)
    points_to_track = mask_data["shapes"][0]["points"]

points_to_mask = []

# Mouse click event callback function
def select_point(event, x, y, flags, param):
    global old_frame_display
    if event == cv.EVENT_LBUTTONDOWN:  # Left mouse button click
        points_to_track.append([x, y])
        print(f"Selected point: {x}, {y}")
        
        # Draw the selected point on the frame
        old_frame_display = cv.circle(old_frame_display, (x, y), 5, (0, 0, 255), -1)
        cv.imshow('Select Points', old_frame_display)

cap = cv.VideoCapture(video_path)

# Get the video's frames per second (fps)
fps = cap.get(cv.CAP_PROP_FPS)

# Calculate the frame delay in milliseconds (1000 ms = 1 second)
frame_delay = int(1000 / fps) if fps > 0 else 30

# Capture the first frame of the video
ret, old_frame = cap.read()
if not ret:
    print("Failed to grab first frame.")
    exit()

old_frame_display = old_frame.copy()

# Convert the selected points into a numpy array and reshape it
if len(points_to_track) > 0:
    p0 = np.array(points_to_track, dtype=np.float32).reshape(-1, 1, 2)
else:
    print("No points selected!")
    exit()

# Convert the first frame to grayscale
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Initialize color list for tracking points
color = [np.random.randint(0, 255, 3).tolist() for _ in range(1000)]

# Create a mask image to draw the tracking lines
mask = np.zeros_like(old_frame)

# MGE-based motion estimation method
def motion_gradient_estimation(point, old_gray, new_gray, threshold=1.0):
    # Compute the gradient in both directions (x and y) using Sobel operators
    grad_x = cv.Sobel(new_gray, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(new_gray, cv.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Get the point location
    x, y = int(point[0]), int(point[1])

    # Make sure the point is within the bounds of the image
    if x < 0 or y < 0 or x >= old_gray.shape[1] or y >= old_gray.shape[0]:
        return None

    # Get the gradient value at the given point
    grad_value = magnitude[y, x]

    # Return the point if its gradient magnitude is above the threshold
    if grad_value >= threshold:
        return (x, y)

    return None

# Tracking Loop with MGE-based motion gradient estimation
frame_counter = 233
while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # List to store points for this frame
    new_p1 = []

    # MGE-based tracking
    for p in p0:
        x, y = p.ravel()  # Ensure p is flattened correctly into (x, y)

        # Use MGE to track motion of the point
        new_point = motion_gradient_estimation([x, y], old_gray, frame_gray, threshold=10)

        if new_point:
            new_p1.append(new_point)

    if len(new_p1) > 0:
        for i, (new, old) in enumerate(zip(new_p1, p0)):
            old_x, old_y = old.ravel()  # Unpack old point correctly
            new_x, new_y = new  # Already a 2D point (x, y)

            # Draw points on the frame
            frame = cv.circle(frame, (int(new_x), int(new_y)), 1, tuple(color[i]), -1)

    # Overlay the mask with the tracks onto the current frame
    img = cv.add(frame, mask)
    cv.imshow('MGE Tracking', img)

    # Exit if 'Esc' key is pressed
    k = cv.waitKey(frame_delay) & 0xff
    if k == 27:
        break

    # Update the previous frame and points for the next iteration
    old_gray = frame_gray.copy()

    # Update points to be used in the next frame
    p0 = np.array(new_p1, dtype=np.float32).reshape(-1, 1, 2) if len(new_p1) > 0 else p0
    points_to_mask.append(p0)
    
    # Update mask to visualize the tracking of the points
    mask = np.zeros_like(frame)  # Reset mask for each frame

    frame_counter += 1

cv.destroyAllWindows()
