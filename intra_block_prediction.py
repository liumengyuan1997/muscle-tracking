import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json
from utils.generate_mask import generate_tracking_mask
from utils.good_points import find_nearest_edge_point

video_path = './data/06_upper.mp4'
json_path = '/Users/liumengyuan/Downloads/muscle_data/06lower_mask/axial_233.json'

# Get points_to_track array from json file
with open(json_path, "r") as f:
    mask_data = json.load(f)
    points_to_track = mask_data["shapes"][0]["points"]

points_to_mask = []

# Mouse click event callback function
def select_point(event, x, y, flags, param):
    global old_frame_display, points_to_track
    if event == cv.EVENT_LBUTTONDOWN:  # Left mouse button click
        points_to_track.append([x, y])
        print(f"Selected point: {x}, {y}")
        
        # Draw the selected point on the frame
        old_frame_display = cv.circle(old_frame_display, (x, y), 1, (0, 0, 255), -1)
        cv.imshow('Select Points', old_frame_display)

cap = cv.VideoCapture(video_path)

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

p0_good = cv.goodFeaturesToTrack(
    old_gray,            # Input grayscale image
    maxCorners=10000,      # Maximum number of corners to return
    qualityLevel=0.02,    # Quality level (between 0 and 1)
    minDistance=1,       # Minimum possible Euclidean distance between returned corners
    blockSize=4,         # Size of the neighborhood considered for corner detection
    useHarrisDetector=False,  # Whether to use Harris corner detector
    k=0.04               # Free parameter of Harris detector (if enabled)
)

# new_p0 = []
# for p in p0:
#     x, y = p.ravel()
#     nearest_edge_point = find_nearest_edge_point([x, y], p0_good)
#     new_p0.append(nearest_edge_point)

new_p0 = p0

# convert p0 to numpy and change the shape
p0 = np.array(new_p0, dtype=np.float32).reshape(-1, 1, 2)
points_to_mask.append(p0)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

use_block_prediction = True  # use block-based prediction or not
block_size = 4
half_block = block_size // 2

# Random colors to display the tracking lines
color = [np.random.randint(0, 255, 3).tolist() for _ in range(1000)]

# Create a mask image to draw the optical flow tracks
mask = np.zeros_like(old_frame)

frame_counter = 1
# Start tracking the points through the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # if frame_counter % 45 == 0:
    #     old_frame_display = frame.copy()
    #     points_to_track = []
    #     cv.namedWindow('Select Points')
    #     cv.setMouseCallback('Select Points', select_point)
        
    #     while True:
    #         cv.imshow('Select Points', old_frame_display)
    #         k = cv.waitKey(1) & 0xFF
    #         if k == 13:  # "Enter" key
    #             break

    #     if len(points_to_track) > 0:
    #         p0 = np.array(points_to_track, dtype=np.float32).reshape(-1, 1, 2)
    #     else:
    #         print("No points selected!")
    #         exit()
    #     # convert p0 to numpy and change the shape
    #     p0 = np.array(p0, dtype=np.float32).reshape(-1, 1, 2)

    # Calculate optical flow for the selected points
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points where optical flow is successfully calculated
    if p1 is not None:
        good_old = p0[st == 1]

        if use_block_prediction:
            smoothed_good_new = []

            for i, point in enumerate(good_old.reshape(-1, 2)):
                x, y = int(point[0]), int(point[1])
                flows = []

                for dx in range(-half_block, half_block + 1):
                    for dy in range(-half_block, half_block + 1):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < old_gray.shape[1]) and (0 <= ny < old_gray.shape[0]):
                            sub_p0 = np.array([[[nx, ny]]], dtype=np.float32)
                            sub_p1, sub_st, _ = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, sub_p0, None, **lk_params)

                            if sub_p1 is not None and sub_st[0][0] == 1:
                                flow = sub_p1[0][0] - sub_p0[0][0]
                                flows.append(flow)

                if flows:
                    avg_flow = np.mean(flows, axis=0)
                    new_point = point + avg_flow
                    smoothed_good_new.append(new_point)
                else:
                    smoothed_good_new.append(point)  # fallback to original

            new_p1 = np.array(smoothed_good_new, dtype=np.float32).reshape(-1, 1, 2)
        else:
            good_new = p1[st == 1]
            new_p1 = np.array(good_new, dtype=np.float32).reshape(-1, 1, 2)
    else:
        print("No good points to track.")
        break

    # Threshold for distance to add a new point (you can adjust this value)
    distance_threshold = 15
    merge_distance = 5

    # # # New list to store updated points, including inserted midpoints
    # new_p1 = []

    # # Calculate distances between neighbors
    # for i in range(len(smoothed_good_new)):
    #     point1 = smoothed_good_new[i]
    #     point2 = smoothed_good_new[(i + 1) % len(smoothed_good_new)]  # connect first and last

    #     # Calculate the Euclidean distance between two points
    #     distance = np.linalg.norm(point2 - point1)

    #     if distance < merge_distance:
    #         merged_point = (point1 + point2) / 2.0
    #         if new_p1 and not np.array_equal(new_p1[-1], merged_point):
    #             new_p1[-1] = merged_point
    #     else:
    #         midpoint = (point1 + point2) / 2.0
    #         if new_p1 and not np.array_equal(new_p1[-1], midpoint):
    #             new_p1.append(midpoint)

    #     new_p1.append(point2)

    # # Convert the updated list of points into the appropriate shape for tracking (N, 1, 2)
    # new_p1 = np.array(new_p1, dtype=np.float32).reshape(-1, 1, 2)

    if len(new_p1) > 0:
        for i, (new, old) in enumerate(zip(new_p1, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv.circle(frame, (int(a), int(b)), 1, tuple(color[i]), -1)
    else:
        print("No good points to track.")

    # Overlay the mask with the tracks onto the current frame
    img = cv.add(frame, mask)
    cv.imshow('Optical Flow Tracking', img)

    # Exit if 'Esc' key is pressed
    k = cv.waitKey(300) & 0xff
    if k == 27:
        break

    # Update the previous frame and points for the next iteration
    old_gray = frame_gray.copy()
    p0 = new_p1
    points_to_mask.append(p0)
    frame_counter += 1

generate_tracking_mask(points_to_mask, (144, 544), "./mask_ibp")

cv.destroyAllWindows()

