import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
# from utils import generate_tracking_mask

video_path = './data/test_lower_longus.mp4'

# Create a list to store the points clicked by the user
points_to_track = []
points_to_mask = []

# Mouse click event callback function
def select_point(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  # Left mouse button click
        points_to_track.append([x, y])
        print(f"Selected point: {x}, {y}")

cap = cv.VideoCapture(video_path)

# Get the video's frames per second (fps)
fps = cap.get(cv.CAP_PROP_FPS)

# Calculate the frame delay in milliseconds (1000 ms = 1 second)
if fps > 0:
    frame_delay = int(1000 / fps)
else:
    # Default to 30ms if fps couldn't be retrieved
    frame_delay = 30

# Set up a window and bind the mouse event to it
cv.namedWindow('Select Points')
cv.setMouseCallback('Select Points', select_point)

# Capture the first frame of the video
ret, old_frame = cap.read()
if not ret:
    print("Failed to grab first frame.")
    exit()

old_frame_display = old_frame.copy()
# Show the first frame and let the user select points
while True:
    cv.imshow('Select Points', old_frame)
    if cv.waitKey(1) & 0xFF == 27:  # Press 'Esc' to finish selecting points
        break


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
    qualityLevel=0.01,    # Quality level (between 0 and 1)
    minDistance=1,       # Minimum possible Euclidean distance between returned corners
    blockSize=4,         # Size of the neighborhood considered for corner detection
    useHarrisDetector=False,  # Whether to use Harris corner detector
    k=0.04               # Free parameter of Harris detector (if enabled)
)

def find_nearest_edge_point(point, edge_points):
    edge_points = edge_points.reshape(-1, 2)
    # Calculate the Euclidean distances
    distances = np.linalg.norm(edge_points - point, axis=1)

    if len(distances) == 0:
        raise ValueError("No points to calculate distance from.")
    nearest_index = np.argmin(distances)

    return edge_points[nearest_index]

# Define the contrast-based edge detection function
def is_edge_by_contrast(point, gray_img, threshold):
    x, y = int(point[0]), int(point[1])
    window_size = 3  # Define the size of the window for the neighborhood (e.g., 3x3)
    half_window = window_size // 2

    if (x - half_window >= 0 and x + half_window < gray_img.shape[1] and
        y - half_window >= 0 and y + half_window < gray_img.shape[0]):
        
        # Extract the local neighborhood around the point
        local_window = gray_img[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
        
        # Calculate the contrast between the center pixel and the neighborhood
        center_intensity = gray_img[y, x]
        contrast = np.abs(local_window - center_intensity).mean()
        
        return contrast > threshold
    return False

def is_edge_by_gradient(point, gray_img, threshold):

    sobel_x = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    x, y = point
    grad_value = gradient_magnitude[y, x]

    return grad_value >= threshold

# iterate p0 points，change it to the nearest p0_best
new_p0 = []
for p in p0:
    x, y = p.ravel()
    nearest_edge_point = find_nearest_edge_point([x, y], p0_good)
    new_p0.append(nearest_edge_point)

# convert p0 to numpy and change the shape
p0 = np.array(new_p0, dtype=np.float32).reshape(-1, 1, 2)
points_to_mask.append(p0)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Random colors to display the tracking lines
color = [np.random.randint(0, 255, 3).tolist() for _ in range(1000)]

# Create a mask image to draw the optical flow tracks
mask = np.zeros_like(old_frame)

frame_counter = 111
# Start tracking the points through the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow for the selected points
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points where optical flow is successfully calculated
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    else:
        print("No good points to track.")
        break


    # Threshold for distance to add a new point (you can adjust this value)
    distance_threshold = 15
    merge_distance = 5

    # New list to store updated points, including inserted midpoints
    new_p1 = []

    # Calculate distances between neighbors
    for i in range(len(good_new)):
        point1 = good_new[i]
        point2 = good_new[(i + 1) % len(good_new)]  # connect first and last

        # Calculate the Euclidean distance between two points
        distance = np.linalg.norm(point2 - point1)

        if distance < merge_distance:
            merged_point = (point1 + point2) / 2.0
            if new_p1 and not np.array_equal(new_p1[-1], merged_point):
                new_p1[-1] = merged_point
        else:
            midpoint = (point1 + point2) / 2.0
            if new_p1 and not np.array_equal(new_p1[-1], midpoint):
                new_p1.append(midpoint)

        new_p1.append(point2)


    edge_corners = []
    # Check outliers every n frames
    if frame_counter % 1 == 0:  
        for corner in new_p1:
            if is_edge_by_contrast(corner, frame_gray, 40):  
                edge_corners.append(corner)
    else:
        edge_corners = new_p1

    # Convert the updated list of points into the appropriate shape for tracking (N, 1, 2)
    new_p1 = np.array(edge_corners, dtype=np.float32).reshape(-1, 1, 2)
    # #---------------------mask----------------------------
    # points_int = np.int32(new_p1)

    # height, width = 166, 516
    # mask_saved = np.zeros((height, width), dtype=np.uint8)

    # cv.fillPoly(mask_saved, [points_int], 255)

    # output_folder = 'mask_longus'
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # mask_file_path = os.path.join(output_folder, f'mask_{frame_counter}.png')
    # cv.imwrite(mask_file_path, mask_saved)

    # print(f"Mask{frame_counter} saved to {mask_file_path}")
    # #---------------------end-----------------------

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
    k = cv.waitKey(frame_delay) & 0xff
    if k == 27:
        break

    # Update the previous frame and points for the next iteration
    old_gray = frame_gray.copy()
    p0 = new_p1
    points_to_mask.append(p0)
    frame_counter += 1

# generate_tracking_mask(points_to_mask, (516, 166), "./mask")

cv.destroyAllWindows()

