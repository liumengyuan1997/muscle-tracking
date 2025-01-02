import numpy as np
import cv2 as cv
from utils.generate_mask import generate_tracking_mask

video_path = './data/06_upper.mp4'

# Create a list to store the points clicked by the user
points_to_track = []

# Create a list to store the masks' points clicked by the user
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
    qualityLevel=0.02,    # Quality level (between 0 and 1)
    minDistance=1,       # Minimum possible Euclidean distance between returned corners
    blockSize=4,         # Size of the neighborhood considered for corner detection
    useHarrisDetector=False,  # Whether to use Harris corner detector
    k=0.04               # Free parameter of Harris detector (if enabled)
)

new_p0 = np.array(p0, dtype=np.float32).reshape(-1, 1, 2)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Random colors to display the tracking lines
color = [np.random.randint(0, 255, 3).tolist() for _ in range(1000)]

# Create a mask image to draw the optical flow tracks
mask = np.zeros_like(old_frame)

frame_counter = 233
# Start tracking the points through the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow for the selected points
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, new_p0, None, **lk_params)

    # Select good points where optical flow is successfully calculated
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = new_p0[st == 1]
    else:
        print("No good points to track.")
        break

    # Threshold for distance to add a new point (you can adjust this value)
    distance_threshold = 15
    merge_distance = 5

    # Convert the updated list of points into the appropriate shape for tracking (N, 1, 2)
    new_p1 = np.array(good_new, dtype=np.float32).reshape(-1, 1, 2)

    if len(new_p1) > 0:
        for i, (new, old) in enumerate(zip(new_p1, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv.circle(frame, (int(a), int(b)), 1, tuple(color[i]), -1)
    else:
        print("No good points to track.")

    # Overlay the mask with the tracks onto the current frame
    img = cv.add(frame, mask)
    cv.imshow('Tracking', img)

    # Exit if 'Esc' key is pressed
    k = cv.waitKey(500) & 0xff
    if k == 27:
        break

    # Update the previous frame and points for the next iteration
    old_gray = frame_gray.copy()
    new_p0 = new_p1
    frame_counter += 1

generate_tracking_mask(points_to_mask, (old_frame.shape[0], old_frame.shape[1]), "./mask")

cv.destroyAllWindows()

