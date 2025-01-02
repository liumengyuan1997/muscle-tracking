import numpy as np
import cv2 as cv
import os
import scipy.interpolate

video_path = './data/test_upper_longus.mp4'

# Mouse click event callback function to select bounding box
bbox = []
def select_bbox(event, x, y, flags, param):
    global bbox
    if event == cv.EVENT_LBUTTONDOWN:
        bbox = [(x, y)]  # Start point
    elif event == cv.EVENT_LBUTTONUP:
        bbox.append((x, y))  # End point
        print(f"Bounding box selected: {bbox}")

# Open video
cap = cv.VideoCapture(video_path)
ret, first_frame = cap.read()
if not ret:
    print("Failed to load video.")
    exit()

cv.namedWindow('Select Bounding Box')
cv.setMouseCallback('Select Bounding Box', select_bbox)

# Get the video's frames per second (fps)
fps = cap.get(cv.CAP_PROP_FPS)

# Calculate the frame delay in milliseconds (1000 ms = 1 second)
if fps > 0:
    frame_delay = int(1000 / fps)
else:
    frame_delay = 30  # Default to 30ms if fps not retrieved

while len(bbox) < 2:
    cv.imshow('Select Bounding Box', first_frame)
    if cv.waitKey(frame_delay) & 0xFF == 27:
        break
cv.destroyAllWindows()

if len(bbox) < 2:
    print("Bounding box not selected. Exiting.")
    exit()

# Initialize variables
frame_counter = 1
old_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

def update_points_with_bbox(frame, bbox):
    """Update points to track using a new bounding box."""
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    roi = frame[y1:y2, x1:x2]

    # Convert to grayscale
    roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # # Detect edges and corners within the bounding box
    # corners = cv.goodFeaturesToTrack(
    #     roi_gray, maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=3
    # )

    # fast
    fast = cv.FastFeatureDetector_create()
    keypoints = fast.detect(roi_gray, None)
    corners = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    # # SIFT Feature Detection
    # sift = cv.SIFT_create()
    # keypoints = sift.detect(roi_gray, None)
    # corners = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    # # Harris Corner Detection
    # harris_corners = cv.cornerHarris(roi_gray, blockSize=3, ksize=3, k=0.04)
    # threshold = 0.001 * harris_corners.max()
    # keypoints = np.argwhere(harris_corners > threshold)  # Get points above threshold
    # corners = [kp[::-1] for kp in keypoints]  # Convert to (x, y) format
    # corners = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
    if corners is not None:
        corners[:, 0, 0] += x1
        corners[:, 0, 1] += y1
        return np.float32(corners).reshape(-1, 1, 2)
    else:
        print("No corners detected in the new bounding box.")
        return None

# Initialize points to track
p0 = update_points_with_bbox(first_frame, bbox)
if p0 is None:
    print("p0 is none")
    exit()

# Optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (500, 3))
mask = np.zeros_like(first_frame)

mask_shape = (first_frame.shape[0], first_frame.shape[1])
tracking_mask = np.zeros(mask_shape, dtype=np.uint8)

output_dir = './tracking_masks'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Reset tracking mask
    tracking_mask = np.zeros(mask_shape, dtype=np.uint8)

    # Check if we need to reset bounding box
    if frame_counter % 20 == 0:
        print("Select new bounding box...")
        bbox = []
        cv.namedWindow('Select New Bounding Box')
        cv.setMouseCallback('Select New Bounding Box', select_bbox)

        while len(bbox) < 2:
            cv.imshow('Select New Bounding Box', frame)
            if cv.waitKey(frame_delay) & 0xFF == 27:
                break
        cv.destroyAllWindows()

        if len(bbox) < 2:
            print("Bounding box not selected. Exiting.")
            break

        # Update points with new bounding box
        p0 = update_points_with_bbox(frame, bbox)
        if p0 is None:
            break

        old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame)  # Reset mask

    # Preprocess the current frame
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        # print(good_new)
    else:
        print("No points to track.")
        break

    # Extract and draw outer contour
    if len(good_new) > 2:
        # Compute the convex hull
        hull = cv.convexHull(good_new.reshape(-1, 2).astype(np.int32))

        # # Optionally smooth the contour with spline interpolation
        # hull_points = hull.reshape(-1, 2)
        # x = hull_points[:, 0]
        # y = hull_points[:, 1]
        
        # if len(hull_points) > 3:  # Spline interpolation if enough points exist
        #     tck, u = scipy.interpolate.splprep([x, y], s=5.0, per=1)
        #     unew = np.linspace(0, 1, 1000)
        #     out = scipy.interpolate.splev(unew, tck)
        #     smoothed_points = np.array([out[0], out[1]]).T.astype(np.int32)
        # else:
        #     smoothed_points = hull_points  # If not enough points, use original hull

        # Draw the smoothed outer contour
        cv.polylines(tracking_mask, [hull], isClosed=True, color=255, thickness=1)

        # # Save the tracking mask to the output directory
        # mask_filename = os.path.join(output_dir, f'mask_frame_{frame_counter:04d}.png')
        # cv.imwrite(mask_filename, tracking_mask)

    # Update visualization
    # Convert tracking_mask to 3 channels
    tracking_mask_color = cv.cvtColor(tracking_mask, cv.COLOR_GRAY2BGR)

    # Add the mask to the frame
    img = cv.add(frame, tracking_mask_color)
    cv.imshow('Tracking', img)

    # Update variables
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    frame_counter += 1

    # Add delay and check for exit
    if cv.waitKey(frame_delay) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()