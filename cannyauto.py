import numpy as np
import cv2 as cv

# Load the video
video_path = './data/test_lower_longus.mp4'
cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
frame_delay = int(1000 / fps) if fps > 0 else 30

# Define bounding box variables
bounding_box = None
drawing = False
ix, iy = -1, -1
ex, ey = -1, -1

# Mouse callback function for selecting the bounding box
def select_bbox(event, x, y, flags, param):
    global ix, iy, bounding_box, drawing, old_frame
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        print("start points: ", x, y)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = old_frame.copy()
            cv.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 1)
            cv.imshow("Select Bounding Box", frame_copy)
    elif event == cv.EVENT_LBUTTONUP:
        ex, ey = x, y
        drawing = False
        print("end points: ", x, y)
        print("select dimension: ", min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        bounding_box = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        cv.rectangle(old_frame, (bounding_box[0], bounding_box[1]), 
                     (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 1)
        cv.imshow("Select Bounding Box", old_frame)

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Failed to read the video.")
    exit()

# Show the first frame and let the user draw the bounding box
cv.namedWindow("Select Bounding Box")
cv.setMouseCallback("Select Bounding Box", select_bbox)
cv.imshow("Select Bounding Box", old_frame)
cv.waitKey(0)
cv.destroyWindow("Select Bounding Box")

if bounding_box is None:
    print("Bounding box not selected.")
    exit()

# Convert the first frame to grayscale
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Detect edges using Canny within the bounding box
def detect_canny_points_in_bbox(gray_image, bbox, lower_thresh=100, upper_thresh=150):
    # x, y, w, h = bbox
    # roi = gray_image[y:y + h, x:x + w]  # Crop to bounding box
    print("Width:", gray_image.shape[1])
    print("Height:", gray_image.shape[0])
    edges = cv.Canny(gray_image, lower_thresh, upper_thresh)

    # Get coordinates of edge points within the bounding box
    y_coords, x_coords = np.where(edges > 0)
    points = np.array(list(zip(x_coords, y_coords)), dtype=np.float32).reshape(-1, 1, 2)
    return points

# Function to filter points within the bounding box
def filter_points_in_bbox(points, bbox):
    x, y, w, h = bbox
    print("bbox dimension: ", x, y, w, h)
    filtered_points = []
    for pt in points:
        px, py = pt.ravel()
        if float(x)+2 < px < float(x + w)-2 and float(y)+2 < py < float(y + h)-2:
            print(f"Point ({px}, {py}) is inside the bounding box.")
            filtered_points.append(pt)
    return np.array(filtered_points, dtype=np.float32).reshape(-1, 1, 2)

# Detect points in the bounding box
p0 = detect_canny_points_in_bbox(old_gray, bounding_box)
p0 = filter_points_in_bbox(p0, bounding_box)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize tracking mask
mask = np.zeros_like(old_frame)



# Main processing loop
frame_counter = 1
while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames to process.")
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect edges and update points every 10 frames
    if frame_counter % 30 == 0:
        cv.namedWindow("Select Bounding Box")
        cv.setMouseCallback("Select Bounding Box", select_bbox)
        cv.imshow("Select Bounding Box", frame_gray)
        cv.waitKey(0)
        cv.destroyWindow("Select Bounding Box")
        new_points = detect_canny_points_in_bbox(frame_gray, bounding_box)
        new_points = filter_points_in_bbox(new_points, bounding_box)
        if len(new_points) > 0:
            p0 = new_points

    # Apply Lucas-Kanade optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    else:
        print("No points to track.")
        break

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv.circle(frame, (int(c), int(d)), 1, (0, 255, 0), -1)

    # Draw the bounding box
    x, y, w, h = bounding_box
    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    img = cv.add(frame, mask)
    cv.imshow('Optical Flow with Muscle Edges', img)

    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    frame_counter += 1

    if cv.waitKey(frame_delay) & 0xFF == 27:  # Exit on 'Esc'
        break

cap.release()
cv.destroyAllWindows()
