import numpy as np
import cv2 as cv
import os
from skimage.segmentation import active_contour
from skimage.filters import gaussian

# Load the video
video_path = './data/06_upper.mp4'
output_directory ='./06outputMasksupper'
cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
frame_delay = int(1000 / fps) if fps > 0 else 30

# Define bounding box variables
bounding_box = None
drawing = False
ix, iy = -1, -1

def refine_with_acm(gray_image, edges, bbox):
    """
    Refine the edge detection results using Active Contour Model (ACM).
    """
    x, y, w, h = bbox
    region_of_interest = gray_image[y:y + h, x:x + w]
    
    # Smooth the image (optional, but recommended for ACM)
    smoothed_image = gaussian(region_of_interest, sigma=1.0)
    
    # Generate initial snake points (e.g., rectangle around the bounding box)
    s = np.linspace(0, 2 * np.pi, 400)
    init_x = x + w // 2 + (w // 2) * np.cos(s)
    init_y = y + h // 2 + (h // 2) * np.sin(s)
    init_snake = np.array([init_x, init_y]).T

    # Apply Active Contour Model
    snake = active_contour(smoothed_image, init_snake, alpha=0.1, beta=1.0, gamma=0.01)

    # Map the refined contour back to the original image coordinates
    snake[:, 0] += x
    snake[:, 1] += y

    return snake

# Mouse callback function for selecting the bounding box
def select_bbox(event, x, y, frame):
    global ix, iy, bounding_box, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        print("start points: ", x, y)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 1)
            cv.imshow("Select Bounding Box", frame_copy)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        print("end points: ", x, y)
        print("select dimension: ", min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        bounding_box = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        cv.rectangle(frame, (bounding_box[0], bounding_box[1]), 
                     (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 1)
        cv.imshow("Select Bounding Box", frame)

def filter_inner_circle(points, threshold_ratio=0.5):
    """
    保留较内侧的点，形成一个更紧凑的圆。

    参数:
    points (numpy.ndarray): 边缘点的坐标数组，形状为 (N, 2)。
    threshold_ratio (float): 用于筛选的阈值比例，越小保留的点越靠近中心。

    返回:
    numpy.ndarray: 筛选后的点。
    """
    # Ensure points is in shape (N, 2)
    points = points.reshape(-1, 2)

    # 计算所有点的几何中心（质心）
    center = np.mean(points, axis=0)

    # 计算每个点到中心的距离
    distances = np.linalg.norm(points - center, axis=1)

    # 计算距离的中位数（或者其他统计指标）
    median_distance = np.median(distances)

    # 筛选出距离小于某个比例阈值的点
    threshold_distance = median_distance * threshold_ratio
    inner_points = points[distances <= threshold_distance]

    return inner_points

# Example function to save the refined tracking mask
def save_tracking_mask(refined_mask, output_dir, frame_index):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the output file path (e.g., mask_001.png, mask_002.png)
    output_path = os.path.join(output_dir, f"mask_{frame_index:03d}.png")
    
    # Save the refined mask as an image
    cv.imwrite(output_path, refined_mask)
    print(f"Mask saved to: {output_path}")
def is_edge_by_gradient(point, gray_img, threshold):
    # Define the vertical kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    # Apply the kernel
    gradient = cv.filter2D(gray_img, cv.CV_64F, kernel)

    abs_gradient = np.abs(gradient)
    
    # Get integer indices
    x, y = int(point[0]), int(point[1])  # Ensure indices are integers
    
    # Check bounds to avoid index errors
    if x < 0 or y < 0 or y >= abs_gradient.shape[0] or x >= abs_gradient.shape[1]:
        raise ValueError(f"Point {point} is out of bounds for the given image.")
    
    # Get the combined gradient value at the specific point
    grad_value = abs_gradient[y, x]
    
    # Check if the gradient value is above the threshold
    return grad_value >= threshold
# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Failed to read the video.")
    exit()

# Show the first frame and let the user draw the bounding box
cv.namedWindow("Select Bounding Box")
cv.setMouseCallback("Select Bounding Box", lambda event, x, y, flags, param: select_bbox(event, x, y, old_frame))
cv.imshow("Select Bounding Box", old_frame)
cv.waitKey(0)
cv.destroyWindow("Select Bounding Box")

if bounding_box is None:
    print("Bounding box not selected.")
    exit()

# Convert the first frame to grayscale
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Detect edges using Canny within the bounding box
def detect_canny_points_in_bbox(gray_image, bbox, lower_thresh=120, upper_thresh=170):
    print("Width:", gray_image.shape[1])
    print("Height:", gray_image.shape[0])
    edges = cv.Canny(gray_image, lower_thresh, upper_thresh)

    # Get coordinates of edge points within the bounding box
    y_coords, x_coords = np.where(edges > 0)
    points = np.array(list(zip(x_coords, y_coords)), dtype=np.float32).reshape(-1, 1, 2)

    inner_points = filter_inner_circle(points, threshold_ratio=1.0)
    return inner_points

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
        cv.setMouseCallback("Select Bounding Box", lambda event, x, y, flags, param: select_bbox(event, x, y, frame_gray))
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

    edge_corners = []
    # Check outliers every n frames
    if frame_counter % 1 == 0:  
        for corner in good_new:       
            if is_edge_by_gradient(corner, frame_gray, 10):  
                edge_corners.append(corner)
        edge_corners = np.array(edge_corners)
    else:
        edge_corners = good_new

    mask_shape = (frame_gray.shape[0], frame_gray.shape[1])
    tracking_mask = np.zeros(mask_shape, dtype=np.uint8)
    if len(edge_corners) > 2:
        hull = cv.convexHull(edge_corners.reshape(-1, 2).astype(np.int32))
        cv.polylines(tracking_mask, [hull], isClosed=True, color=255, thickness=1)

    tracking_mask_color = cv.cvtColor(tracking_mask, cv.COLOR_GRAY2BGR)
    # save_tracking_mask(tracking_mask, output_directory, frame_counter)

    # Draw the tracks
    for i, (new, old) in enumerate(zip(edge_corners, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv.circle(frame, (int(c), int(d)), 1, (0, 255, 0), -1)

    # Draw the bounding box
    x, y, w, h = bounding_box
    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    img = cv.add(frame, tracking_mask_color)
    cv.imshow('Optical Flow with Muscle Edges', img)

    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = edge_corners.reshape(-1, 1, 2)

    frame_counter += 1

    if cv.waitKey(frame_delay) & 0xFF == 27:  # Exit on 'Esc'
        break

cap.release()
cv.destroyAllWindows()
