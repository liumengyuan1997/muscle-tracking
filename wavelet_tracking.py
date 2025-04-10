# import numpy as np
# import cv2 as cv
# import os
# import pywt
# from utils.generate_mask import generate_tracking_mask

# video_path = './data/06_upper.mp4'
# output_dir = "./mask_wavelet"
# points_to_mask = []

# def is_edge_by_haar(point, gray_img, threshold):
#     point = np.squeeze(point)  # removes unnecessary dimensions
#     x, y = int(point[0] // 2), int(point[1] // 2)

#     coeffs2 = pywt.dwt2(gray_img, 'haar')
#     cA, (cH, cV, cD) = coeffs2

#     if x >= cH.shape[1] or y >= cH.shape[0]:
#         return False

#     local_strength = np.abs(cH[y, x]) + np.abs(cV[y, x]) + np.abs(cD[y, x])
#     return local_strength > threshold

# # ------------------ [1] Haar feature extraction ------------------
# def extract_haar_keypoints(gray_img, threshold=20):
#     coeffs2 = pywt.dwt2(gray_img, 'haar')
#     _, (cH, cV, cD) = coeffs2
#     magnitude = np.abs(cH) + np.abs(cV) + np.abs(cD)
#     keypoints = np.argwhere(magnitude > threshold)
#     keypoints = keypoints * 2  # scale back
#     return keypoints[:, [1, 0]]  # [x, y]

# # ------------------ [2] manually select ROI polygon ------------------
# def select_polygon_roi(image):
#     points = []
#     clone = image.copy()

#     def click_event(event, x, y, flags, param):
#         if event == cv.EVENT_LBUTTONDOWN:
#             points.append((x, y))
#             cv.circle(clone, (x, y), 3, (0, 255, 0), -1)
#             if len(points) > 1:
#                 cv.line(clone, points[-2], points[-1], (255, 0, 0), 1)
#             cv.imshow("Draw ROI", clone)

#     cv.namedWindow("Draw ROI")
#     cv.setMouseCallback("Draw ROI", click_event)

#     while True:
#         cv.imshow("Draw ROI", clone)
#         key = cv.waitKey(1) & 0xFF
#         if key == 13:  # Enter键完成
#             break
#         elif key == 27:
#             points.clear()
#             break

#     cv.destroyWindow("Draw ROI")
#     return np.array(points, dtype=np.int32)

# # ------------------ [3] initialization ------------------
# cap = cv.VideoCapture(video_path)
# ret, old_frame = cap.read()
# if not ret:
#     print("Cannot open video.")
#     exit()

# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# # get ROI and features
# poly_roi = select_polygon_roi(old_frame.copy())

# def get_filtered_p0_from_roi(gray_img, polygon, threshold=40):
#     mask_poly = np.zeros_like(gray_img, dtype=np.uint8)
#     cv.fillPoly(mask_poly, [polygon], 255)
#     masked = cv.bitwise_and(gray_img, gray_img, mask=mask_poly)
#     raw_pts = extract_haar_keypoints(masked, threshold)

#     filtered = []
#     for pt in raw_pts:
#         if cv.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0:
#             filtered.append(pt)
#     return np.array(filtered, dtype=np.float32).reshape(-1, 1, 2)

# p0 = get_filtered_p0_from_roi(old_gray, poly_roi)
# points_to_mask.append(p0)

# # Optical Flow Params
# lk_params = dict(winSize=(10, 10),
#                  maxLevel=2,
#                  criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# mask = np.zeros_like(old_frame)
# color = [np.random.randint(0, 255, 3).tolist() for _ in range(1000)]
# frame_counter = 1

# # ------------------ [4] main loop ------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video.")
#         break

#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # # get ROI per 45 frames
#     # if frame_counter % 45 == 0:
#     #     print(f"\nFrame {frame_counter}: Select new ROI")
#     #     poly_roi = select_polygon_roi(frame.copy())
#     #     new_p0 = get_filtered_p0_from_roi(frame_gray, poly_roi)
#     #     if len(new_p0) >= 3:
#     #         p0 = new_p0
#     #     else:
#     #         print("Too few points in new ROI, keeping old points.")

#     # Optical flow
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

#     if p1 is None or st.sum() == 0:
#         print("No good points to track.")
#         break

#     good_new = p1[st == 1]
#     good_old = p0[st == 1]
#     new_p1 = good_new.reshape(-1, 1, 2)

#     # edge_corners = []
#     # if frame_counter % 10 == 0:
#     #     for corner in new_p1:
#     #         if is_edge_by_haar(corner, frame_gray, threshold=20):
#     #             edge_corners.append(corner)
#     # else:
#     edge_corners = new_p1

#     new_p1 = np.array(edge_corners, dtype=np.float32).reshape(-1, 1, 2)

#     if len(new_p1) > 0:
#         for i, (new, old) in enumerate(zip(new_p1, good_old)):
#             a, b = new.ravel()
#             c, d = old.ravel()
#             frame = cv.circle(frame, (int(a), int(b)), 1, tuple(color[i]), -1)
#     else:
#         print("No keypoints remaining.")

#     img = cv.add(frame, mask)
#     cv.imshow('Optical Flow Tracking', img)

#     # Draw points
#     for i, (new, old) in enumerate(zip(new_p1, good_old)):
#         a, b = new.ravel()
#         frame = cv.circle(frame, (int(a), int(b)), 1, tuple(color[i % len(color)]), -1)

#     img = cv.add(frame, mask)
#     cv.imshow("Tracking", img)

#     k = cv.waitKey(200) & 0xFF
#     if k == 27:
#         break

#     old_gray = frame_gray.copy()
#     p0 = new_p1
#     points_to_mask.append(p0)
#     frame_counter += 1

# # ------------------ [5] generate mask ------------------
# generate_tracking_mask(points_to_mask, (old_frame.shape[0], old_frame.shape[1]), output_dir)

# cap.release()
# cv.destroyAllWindows()

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json
import os
import pywt
from utils.generate_mask import generate_tracking_mask

video_path = './data/06_upper.mp4'
points_to_mask = []

# Step 1: Initialize video and read first frame
cap = cv.VideoCapture(video_path)
ret, old_frame = cap.read()
if not ret:
    print("Failed to grab first frame.")
    exit()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Step 2: Select ROI manually
bbox = cv.selectROI("Select ROI", old_frame, False, False)
cv.destroyWindow("Select ROI")
x, y, w, h = map(int, bbox)
roi = old_gray[y:y+h, x:x+w]

# Step 3: Extract Haar wavelet keypoints
def extract_haar_keypoints(gray_roi, threshold=20):
    coeffs2 = pywt.dwt2(gray_roi, 'haar')
    _, (cH, cV, cD) = coeffs2
    magnitude = np.abs(cH) + np.abs(cV) + np.abs(cD)
    keypoints = np.argwhere(magnitude > threshold)
    keypoints = keypoints * 2  # scale back to original size
    return keypoints[:, [1, 0]]  # convert to [x, y]

haar_kp = extract_haar_keypoints(roi, threshold=40)
haar_kp[:, 0] += x
haar_kp[:, 1] += y
p0 = np.array(haar_kp, dtype=np.float32).reshape(-1, 1, 2)
points_to_mask.append(p0)

# Step 4: Setup for tracking
lk_params = dict(winSize=(10, 10),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = [np.random.randint(0, 255, 3).tolist() for _ in range(1000)]
mask = np.zeros_like(old_frame)

frame_counter = 1

# Step 5: Start tracking loop
while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # if frame_counter % 40 == 0:
    #     print(f"\nFrame {frame_counter}: Re-select ROI")

    #     # re-select ROI
    #     temp_display = frame.copy()
    #     bbox = cv.selectROI("Re-Select ROI", temp_display, False, False)
    #     cv.destroyWindow("Re-Select ROI")

    #     if bbox[2] == 0 or bbox[3] == 0:
    #         print("Empty ROI selected, skipping update.")
    #     else:
    #         x, y, w, h = map(int, bbox)
    #         roi = frame_gray[y:y+h, x:x+w]

    #         haar_kp = extract_haar_keypoints(roi, threshold=40)

    #         if len(haar_kp) < 3:
    #             print("Too few keypoints detected in new ROI, skipping update.")
    #         else:
    #             haar_kp[:, 0] += x
    #             haar_kp[:, 1] += y
    #             p0 = np.array(haar_kp, dtype=np.float32).reshape(-1, 1, 2)

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    else:
        print("No good points to track.")
        break

    edge_corners = good_new  # optional: could filter by contrast or gradient

    new_p1 = np.array(edge_corners, dtype=np.float32).reshape(-1, 1, 2)

    if len(new_p1) > 0:
        for i, (new, old) in enumerate(zip(new_p1, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv.circle(frame, (int(a), int(b)), 1, tuple(color[i]), -1)
    else:
        print("No keypoints remaining.")

    img = cv.add(frame, mask)
    cv.imshow('Optical Flow Tracking', img)

    k = cv.waitKey(200) & 0xff
    if k == 27:  # ESC
        break

    old_gray = frame_gray.copy()
    p0 = new_p1
    points_to_mask.append(p0)
    frame_counter += 1

# Step 6: Generate mask from tracked points
generate_tracking_mask(points_to_mask, (old_frame.shape[0], old_frame.shape[1]), "./mask_wavelet")

cap.release()
cv.destroyAllWindows()

