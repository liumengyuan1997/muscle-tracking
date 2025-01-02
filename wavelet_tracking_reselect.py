import numpy as np
import cv2 as cv
import pywt
from utils.generate_mask import generate_tracking_mask

video_path = './data/06_lower.mp4'
points_to_mask = []

# Step 1: Initialize video and read first frame
cap = cv.VideoCapture(video_path)
ret, old_frame = cap.read()
if not ret:
    print("Failed to grab first frame.")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Step 2: Select ROI manually
bbox = cv.selectROI("Select ROI", old_frame, True, False)
cv.destroyWindow("Select ROI")
x, y, w, h = bbox
roi = old_gray[y:y+h, x:x+w]

# Step 3: Extract Haar wavelet keypoints
def extract_haar_keypoints(gray_roi, threshold=10):
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

    if frame_counter % 40 == 0:
        print(f"\nFrame {frame_counter}: Re-select ROI")

        # re-select ROI
        temp_display = frame.copy()
        bbox = cv.selectROI("Re-Select ROI", temp_display, True, False)
        cv.destroyWindow("Re-Select ROI")

        if bbox[2] == 0 or bbox[3] == 0:
            print("Empty ROI selected, skipping update.")
        else:
            x, y, w, h = bbox
            roi = frame_gray[y:y+h, x:x+w]

            haar_kp = extract_haar_keypoints(roi, threshold=40)

            if len(haar_kp) < 3:
                print("Too few keypoints detected in new ROI, skipping update.")
            else:
                haar_kp[:, 0] += x
                haar_kp[:, 1] += y
                new_p1 = np.array(haar_kp, dtype=np.float32).reshape(-1, 1, 2)
    else:
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
generate_tracking_mask(points_to_mask, (old_frame.shape[0], old_frame.shape[1]), "./mask_wavelet_reselect")

cap.release()
cv.destroyAllWindows()

