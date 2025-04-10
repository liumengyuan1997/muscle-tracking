import numpy as np
import cv2 as cv
import json

video_path = './data/06_upper.mp4'
json_path = '/Users/liumengyuan/Downloads/muscle_data/06lower_mask/axial_233.json'

with open(json_path, "r") as f:
    mask_data = json.load(f)
    points_to_track = mask_data["shapes"][0]["points"]

cap = cv.VideoCapture(video_path)
ret, old_frame = cap.read()
if not ret:
    print("Failed to grab first frame.")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

p0_manual = np.array(points_to_track, dtype=np.float32).reshape(-1, 1, 2)
p0_detected = cv.goodFeaturesToTrack(
    old_gray, maxCorners=10000, qualityLevel=0.02,
    minDistance=1, blockSize=4, useHarrisDetector=False, k=0.04
)

def find_nearest_edge_point(point, edge_points):
    edge_points = edge_points.reshape(-1, 2)
    distances = np.linalg.norm(edge_points - point, axis=1)
    if len(distances) == 0:
        raise ValueError("No points to calculate distance from.")
    nearest_index = np.argmin(distances)
    return edge_points[nearest_index]

new_p0 = p0_manual

# new_p0 = []
# for p in p0_manual:
#     x, y = p.ravel()
#     nearest_edge_point = find_nearest_edge_point([x, y], p0_detected)
#     new_p0.append(nearest_edge_point)

p0 = np.array(new_p0, dtype=np.float32).reshape(-1, 1, 2)

lk_params = dict(winSize=(10, 10), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = [np.random.randint(0, 255, 3).tolist() for _ in range(1000)]
# color = [(0, 255, 0)] * 1000

print("length of p0", len(p0))

kalman_filters = []
for pt in p0:
    kalman = cv.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1

    x, y = pt.ravel()
    kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
    kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

    kalman_filters.append(kalman)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No more frames.')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # use LK Optical Flow to calculate the motion
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    predicted_points = []

    if p1 is not None:
        for i, kf in enumerate(kalman_filters):
            if st[i] == 1:
                measurement = np.array([[p1[i][0][0]], [p1[i][0][1]]], dtype=np.float32)
                kf.correct(measurement)

            pred = kf.predict()
            predicted_points.append(pred[:2].reshape(1, 2))
    else:
        for kf in kalman_filters:
            pred = kf.predict()
            predicted_points.append(pred[:2].reshape(1, 2))

    predicted_points = np.array(predicted_points, dtype=np.float32).reshape(-1, 1, 2)

    for i, pt in enumerate(predicted_points):
        a, b = pt[0]
        if 0 <= a < frame.shape[1] and 0 <= b < frame.shape[0]:
            frame = cv.circle(frame, (int(a), int(b)), 1, tuple(color[i % len(color)]), -1)
        else:
            print(f"[{i}] Predicted: ({a:.1f}, {b:.1f})")

    cv.imshow('Kalman + Optical Flow Tracking', frame)

    k = cv.waitKey(500) & 0xff
    if k == 27:
        break

    old_gray = frame_gray.copy()
    p0 = predicted_points.copy()

cap.release()
cv.destroyAllWindows()
