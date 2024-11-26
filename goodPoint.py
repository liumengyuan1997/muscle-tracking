import cv2 as cv
import numpy as np

# Capture video or image
cap = cv.VideoCapture('test.mp4')  # Change to your video path

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Capture the first frame
ret, old_frame = cap.read()
if not ret:
    print("Failed to grab the first frame")
    exit()

# Convert the first frame to grayscale
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# old_gray = cv.imread('test_image.png', cv.IMREAD_GRAYSCALE)

# Detect good features to track using the Shi-Tomasi method
p0 = cv.goodFeaturesToTrack(
    old_gray,            # Input grayscale image
    maxCorners=10000,      # Maximum number of corners to return
    qualityLevel=0.01,    # Quality level (between 0 and 1)
    minDistance=1,       # Minimum possible Euclidean distance between returned corners
    blockSize=4,         # Size of the neighborhood considered for corner detection
    useHarrisDetector=False,  # Whether to use Harris corner detector
    k=0.04               # Free parameter of Harris detector (if enabled)
)

# Draw the detected good feature points (p0) on the image
for i in range(p0.shape[0]):
    x, y = p0[i, 0]
    cv.circle(old_gray, (int(x), int(y)), 1, (0, 255, 0), -1)  # Draw green filled circle

# Display the image with good feature points
cv.imshow('Good Features', old_gray)

# Wait until a key is pressed
cv.waitKey(0)

# Close all windows
cv.destroyAllWindows()
