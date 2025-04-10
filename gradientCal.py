import numpy as np
import cv2 as cv

# Define a function to compute gradients over a 20x20 region
def calculate_gradient_20x20_region(gray_img, x, y):
    # Compute gradients using the Sobel operator
    grad_x = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=3)  # Gradient in the x-direction
    grad_y = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=3)  # Gradient in the y-direction
    
    # Extract gradients for the 20x20 region
    gx = grad_x[y-10:y+10, x-10:x+10]
    gy = grad_y[y-10:y+10, x-10:x+10]
    
    # Compute gradient magnitude for the region (then sum or average)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    # Return the average gradient magnitude of the region
    return np.mean(gradient_magnitude)

# Path to the video
video_path = './data/06_lower.mp4'

cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video.")
    exit()

frame_counter = 0

# Loop through each frame
while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Create an empty image to store the average gradient of each 20x20 region
    gradient_img = np.zeros_like(gray, dtype=np.float32)

    # Iterate over the center point of each 20x20 region, using a stride to reduce computation
    for y in range(10, gray.shape[0] - 10, 20):  # Step through every 20x20 region by center point
        for x in range(10, gray.shape[1] - 10, 20):
            # Calculate gradient for this 20x20 region
            gradient_magnitude = calculate_gradient_20x20_region(gray, x, y)
            gradient_img[y, x] = gradient_magnitude  # Store the average gradient value for the region
            
            # Display the value on the image
            text = f"{gradient_magnitude:.1f}"  # Format value to 1 decimal place
            cv.putText(frame, text, (x - 15, y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv.LINE_AA)

            # Draw a rectangle for each 20x20 region
            top_left = (x - 10, y - 10)
            bottom_right = (x + 10, y + 10)
            cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)  # Draw green rectangle

    # Normalize the gradient image to 0–255 range
    cv.normalize(gradient_img, gradient_img, 0, 255, cv.NORM_MINMAX)
    gradient_img = np.uint8(gradient_img)  # Convert to 8-bit image type

    # Display the image with values and grid
    cv.imshow('Gradient Magnitude (20x20 regions) with Grid', frame)

    # Check for exit key
    k = cv.waitKey(500) & 0xFF
    if k == 27:  # Exit on Esc key
        break

    frame_counter += 1

cap.release()
cv.destroyAllWindows()
