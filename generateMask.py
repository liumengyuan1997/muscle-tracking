import json
import numpy as np
import cv2

# Load the JSON data
with open('axial_268.json', 'r') as file:
    data = json.load(file)

# Extract the image dimensions and polygon points
image_shape = (166, 516)
mask = np.zeros(image_shape, dtype=np.uint8)

for shape in data['shapes']:
    points = np.array(shape['points'], dtype=np.int32)
    cv2.fillPoly(mask, [points], color=255)

# Save the mask
cv2.imwrite('mask.png', mask)

# To visualize
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()