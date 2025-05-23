import cv2
import pywt
import matplotlib.pyplot as plt

# Read the image and convert it to grayscale
img = cv2.imread('./data/axial_268.png', cv2.IMREAD_GRAYSCALE)

# Ensure the image size is a power of 2 (resize if needed)
# img = cv2.resize(img, (256, 256))

# Perform 2D wavelet transform (using Haar wavelet)
coeffs2 = pywt.dwt2(img, 'haar')
LL, (LH, HL, HH) = coeffs2

# Visualize the original image and the four subbands
titles = ['Original', 'LL (Approximation)', 'LH (Horizontal)', 'HL (Vertical)', 'HH (Diagonal)']
images = [img, LL, LH, HL, HH]

plt.figure(figsize=(10, 8))
for i, (title, im) in enumerate(zip(titles, images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(im, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
