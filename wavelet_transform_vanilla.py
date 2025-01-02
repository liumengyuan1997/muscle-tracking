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
titles = ['I (ROI)', 'A (Approximation)', 'H (Horizontal)', 'V (Vertical)', 'D (Diagonal)']
images = [img, LL, LH, HL, HH]

plt.figure(figsize=(10, 8))
for i, (title, im) in enumerate(zip(titles, images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(im, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

if len(images) < 6:
    for j in range(len(images), 6):
        ax = plt.subplot(2, 3, j + 1)
        ax.axis('off')

plt.show()
