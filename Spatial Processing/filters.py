import os
import math
import cv2 
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import random

img = cv2.imread("images/flower.avif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, c = img.shape

R = img[:, :, 0].astype(np.float32)
G = img[:, :, 1].astype(np.float32)
B = img[:, :, 2].astype(np.float32)

# Playing with Intensity
Y  = np.clip(0.299*R + 0.587*G + 0.114*B, 0, 255).astype(np.uint8)


# Vectorized averaging using slicing (no loops!)
blurr = np.zeros_like(img, dtype=np.float32)

# Add all 9 neighboring pixels using array slicing -> invisible blurr
blurr[1:h-1, 1:w-1] = (
    img[0:h-2, 0:w-2].astype(np.float32) +  # top-left
    img[0:h-2, 1:w-1].astype(np.float32) +  # top-center
    img[0:h-2, 2:w].astype(np.float32) +    # top-right
    img[1:h-1, 0:w-2].astype(np.float32) +  # middle-left
    img[1:h-1, 1:w-1].astype(np.float32) +  # center
    img[1:h-1, 2:w].astype(np.float32) +    # middle-right
    img[2:h, 0:w-2].astype(np.float32) +    # bottom-left
    img[2:h, 1:w-1].astype(np.float32) +    # bottom-center
    img[2:h, 2:w].astype(np.float32)        # bottom-right
) / 9.0


blurr = blurr.astype(np.uint8)
 
blurred = cv2.blur(img, (100,100)) # averaging a 100x100 size block -> visible blurr

gaussianBlurr = cv2.GaussianBlur(Y, (91, 91), 0)


# Gaussian Noise Generation
noise = np.zeros(Y.shape, dtype=np.float32)
cv2.randn(noise, mean=0, stddev=55)

noisy = Y.astype(np.float32) + noise
noisy = np.clip(noisy, 0 ,255).astype(np.uint8)




# MEDIAN FILTER
fil = cv2.medianBlur(noisy, ksize=11)

diff = Y - fil

# SOBEL FILTER
kernel_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)


kernel_y = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]], dtype=np.float32)


g_x = cv2.filter2D(noisy, cv2.CV_64F, kernel_x)
g_y = cv2.filter2D(noisy, cv2.CV_64F, kernel_y)

magnitude = np.sqrt(g_x**2 + g_y**2)
edges = np.uint8(255 * magnitude / magnitude.max())


# LAPLACIAN FILTER
lap = cv2.Laplacian(noisy, ddepth=cv2.CV_64F)
lap = np.abs(lap)
lap = np.uint8(lap)


# Unsharp Masking
extracted_details = Y - gaussianBlurr
sharped = Y + extracted_details


# Bilateral Filter
bilateral = cv2.bilateralFilter(img, 131, 60, 90)

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

# plt.subplot(3, 2, 2)
# plt.imshow(sharped, cmap='gray')
# plt.title("sharped")
# plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gaussianBlurr)
plt.title("gaussian blurr")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(bilateral)
plt.title("Bilateral Filter")
plt.axis("off")

# plt.subplot(3, 2, 3)
# plt.imshow(edges, cmap='gray')
# plt.title("sobel filter (edges)")
# plt.axis("off")

# plt.subplot(3, 2, 4)
# plt.imshow(lap, cmap='gray')
# plt.title("laplacian (edges)")
# plt.axis("off")


plt.show()









































































































