import os
import math
import cv2 
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import random

img = cv2.imread("images/cat.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, c = img.shape

R = img[:, :, 0].astype(np.float32)
G = img[:, :, 1].astype(np.float32)
B = img[:, :, 2].astype(np.float32)

# Playing with Intensity
Y  = np.clip(0.299*R + 0.587*G + 0.114*B, 0, 255).astype(np.uint8)


# Dilation
dilated = cv2.dilate(img, np.ones((7, 7), np.uint8), iterations=1)

# Brightening
brightened = np.clip(img * 1.5, 0, 255).astype(np.uint8)

# Erosion
eroded = cv2.erode(img, np.ones((7, 7), np.uint8), iterations=1)


plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(dilated)
plt.title("Dilated")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(brightened)
plt.title("Brightened")
plt.axis("off")


plt.show()








































































































