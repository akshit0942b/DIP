import os
import math
import cv2 
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import random

img = cv2.imread("dog.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, c = img.shape

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
gaussianBlurr = cv2.GaussianBlur(img, (301, 301), 0)

R = img[:, :, 0].astype(np.float32)
G = img[:, :, 1].astype(np.float32)
B = img[:, :, 2].astype(np.float32)

# Playing with Intensity
Y  = np.clip(0.299*R + 0.587*G + 0.114*B, 0, 255).astype(np.uint8)


# some random noise generation
for _ in range(100):
    c = random.random()
    noisy = np.where(Y == int(c*255), 255, Y)


noise = noisy - Y

fil = cv2.medianBlur(noisy, ksize=51)

plt.subplot(3, 2, 1)
plt.imshow(Y, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.imshow(noisy, cmap='gray')
plt.title("Noisy")
plt.axis("off")

plt.subplot(3, 2, 3)
plt.imshow(noise, cmap='gray')
plt.title("Added noise")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.imshow(fil, cmap='gray')
plt.title("median filter")
plt.axis("off")


plt.show()









































































































