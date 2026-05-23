import os
import math
import cv2 
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import random

img = cv2.imread("images/lines.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, c = img.shape

R = img[:, :, 0].astype(np.float32)
G = img[:, :, 1].astype(np.float32)
B = img[:, :, 2].astype(np.float32)

# Playing with Intensity
Y  = np.clip(0.299*R + 0.587*G + 0.114*B, 0, 255).astype(np.uint8)


B1 = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0]
], dtype=np.uint8)

B2 = np.array([
    [1,0,1],
    [0,0,0],
    [1,0,1]
], dtype=np.uint8)

erode1 = cv2.erode(img, B1)

img_comp = 255 - img

erode2 = cv2.erode(img_comp, B2)

hitmiss = erode1 & erode2

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(erode1, cmap='grey')
plt.title("Erode 1")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(erode2, cmap='grey')
plt.title("Erode 2")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(hitmiss, cmap='grey')
plt.title("Hitmiss")
plt.axis("off")

plt.show()





