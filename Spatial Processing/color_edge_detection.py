import os
import math
import cv2 
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt

img = cv2.imread("baymax.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


R = img[:, :, 0].astype(np.float32)
G = img[:, :, 1].astype(np.float32)
B = img[:, :, 2].astype(np.float32)

sobelx_R = cv2.Sobel(R, cv2.CV_32F, 1, 0, ksize=3)
sobelx_G = cv2.Sobel(G, cv2.CV_32F, 1, 0, ksize=3)
sobelx_B = cv2.Sobel(B, cv2.CV_32F, 1, 0, ksize=3)

sobely_R = cv2.Sobel(R, cv2.CV_32F, 0, 1, ksize=3)
sobely_G = cv2.Sobel(G, cv2.CV_32F, 0, 1, ksize=3)
sobely_B = cv2.Sobel(B, cv2.CV_32F, 0, 1, ksize=3)

Gx = np.sqrt(sobelx_R**2 + sobelx_G**2 + sobelx_B**2)
Gy = np.sqrt(sobely_R**2 + sobely_G**2 + sobely_B**2)

edge_mag = np.sqrt(Gx**2 + Gy**2)

edge_mag = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX)
edge_mag = edge_mag.astype(np.uint8)

plt.subplot(2, 1, 1)
plt.imshow(img)
plt.axis("off")

plt.subplot(2, 1, 2)
plt.imshow(edge_mag)
plt.axis("off")

plt.show()



