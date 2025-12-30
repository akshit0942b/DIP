import math
import cv2 
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt


# def translate_image(img, tx, ty):
#     """
#     img : input RGB image (H x W x C)
#     tx  : translation in x direction (columns)
#     ty  : translation in y direction (rows)
#     """
#     h, w, c = img.shape
#     output = np.zeros_like(img)

#     # Translation matrix (homogeneous)
#     T = np.array([
#         [1, 0, tx],
#         [0, 1, ty],
#         [0, 0, 1]
#     ])

#     # Inverse for backward mapping
#     T_inv = np.linalg.inv(T)

#     for y in range(h):
#         for x in range(w):
#             src = T_inv @ np.array([x, y, 1])
#             xs, ys = int(src[0]), int(src[1])

#             if 0 <= xs < w and 0 <= ys < h:
#                 output[y, x] = img[ys, xs]

#     return output



# def scale_image(img, sx, sy):
#     """
#     img : input RGB image (H x W x C)
#     sx  : scaling factor in x direction
#     sy  : scaling factor in y direction
#     """
#     h, w, c = img.shape
#     output = np.zeros_like(img)

#     S = np.array([
#         [sx, 0,  0],
#         [0,  sy, 0],
#         [0,  0,  1]
#     ])

#     S_inv = np.linalg.inv(S)

#     for y in range(h):
#         for x in range(w):
#             src = S_inv @ np.array([x, y, 1])
#             xs, ys = int(src[0]), int(src[1])

#             if 0 <= xs < w and 0 <= ys < h:
#                 output[y, x] = img[ys, xs]

#     return output



# def rotate_image(img, angle_deg):
#     """
#     img       : input RGB image (H x W x C)
#     angle_deg : rotation angle in degrees (counter-clockwise)
#     """
#     h, w, c = img.shape
#     output = np.zeros_like(img)

#     theta = np.deg2rad(angle_deg)
#     cos, sin = np.cos(theta), np.sin(theta)

#     cx, cy = w // 2, h // 2

#     T1 = np.array([
#         [1, 0, -cx],
#         [0, 1, -cy],
#         [0, 0,  1]
#     ])

#     R = np.array([
#         [cos, -sin, 0],
#         [sin,  cos, 0],
#         [0,     0,  1]
#     ])

#     T2 = np.array([
#         [1, 0, cx],
#         [0, 1, cy],
#         [0, 0, 1]
#     ])

#     M = T2 @ R @ T1
#     M_inv = np.linalg.inv(M)

#     for y in range(h):
#         for x in range(w):
#             src = M_inv @ np.array([x, y, 1])
#             xs, ys = int(src[0]), int(src[1])

#             if 0 <= xs < w and 0 <= ys < h:
#                 output[y, x] = img[ys, xs]

#     return output





# img = cv2.imread("20943629.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# h, w, c = img.shape;          # 3438 5157

# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.axis("off")


# translate = translate_image(img, 50, 50)
# plt.subplot(1, 4, 2)
# plt.imshow(translate)
# plt.axis("off")

# scaled = scale_image(img, 2, 2)
# plt.subplot(1, 2, 2)
# plt.imshow(scaled)
# plt.axis("off")


# rotated = rotate_image(img, 45)
# plt.subplot(1, 2, 2)
# plt.imshow(rotated)
# plt.axis("off")



#  replacing every pixel by the average of their 3x3 neighborhood
# float_image = img.astype(np.float32)
# new_image = float_image.copy();
# for i in range(1, 3436):
#     for j in range(1, 5156):
#         avg = (img[i-1, j-1] + img[i-1, j] + img[i-1, j+1] + img[i, j-1] + img[i, j] + img[i, j+1] + img[i+1, j-1] + img[i+1, j] + img[i+1, j+1]) / 9.0;
#         new_image[i, j] = avg
        
# new_image = np.clip(new_image, 0, 255).astype(np.uint8)

# plt.subplot(1, 2, 2)
# plt.imshow(new_image)
# plt.axis("off")


# plt.show()
        

        
        







































































































































