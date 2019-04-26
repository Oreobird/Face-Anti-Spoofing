import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

PROJECT_DIR = "E:/ml/Face-Anti-Spoofing/"
NUAA_DATA_DIR = PROJECT_DIR + 'data/NUAA/'

REAL_FACE_SAMPLE = NUAA_DATA_DIR + 'ClientFace/0001/0001_00_00_01_0.jpg'
SPOOF_FACE_SAMPLE = NUAA_DATA_DIR + 'ImposterFace/0001/0001_00_00_01_0.jpg'


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()

real_img_bgr = cv2.imread(REAL_FACE_SAMPLE, cv2.IMREAD_COLOR)
real_img_hsv = cv2.cvtColor(real_img_bgr, cv2.COLOR_BGR2HSV)

spoof_img_bgr = cv2.imread(SPOOF_FACE_SAMPLE, cv2.IMREAD_COLOR)
spoof_img_hsv = cv2.cvtColor(spoof_img_bgr, cv2.COLOR_BGR2HSV)

real_img_ycrcb = cv2.cvtColor(real_img_bgr, cv2.COLOR_BGR2YCrCb)
spoof_img_ycrcb = cv2.cvtColor(spoof_img_bgr, cv2.COLOR_BGR2YCrCb)

real_img_rgb = cv2.cvtColor(real_img_bgr, cv2.COLOR_BGR2RGB)
spoof_img_rgb = cv2.cvtColor(spoof_img_bgr, cv2.COLOR_BGR2RGB)

images = []
titles = ['rgb', 'hsv', 'h', 's', 'v', 'ycrcb', 'y', 'cr', 'cb', 'rgb', 'hsv', 'h', 's', 'v', 'ycrcb', 'y', 'cr', 'cb']

# Real face
images.append(real_img_rgb)
images.append(real_img_hsv)
for i in range(3):
    images.append(real_img_hsv[..., i])

images.append(real_img_ycrcb)
for i in range(3):
    images.append(real_img_ycrcb[..., i])

# Fake face
images.append(spoof_img_rgb)
images.append(spoof_img_hsv)
for i in range(3):
    images.append(spoof_img_hsv[..., i])

images.append(spoof_img_ycrcb)
for i in range(3):
    images.append(spoof_img_ycrcb[..., i])
    
# Display
display_images(images, titles, cols=9, cmap='gray')