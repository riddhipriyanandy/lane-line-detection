"""
generate_sample_input.py

One-time helper script used to create a small synthetic road image
(input/test_road.jpg) so the project can be run and tested immediately
without needing to source a real dashcam photo.

This is NOT part of the core lane-detection pipeline. Feel free to
delete it and drop in your own real road images/videos instead.
"""

import cv2
import numpy as np

width, height = 960, 540
img = np.zeros((height, width, 3), dtype=np.uint8)

# Sky
img[:int(height * 0.55), :] = (180, 140, 90)   # muted blue-ish sky (BGR)
# Road (asphalt grey)
img[int(height * 0.55):, :] = (60, 60, 60)

# Road shoulders (slightly lighter, gives Canny something to grab near edges)
cv2.line(img, (0, height), (int(width * 0.40), int(height * 0.55)), (110, 110, 110), 3)
cv2.line(img, (width, height), (int(width * 0.62), int(height * 0.55)), (110, 110, 110), 3)

# Left lane marking (solid white line, angled)
cv2.line(img, (150, height), (430, int(height * 0.60)), (230, 230, 230), 8)

# Right lane marking (dashed white line, angled)
p1 = np.array([810, height])
p2 = np.array([560, int(height * 0.60)])
num_dashes = 8
for i in range(num_dashes):
    t1 = i / num_dashes
    t2 = t1 + 0.06
    start = (p1 * (1 - t1) + p2 * t1).astype(int)
    end = (p1 * (1 - t2) + p2 * t2).astype(int)
    cv2.line(img, tuple(start), tuple(end), (230, 230, 230), 8)

cv2.imwrite("input/test_road.jpg", img)
print("Sample image written to input/test_road.jpg")
