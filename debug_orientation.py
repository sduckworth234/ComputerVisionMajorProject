#!/usr/bin/env python
"""
Debug: Check if reference court lines are oriented correctly after transformation
"""
import cv2
import numpy as np

# Load reference and frame
ref = cv2.imread('court_configurations/court_reference.png', 0)
cap = cv2.VideoCapture('data/full_video/tennis_full_video.mp4')
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]

# Reference corners
ref_corners = np.float32([
    [286, 561],      # Top-left
    [1379, 561],     # Top-right
    [286, 2935],     # Bottom-left
    [1379, 2935]     # Bottom-right
])

# Frame corners (from detection)
frame_corners = np.float32([
    [369, 321],      # Top-left
    [913, 321],      # Top-right
    [253, 573],      # Bottom-left (baseline left)
    [1030, 573]      # Bottom-right (baseline right)
])

# Compute homography
matrix, _ = cv2.findHomography(ref_corners, frame_corners)

# Transform specific reference lines to see orientation
print("="*70)
print("REFERENCE LINE TRANSFORMATIONS")
print("="*70)

# 1. Bottom baseline (HORIZONTAL in reference)
ref_bottom_baseline = np.float32([[[286, 2935]], [[1379, 2935]]])
transformed = cv2.perspectiveTransform(ref_bottom_baseline, matrix)
pt1 = transformed[0][0]
pt2 = transformed[1][0]
print(f"\n1. BOTTOM BASELINE (y=2935 in ref, HORIZONTAL):")
print(f"   Reference: (286, 2935) to (1379, 2935)")
print(f"   Transformed: ({pt1[0]:.1f}, {pt1[1]:.1f}) to ({pt2[0]:.1f}, {pt2[1]:.1f})")
if abs(pt1[1] - pt2[1]) < 10:
    print(f"   ✓ HORIZONTAL in frame (y≈{pt1[1]:.1f})")
else:
    print(f"   ✗ NOT HORIZONTAL! (y from {pt1[1]:.1f} to {pt2[1]:.1f})")

# 2. Top baseline (HORIZONTAL in reference)
ref_top_baseline = np.float32([[[286, 561]], [[1379, 561]]])
transformed = cv2.perspectiveTransform(ref_top_baseline, matrix)
pt1 = transformed[0][0]
pt2 = transformed[1][0]
print(f"\n2. TOP BASELINE (y=561 in ref, HORIZONTAL):")
print(f"   Reference: (286, 561) to (1379, 561)")
print(f"   Transformed: ({pt1[0]:.1f}, {pt1[1]:.1f}) to ({pt2[0]:.1f}, {pt2[1]:.1f})")
if abs(pt1[1] - pt2[1]) < 10:
    print(f"   ✓ HORIZONTAL in frame (y≈{pt1[1]:.1f})")
else:
    print(f"   ✗ NOT HORIZONTAL! (y from {pt1[1]:.1f} to {pt2[1]:.1f})")

# 3. Left sideline (VERTICAL in reference)
ref_left_sideline = np.float32([[[286, 561]], [[286, 2935]]])
transformed = cv2.perspectiveTransform(ref_left_sideline, matrix)
pt1 = transformed[0][0]
pt2 = transformed[1][0]
print(f"\n3. LEFT SIDELINE (x=286 in ref, VERTICAL):")
print(f"   Reference: (286, 561) to (286, 2935)")
print(f"   Transformed: ({pt1[0]:.1f}, {pt1[1]:.1f}) to ({pt2[0]:.1f}, {pt2[1]:.1f})")
if abs(pt1[0] - pt2[0]) < 50:
    print(f"   ✓ VERTICAL in frame (x≈{pt1[0]:.1f})")
else:
    print(f"   ✗ NOT VERTICAL! (x from {pt1[0]:.1f} to {pt2[0]:.1f})")

# 4. Right sideline (VERTICAL in reference)
ref_right_sideline = np.float32([[[1379, 561]], [[1379, 2935]]])
transformed = cv2.perspectiveTransform(ref_right_sideline, matrix)
pt1 = transformed[0][0]
pt2 = transformed[1][0]
print(f"\n4. RIGHT SIDELINE (x=1379 in ref, VERTICAL):")
print(f"   Reference: (1379, 561) to (1379, 2935)")
print(f"   Transformed: ({pt1[0]:.1f}, {pt1[1]:.1f}) to ({pt2[0]:.1f}, {pt2[1]:.1f})")
if abs(pt1[0] - pt2[0]) < 50:
    print(f"   ✓ VERTICAL in frame (x≈{pt1[0]:.1f})")
else:
    print(f"   ✗ NOT VERTICAL! (x from {pt1[0]:.1f} to {pt2[0]:.1f})")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("If baselines are HORIZONTAL and sidelines are VERTICAL, orientation is CORRECT")
print("If baselines are VERTICAL and sidelines are HORIZONTAL, there's a 90° rotation!")
print("="*70)
