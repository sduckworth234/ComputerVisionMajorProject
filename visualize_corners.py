#!/usr/bin/env python
"""
Visualize the reference court corners and detected baseline corners
"""
import cv2
import numpy as np

# Load reference court
ref = cv2.imread('court_configurations/court_reference.png', 0)
ref_color = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

# Define the 4 corners of the reference court
# From court_reference.py:
# baseline_top = ((286, 561), (1379, 561))
# baseline_bottom = ((286, 2935), (1379, 2935))
# left_court_line = ((286, 561), (286, 2935))
# right_court_line = ((1379, 561), (1379, 2935))

ref_corners = [
    (286, 561, "TOP-LEFT"),
    (1379, 561, "TOP-RIGHT"),
    (286, 2935, "BOTTOM-LEFT"),
    (1379, 2935, "BOTTOM-RIGHT")
]

# Draw corners on reference with different colors
colors = [
    (255, 0, 0),    # BLUE
    (0, 255, 255),  # YELLOW
    (255, 0, 255),  # MAGENTA
    (0, 255, 0)     # GREEN
]

print("="*70)
print("REFERENCE COURT CORNERS")
print("="*70)
print(f"Reference shape: {ref.shape} (H={ref.shape[0]}, W={ref.shape[1]})")
print()

for i, ((x, y, label), color) in enumerate(zip(ref_corners, colors)):
    # Draw large circle
    cv2.circle(ref_color, (x, y), 30, color, -1)
    # Draw label
    cv2.putText(ref_color, label, (x + 50, y + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 3, color, 8)
    cv2.putText(ref_color, f"({x}, {y})", (x + 50, y + 80),
               cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)

    print(f"{i+1}. {label}: ({x}, {y}) - COLOR: {['BLUE', 'YELLOW', 'MAGENTA', 'GREEN'][i]}")

# Draw the lines between corners
cv2.line(ref_color, (286, 561), (1379, 561), (255, 255, 255), 10)  # Top line
cv2.line(ref_color, (286, 2935), (1379, 2935), (255, 255, 255), 10)  # Bottom line
cv2.line(ref_color, (286, 561), (286, 2935), (255, 255, 255), 10)  # Left line
cv2.line(ref_color, (1379, 561), (1379, 2935), (255, 255, 255), 10)  # Right line

# Add title
cv2.putText(ref_color, "REFERENCE COURT CORNERS", (100, 300),
           cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 12)

cv2.imwrite('reference_corners_annotated.jpg', ref_color)
print("\nSaved: reference_corners_annotated.jpg")
print("="*70)

# Now load the video frame and annotate the detected baseline
cap = cv2.VideoCapture('data/full_video/tennis_full_video.mp4')
ret, frame = cap.read()
cap.release()

frame_annotated = frame.copy()

# The detected baseline (from our earlier detection)
# Bottom baseline at y=572, spans x=334 to x=948
detected_baseline_left = (334, 572)
detected_baseline_right = (948, 572)

# Also show the projected top corners
detected_top_left = (334, 320)
detected_top_right = (948, 320)

frame_corners = [
    (*detected_top_left, "DETECTED TOP-LEFT"),
    (*detected_top_right, "DETECTED TOP-RIGHT"),
    (*detected_baseline_left, "DETECTED BOTTOM-LEFT (BASELINE)"),
    (*detected_baseline_right, "DETECTED BOTTOM-RIGHT (BASELINE)")
]

print("\n" + "="*70)
print("DETECTED FRAME CORNERS")
print("="*70)
print(f"Frame shape: {frame.shape} (H={frame.shape[0]}, W={frame.shape[1]})")
print()

for i, ((x, y, label), color) in enumerate(zip(frame_corners, colors)):
    # Draw large circle
    cv2.circle(frame_annotated, (x, y), 15, color, -1)
    # Draw label
    cv2.putText(frame_annotated, label, (x + 20, y - 10 if i < 2 else y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame_annotated, f"({x}, {y})", (x + 20, y + 10 if i < 2 else y + 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    print(f"{i+1}. {label}: ({x}, {y}) - COLOR: {['BLUE', 'YELLOW', 'MAGENTA', 'GREEN'][i]}")

# Draw the baseline prominently
cv2.line(frame_annotated, detected_baseline_left, detected_baseline_right, (0, 0, 255), 5)
cv2.putText(frame_annotated, "DETECTED BASELINE", (detected_baseline_left[0], detected_baseline_left[1] - 20),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# Draw the trapezoid
cv2.line(frame_annotated, detected_top_left, detected_top_right, (255, 255, 255), 3)
cv2.line(frame_annotated, detected_top_left, detected_baseline_left, (255, 255, 255), 3)
cv2.line(frame_annotated, detected_top_right, detected_baseline_right, (255, 255, 255), 3)

cv2.imwrite('frame_corners_annotated.jpg', frame_annotated)
print("\nSaved: frame_corners_annotated.jpg")
print("="*70)

print("\n" + "="*70)
print("COLOR MATCHING")
print("="*70)
print("Each corner has a color:")
print("  1. BLUE")
print("  2. YELLOW")
print("  3. MAGENTA")
print("  4. GREEN")
print()
print("Look at the two images and tell me:")
print("  Which reference corner (color) should map to which frame corner (color)?")
print()
print("The DETECTED BASELINE (red line in frame) should align with")
print("one of the edges of the reference court (the SHORT edge).")
print("="*70)
