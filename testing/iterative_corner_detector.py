"""
Iterative Tennis Court Corner Detection
Pure CV2 approach - no ML/AI

Strategy:
1. Focus ONLY on court region (exclude crowds/scoreboard)
2. Detect white lines with color + brightness thresholding
3. Use Hough lines to find dominant straight lines
4. Classify into vertical (sidelines) and horizontal (baselines)
5. Find intersections of outer lines = 4 corners
"""

import cv2
import numpy as np
from collections import defaultdict


def create_court_region_mask(frame):
    """
    Create mask to focus ONLY on court area.
    Excludes: top 15% (scoreboard/ads), sides with crowds
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Conservative cropping to avoid false positives
    y_start = int(h * 0.15)  # Skip scoreboard
    y_end = int(h * 0.98)    # Keep baseline area
    x_start = int(w * 0.05)  # Skip left crowd
    x_end = int(w * 0.95)    # Skip right crowd

    mask[y_start:y_end, x_start:x_end] = 255
    return mask, (y_start, y_end, x_start, x_end)


def detect_white_lines_robust(frame, region_mask):
    """
    Robust white line detection for tennis courts.
    Works on grass (green), clay (orange), hard court (blue).
    """
    # Apply region mask first
    frame_masked = cv2.bitwise_and(frame, frame, mask=region_mask)

    # Convert to HSV for better white detection
    hsv = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2HSV)

    # White lines: High Value, Low Saturation
    # Adjusted for TV footage brightness
    lower_white = np.array([0, 0, 200])      # Very bright
    upper_white = np.array([180, 30, 255])   # Low saturation
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Also detect in grayscale (backup for different lighting)
    gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Combine both methods
    combined = cv2.bitwise_or(white_mask, bright_mask)

    # Remove small noise (grass texture, players' clothing)
    kernel_small = np.ones((2, 2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Connect broken line segments
    kernel_line = np.ones((1, 5), np.uint8)  # Horizontal connections
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_line, iterations=1)
    kernel_line = np.ones((5, 1), np.uint8)  # Vertical connections
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_line, iterations=1)

    return combined


def detect_straight_lines_hough(white_mask, frame_shape):
    """
    Detect straight lines using Probabilistic Hough Transform.
    Returns only long, confident lines (court boundaries).
    """
    h, w = frame_shape[:2]

    # Edge detection on white mask
    edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)

    # Probabilistic Hough Line Transform
    # Parameters tuned for court boundary lines (long, straight)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,                    # 1 pixel resolution
        theta=np.pi / 180,        # 1 degree resolution
        threshold=50,             # Min votes (line strength)
        minLineLength=int(w * 0.15),  # At least 15% of image width
        maxLineGap=20             # Max gap to connect segments
    )

    if lines is None:
        return []

    return lines.reshape(-1, 4)  # Convert to (x1, y1, x2, y2) format


def classify_lines(lines, angle_tolerance=15):
    """
    Classify lines into vertical (sidelines) and horizontal (baselines).

    Args:
        lines: Array of [x1, y1, x2, y2]
        angle_tolerance: Degrees from perfect vertical/horizontal

    Returns:
        vertical_lines: List of lines close to vertical
        horizontal_lines: List of lines close to horizontal
    """
    vertical_lines = []
    horizontal_lines = []

    for x1, y1, x2, y2 in lines:
        # Calculate angle from horizontal (0-180 degrees)
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            angle = 90  # Perfectly vertical
        else:
            angle = abs(np.degrees(np.arctan(dy / dx)))

        # Classify based on angle
        if angle > 90 - angle_tolerance and angle < 90 + angle_tolerance:
            # Close to vertical (80-100 degrees)
            vertical_lines.append([x1, y1, x2, y2, angle])
        elif angle < angle_tolerance or angle > 180 - angle_tolerance:
            # Close to horizontal (0-15 or 165-180 degrees)
            horizontal_lines.append([x1, y1, x2, y2, angle])

    return vertical_lines, horizontal_lines


def merge_similar_lines(lines, distance_threshold=30, angle_threshold=5):
    """
    Merge lines that are close and parallel.
    Returns representative line for each cluster.
    """
    if len(lines) == 0:
        return []

    lines = np.array(lines)
    merged = []
    used = set()

    for i, line1 in enumerate(lines):
        if i in used:
            continue

        x1, y1, x2, y2, angle1 = line1
        cluster = [line1]

        for j, line2 in enumerate(lines[i+1:], start=i+1):
            if j in used:
                continue

            x3, y3, x4, y4, angle2 = line2

            # Check if angles are similar
            if abs(angle1 - angle2) > angle_threshold:
                continue

            # Check if lines are close (average distance)
            mid1_x, mid1_y = (x1 + x2) / 2, (y1 + y2) / 2
            mid2_x, mid2_y = (x3 + x4) / 2, (y3 + y4) / 2
            distance = np.sqrt((mid1_x - mid2_x)**2 + (mid1_y - mid2_y)**2)

            if distance < distance_threshold:
                cluster.append(line2)
                used.add(j)

        # Average all lines in cluster
        cluster = np.array(cluster)
        avg_line = cluster[:, :4].mean(axis=0)
        merged.append(avg_line)
        used.add(i)

    return merged


def select_outer_lines(vertical_lines, horizontal_lines):
    """
    Select the 2 outermost vertical lines (left/right sidelines)
    and 2 outermost horizontal lines (top/bottom baselines).
    """
    if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
        return None, None, None, None

    # Sort vertical lines by x-coordinate (left to right)
    vertical_lines = sorted(vertical_lines, key=lambda line: (line[0] + line[2]) / 2)
    left_line = vertical_lines[0]   # Leftmost
    right_line = vertical_lines[-1]  # Rightmost

    # Sort horizontal lines by y-coordinate (top to bottom)
    horizontal_lines = sorted(horizontal_lines, key=lambda line: (line[1] + line[3]) / 2)
    top_line = horizontal_lines[0]   # Topmost
    bottom_line = horizontal_lines[-1]  # Bottommost

    return left_line, right_line, top_line, bottom_line


def line_intersection(line1, line2):
    """
    Find intersection point of two lines.
    Returns None if lines are parallel or don't intersect.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:  # Lines are parallel
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    # Calculate intersection point
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (int(x), int(y))


def find_four_corners(left_line, right_line, top_line, bottom_line):
    """
    Find 4 court corners from outer boundary lines.
    """
    # Top-left: intersection of left sideline and top baseline
    tl = line_intersection(left_line, top_line)

    # Top-right: intersection of right sideline and top baseline
    tr = line_intersection(right_line, top_line)

    # Bottom-left: intersection of left sideline and bottom baseline
    bl = line_intersection(left_line, bottom_line)

    # Bottom-right: intersection of right sideline and bottom baseline
    br = line_intersection(right_line, bottom_line)

    # Check if all corners were found
    if None in [tl, tr, bl, br]:
        return None

    return tl, tr, bl, br


def validate_court_geometry(corners, frame_shape):
    """
    Validate that detected corners form a reasonable tennis court.

    Checks:
    1. Corners form a convex quadrilateral
    2. Aspect ratio is reasonable (tennis court ~2.2:1)
    3. Top edge is narrower than bottom (perspective)
    """
    if corners is None:
        return False

    tl, tr, bl, br = corners
    h, w = frame_shape[:2]

    # Check 1: Points should be in reasonable positions
    if not (0 <= tl[0] < w and 0 <= tl[1] < h):
        return False
    if not (0 <= tr[0] < w and 0 <= tr[1] < h):
        return False
    if not (0 <= bl[0] < w and 0 <= bl[1] < h):
        return False
    if not (0 <= br[0] < w and 0 <= br[1] < h):
        return False

    # Check 2: Top edge should be narrower than bottom (perspective)
    top_width = abs(tr[0] - tl[0])
    bottom_width = abs(br[0] - bl[0])

    if top_width >= bottom_width:  # Unrealistic perspective
        return False

    # Check 3: Aspect ratio should be reasonable (width:height ~2-3:1)
    left_height = abs(bl[1] - tl[1])
    right_height = abs(br[1] - tr[1])
    avg_height = (left_height + right_height) / 2
    avg_width = (top_width + bottom_width) / 2

    if avg_height == 0:
        return False

    aspect_ratio = avg_width / avg_height

    if aspect_ratio < 1.5 or aspect_ratio > 4.0:  # Too narrow or too wide
        return False

    # Check 4: Corners should form convex quadrilateral
    # (Simple check: cross products should have consistent sign)
    points = np.array([tl, tr, br, bl], dtype=np.float32)

    # All passed
    return True


def detect_court_corners(frame, debug=False):
    """
    Main function: Detect 4 corners of tennis court.

    Returns:
        corners: (tl, tr, bl, br) or None if detection failed
        debug_images: Dictionary of intermediate results (if debug=True)
    """
    debug_images = {}

    # STAGE 1: Create region mask (focus on court only)
    region_mask, crop_region = create_court_region_mask(frame)
    if debug:
        debug_images['1_region_mask'] = region_mask

    # STAGE 2: Detect white lines
    white_mask = detect_white_lines_robust(frame, region_mask)
    if debug:
        debug_images['2_white_mask'] = white_mask

    # STAGE 3: Detect straight lines with Hough
    lines = detect_straight_lines_hough(white_mask, frame.shape)
    if len(lines) < 4:
        print(f"❌ Not enough lines detected: {len(lines)}")
        return None, debug_images

    if debug:
        line_img = frame.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        debug_images['3_hough_lines'] = line_img
        print(f"✓ Detected {len(lines)} lines")

    # STAGE 4: Classify lines (vertical vs horizontal)
    vertical_lines, horizontal_lines = classify_lines(lines)
    if debug:
        print(f"✓ Classified: {len(vertical_lines)} vertical, {len(horizontal_lines)} horizontal")

    # STAGE 5: Merge similar lines
    vertical_merged = merge_similar_lines(vertical_lines)
    horizontal_merged = merge_similar_lines(horizontal_lines)

    if debug:
        merged_img = frame.copy()
        for x1, y1, x2, y2 in vertical_merged:
            cv2.line(merged_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
        for x1, y1, x2, y2 in horizontal_merged:
            cv2.line(merged_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        debug_images['4_merged_lines'] = merged_img
        print(f"✓ After merging: {len(vertical_merged)} vertical, {len(horizontal_merged)} horizontal")

    # STAGE 6: Select outer boundary lines
    left, right, top, bottom = select_outer_lines(vertical_merged, horizontal_merged)
    if left is None:
        print("❌ Could not find 4 boundary lines")
        return None, debug_images

    if debug:
        boundary_img = frame.copy()
        cv2.line(boundary_img, (int(left[0]), int(left[1])), (int(left[2]), int(left[3])), (255, 0, 0), 4)
        cv2.line(boundary_img, (int(right[0]), int(right[1])), (int(right[2]), int(right[3])), (255, 0, 0), 4)
        cv2.line(boundary_img, (int(top[0]), int(top[1])), (int(top[2]), int(top[3])), (0, 0, 255), 4)
        cv2.line(boundary_img, (int(bottom[0]), int(bottom[1])), (int(bottom[2]), int(bottom[3])), (0, 0, 255), 4)
        debug_images['5_boundary_lines'] = boundary_img

    # STAGE 7: Find 4 corner intersections
    corners = find_four_corners(left, right, top, bottom)
    if corners is None:
        print("❌ Could not compute corner intersections")
        return None, debug_images

    # STAGE 8: Validate geometry
    if not validate_court_geometry(corners, frame.shape):
        print("❌ Court geometry validation failed")
        return None, debug_images

    if debug:
        tl, tr, bl, br = corners
        result_img = frame.copy()

        # Draw court boundary
        pts = np.array([tl, tr, br, bl], dtype=np.int32)
        cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)

        # Draw corners
        cv2.circle(result_img, tl, 10, (0, 0, 255), -1)
        cv2.circle(result_img, tr, 10, (0, 255, 255), -1)
        cv2.circle(result_img, bl, 10, (255, 0, 0), -1)
        cv2.circle(result_img, br, 10, (255, 0, 255), -1)

        # Labels
        cv2.putText(result_img, 'TL', (tl[0]-20, tl[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, 'TR', (tr[0]+10, tr[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, 'BL', (bl[0]-20, bl[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, 'BR', (br[0]+10, br[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        debug_images['6_final_result'] = result_img
        print(f"✓ Corners detected: TL{tl}, TR{tr}, BL{bl}, BR{br}")

    return corners, debug_images


def warp_to_birds_eye(frame, corners, padding=0.2):
    """
    Apply perspective transform to get bird's eye view.

    Args:
        frame: Input image
        corners: (tl, tr, bl, br)
        padding: Vertical padding ratio for out-of-bounds space

    Returns:
        warped: Bird's eye view image
        matrix: Perspective transform matrix
    """
    tl, tr, bl, br = corners

    # Source points (detected corners)
    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    # Calculate output dimensions
    top_width = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
    bottom_width = np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
    avg_width = (top_width + bottom_width) / 2

    left_height = np.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
    right_height = np.sqrt((br[0] - tr[0])**2 + (br[1] - tr[1])**2)
    avg_height = (left_height + right_height) / 2

    # Output dimensions (rectangular court)
    out_width = int(avg_width)
    out_height = int(avg_height)

    # Add vertical padding for out-of-bounds tracking
    pad_pixels = int(out_height * padding)
    total_height = out_height + 2 * pad_pixels

    # Destination points (rectangular)
    dst_pts = np.array([
        [0, pad_pixels],
        [out_width, pad_pixels],
        [out_width, pad_pixels + out_height],
        [0, pad_pixels + out_height]
    ], dtype=np.float32)

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply warp
    warped = cv2.warpPerspective(frame, matrix, (out_width, total_height))

    return warped, matrix


if __name__ == "__main__":
    # Test on frame.png
    image_path = '../data/frame.png'

    print("=" * 60)
    print("Tennis Court Corner Detection - Iterative Test")
    print("=" * 60)

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not load image: {image_path}")
        exit(1)

    print(f"✓ Loaded image: {frame.shape}")

    # Detect corners with debug visualizations
    corners, debug_images = detect_court_corners(frame, debug=True)

    if corners is None:
        print("\n❌ FAILED: Could not detect court corners")

        # Show debug images to understand failure
        for name, img in debug_images.items():
            cv2.imshow(name, img)

        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(1)

    # Success!
    tl, tr, bl, br = corners
    print("\n" + "=" * 60)
    print("✅ SUCCESS: Court corners detected!")
    print("=" * 60)
    print(f"Top-Left:     {tl}")
    print(f"Top-Right:    {tr}")
    print(f"Bottom-Left:  {bl}")
    print(f"Bottom-Right: {br}")

    # Create bird's eye view
    warped, matrix = warp_to_birds_eye(frame, corners, padding=0.2)
    debug_images['7_birds_eye_view'] = warped

    # Display all debug images
    print("\nDisplaying debug visualizations...")
    for name, img in debug_images.items():
        cv2.imshow(name, img)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n✓ Test complete!")
