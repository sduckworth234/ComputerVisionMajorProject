# Tennis court corner detection using baseline-anchored approach
# Finds bottom baseline first, then uses court geometry to locate top corners

import cv2
import numpy as np
import argparse


def detect_strict_white_lines(frame):
    # Detect white lines using LAB, HSV, and grayscale thresholds
    # Convert to LAB color space (better for color distances)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Very relaxed white detection: lower L threshold, higher chroma tolerance
    # Allows darker/bluer whites from far baseline
    white_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    white_mask[(l_channel > 150) & (np.abs(a_channel - 128) < 35) & (np.abs(b_channel - 128) < 35)] = 255

    # HSV-based white detection (very lenient)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Lower thresholds: moderate value, higher saturation tolerance
    hsv_white = np.zeros(frame.shape[:2], dtype=np.uint8)
    hsv_white[(v > 150) & (s < 70)] = 255

    # Also use simple grayscale threshold for bright regions (lower)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, gray_white = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    # Combine all three methods
    combined = cv2.bitwise_or(white_mask, hsv_white)
    combined = cv2.bitwise_or(combined, gray_white)

    # Minimal morphology - just clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined


def find_baseline(white_mask, frame_shape):
    # Find bottom baseline by detecting longest horizontal line in bottom third
    height, width = frame_shape[:2]

    # Focus on bottom third of image where baseline should be
    bottom_third_y = int(height * 0.66)
    bottom_region = white_mask[bottom_third_y:, :]

    # Detect edges
    edges = cv2.Canny(bottom_region, 50, 150)

    # Detect lines using Hough Transform with more lenient gap tolerance
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                           minLineLength=50, maxLineGap=50)

    if lines is None or len(lines) == 0:
        print("Warning: No lines detected in bottom region!")
        return None, None

    # Filter for horizontal lines (angle close to 0 or 180 degrees)
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Adjust y coordinates back to full frame
        y1 += bottom_third_y
        y2 += bottom_third_y

        # Calculate angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = abs(angle)

        # Keep nearly horizontal lines (within 10 degrees of horizontal)
        if angle < 10 or angle > 170:
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            horizontal_lines.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'length': length,
                'y_avg': (y1 + y2) / 2
            })

    if len(horizontal_lines) == 0:
        print("Warning: No horizontal lines found!")
        return None, None

    # Merge line segments that are on the same horizontal line
    # Group lines by similar Y coordinate (within 20 pixels)
    y_tolerance = 20
    merged_lines = []

    # Sort by y_avg to group nearby lines
    horizontal_lines.sort(key=lambda l: l['y_avg'])

    used = [False] * len(horizontal_lines)

    for i, line in enumerate(horizontal_lines):
        if used[i]:
            continue

        # Start a new group with this line
        group = [line]
        used[i] = True

        # Find all lines at similar Y coordinate
        for j in range(i+1, len(horizontal_lines)):
            if used[j]:
                continue
            if abs(horizontal_lines[j]['y_avg'] - line['y_avg']) < y_tolerance:
                group.append(horizontal_lines[j])
                used[j] = True

        # Merge this group into one line
        # Take leftmost x1 and rightmost x2
        all_x = [l['x1'] for l in group] + [l['x2'] for l in group]
        all_y = [l['y1'] for l in group] + [l['y2'] for l in group]

        min_x = min(all_x)
        max_x = max(all_x)
        avg_y = int(np.mean(all_y))

        merged_length = max_x - min_x

        merged_lines.append({
            'x1': min_x, 'y1': avg_y,
            'x2': max_x, 'y2': avg_y,
            'length': merged_length,
            'y_avg': avg_y,
            'segment_count': len(group)
        })

    # Find the longest merged line (likely the baseline)
    merged_lines.sort(key=lambda l: l['length'], reverse=True)
    baseline = merged_lines[0]

    print(f"  Baseline found: length={baseline['length']:.1f}px, y={baseline['y_avg']:.1f}, segments_merged={baseline['segment_count']}")

    # Get endpoints (ensure left point is actually on the left)
    x1, y1 = baseline['x1'], baseline['y1']
    x2, y2 = baseline['x2'], baseline['y2']

    if x1 <= x2:
        left_pt = (x1, y1)
        right_pt = (x2, y2)
    else:
        left_pt = (x2, y2)
        right_pt = (x1, y1)

    return left_pt, right_pt


def find_top_baseline(white_mask, bl_corner, br_corner, frame_shape):
    # Find top baseline using perspective constraints (must be inside bottom baseline)
    height, width = frame_shape[:2]

    # Bottom baseline properties
    bottom_mid_x = (bl_corner[0] + br_corner[0]) // 2
    bottom_mid_y = (bl_corner[1] + br_corner[1]) // 2
    bottom_width = abs(br_corner[0] - bl_corner[0])
    bottom_left_x = bl_corner[0]
    bottom_right_x = br_corner[0]

    # Search region: upper half of frame (top baseline should be in upper 20-60% of frame)
    search_y_min = max(0, int(height * 0.10))  # Don't search above top 10%
    search_y_max = int(bottom_mid_y * 0.7)  # Search up to ~30-40% from top

    print(f"  Searching for top baseline between y={search_y_min} and y={search_y_max}")
    print(f"  Bottom baseline: x=[{bottom_left_x}, {bottom_right_x}], y={bottom_mid_y}, width={bottom_width:.1f}px")

    # Focus on top region
    search_region = white_mask[search_y_min:search_y_max, :]

    # Detect edges in white mask
    edges = cv2.Canny(search_region, 50, 150)

    # Detect lines with lenient gap tolerance (same as bottom baseline approach)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20,
                           minLineLength=50, maxLineGap=60)

    if lines is None or len(lines) == 0:
        print("  Warning: No lines detected in top region!")
        return None, None

    # Filter for horizontal lines
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Adjust y coordinates back to full frame
        y1 += search_y_min
        y2 += search_y_min

        # Calculate angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = abs(angle)

        # Keep nearly horizontal lines (within 10 degrees)
        if angle < 10 or angle > 170:
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            horizontal_lines.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'length': length,
                'y_avg': (y1 + y2) / 2
            })

    if len(horizontal_lines) == 0:
        print("  Warning: No horizontal lines found in top region!")
        return None, None

    print(f"  Found {len(horizontal_lines)} horizontal line segments in top region")

    # MERGE LINE SEGMENTS AT SIMILAR Y COORDINATES (same approach as bottom baseline)
    y_tolerance = 20  # pixels
    merged_lines = []

    horizontal_lines.sort(key=lambda l: l['y_avg'])
    used = [False] * len(horizontal_lines)

    for i, line in enumerate(horizontal_lines):
        if used[i]:
            continue

        # Start new group
        group = [line]
        used[i] = True

        # Find all lines at similar Y
        for j in range(i+1, len(horizontal_lines)):
            if used[j]:
                continue
            if abs(horizontal_lines[j]['y_avg'] - line['y_avg']) < y_tolerance:
                group.append(horizontal_lines[j])
                used[j] = True

        # Merge group: take leftmost and rightmost X coordinates
        all_x = [l['x1'] for l in group] + [l['x2'] for l in group]
        all_y = [l['y1'] for l in group] + [l['y2'] for l in group]

        min_x = min(all_x)
        max_x = max(all_x)
        avg_y = int(np.mean(all_y))
        merged_length = max_x - min_x

        merged_lines.append({
            'x1': min_x, 'y1': avg_y,
            'x2': max_x, 'y2': avg_y,
            'length': merged_length,
            'y_avg': avg_y,
            'segment_count': len(group)
        })

    print(f"  Merged into {len(merged_lines)} horizontal lines")

    # APPLY HEURISTICS:
    # 1. Top baseline must be INSIDE bottom baseline horizontally (due to oblique perspective)
    # 2. Top baseline should be narrower than bottom (30-80% width ratio)
    # 3. Should be roughly centered horizontally relative to bottom baseline
    # 4. Should be at reasonable vertical distance (not too close to bottom)

    best_line = None
    best_score = 0

    for line in merged_lines:
        line_left_x = line['x1']
        line_right_x = line['x2']
        line_mid_x = (line_left_x + line_right_x) / 2

        width_ratio = line['length'] / bottom_width
        vertical_dist = bottom_mid_y - line['y_avg']

        # Check if line is INSIDE bottom baseline horizontally
        # Top left should be >= bottom left, top right should be <= bottom right
        is_inside_horizontally = (line_left_x >= bottom_left_x - 50) and (line_right_x <= bottom_right_x + 50)

        # Check if horizontally centered relative to bottom
        horizontal_offset = abs(line_mid_x - bottom_mid_x)
        is_centered = horizontal_offset < bottom_width * 0.15  # Within 15% of bottom width

        # Check width ratio (should be narrower)
        valid_width = 0.25 < width_ratio < 0.85

        # Check vertical distance (should be reasonable)
        valid_distance = vertical_dist > bottom_mid_y * 0.25  # At least 25% of frame height up

        if is_inside_horizontally and is_centered and valid_width and valid_distance:
            # Score: favor longer lines that are well-positioned
            score = line['length'] * 1.0
            # Bonus for being very centered
            if horizontal_offset < bottom_width * 0.05:
                score *= 1.2

            if score > best_score:
                best_score = score
                best_line = line
                print(f"    Candidate: y={line['y_avg']:.1f}, length={line['length']:.1f}, width_ratio={width_ratio:.2f}, centered={is_centered}, inside={is_inside_horizontally}, score={score:.1f}")

    if best_line is None:
        print("  Warning: No valid top baseline found with strict heuristics, relaxing constraints...")
        # Fallback: take longest line that's narrower and roughly centered
        for line in merged_lines:
            width_ratio = line['length'] / bottom_width
            line_mid_x = (line['x1'] + line['x2']) / 2
            horizontal_offset = abs(line_mid_x - bottom_mid_x)

            if width_ratio < 0.9 and horizontal_offset < bottom_width * 0.3:
                score = line['length']
                if score > best_score:
                    best_score = score
                    best_line = line

    if best_line is None:
        # Last resort: just take longest line
        merged_lines.sort(key=lambda l: l['length'], reverse=True)
        best_line = merged_lines[0]
        print(f"  Warning: Using absolute longest line as fallback")

    print(f"  Top baseline selected: y={best_line['y_avg']:.1f}, length={best_line['length']:.1f}px, segments_merged={best_line['segment_count']}")
    print(f"  Width ratio (top/bottom): {best_line['length']/bottom_width:.2f}")
    print(f"  Top baseline X range: [{best_line['x1']}, {best_line['x2']}]")
    print(f"  Bottom baseline X range: [{bottom_left_x}, {bottom_right_x}]")

    # Extract top corners from baseline endpoints
    x1, y1 = best_line['x1'], best_line['y1']
    x2, y2 = best_line['x2'], best_line['y2']

    if x1 <= x2:
        tl_corner = (x1, y1)
        tr_corner = (x2, y2)
    else:
        tl_corner = (x2, y2)
        tr_corner = (x1, y1)

    return tl_corner, tr_corner


def find_top_corners_from_baseline(white_mask, bl_corner, br_corner, frame_shape):
    # Find top corners using court geometry (baseline/sideline ratio ~2.17:1)
    height, width = frame_shape[:2]

    # Calculate baseline width in pixels
    baseline_width_px = np.sqrt((br_corner[0] - bl_corner[0])**2 +
                                (br_corner[1] - bl_corner[1])**2)

    print(f"  Baseline width: {baseline_width_px:.1f}px")

    # Find top baseline using perspective constraints
    tl_corner, tr_corner = find_top_baseline(
        white_mask, bl_corner, br_corner, frame_shape
    )

    return tl_corner, tr_corner


def search_for_corner_along_sideline(white_mask, bottom_corner, expected_length,
                                     baseline_angle, side, frame_shape):
    # Search for top corner along sideline using angle and distance heuristics
    height, width = frame_shape[:2]
    bx, by = bottom_corner

    # Due to perspective, sidelines converge
    # Left sideline goes up-left, right sideline goes up-right
    # But angle depends on camera perspective

    # Search in a range of angles from the bottom corner
    if side == 'left':
        # Left sideline: search from -60° to -120° (upward and slightly left)
        search_angles = np.linspace(-60, -120, 30)  # degrees
    else:
        # Right sideline: search from -120° to -60° (upward and slightly right)
        search_angles = np.linspace(-120, -60, 30)

    # Search at multiple distances (0.8x to 1.5x expected length)
    search_distances = np.linspace(expected_length * 0.8, expected_length * 1.5, 20)

    # Find white line corners in search region
    edges = cv2.Canny(white_mask, 50, 150)

    best_corner = None
    best_score = 0

    for angle_deg in search_angles:
        angle_rad = np.radians(angle_deg)

        for dist in search_distances:
            # Calculate search point
            tx = int(bx + dist * np.cos(angle_rad))
            ty = int(by + dist * np.sin(angle_rad))

            # Check if point is in bounds
            if not (20 < tx < width - 20 and 20 < ty < height - 20):
                continue

            # Check neighborhood for white line presence
            search_radius = 15
            x1 = max(0, tx - search_radius)
            x2 = min(width, tx + search_radius)
            y1 = max(0, ty - search_radius)
            y2 = min(height, ty + search_radius)

            neighborhood = white_mask[y1:y2, x1:x2]
            edge_neighborhood = edges[y1:y2, x1:x2]

            # Score based on white pixels and edge strength
            white_density = np.sum(neighborhood > 0) / (neighborhood.size + 1e-6)
            edge_strength = np.sum(edge_neighborhood > 0) / (edge_neighborhood.size + 1e-6)

            score = white_density * 0.5 + edge_strength * 0.5

            if score > best_score:
                best_score = score
                best_corner = (tx, ty)

    if best_corner is None:
        # Fallback: simple projection
        fallback_angle = -90 if side == 'left' else -90  # straight up
        tx = int(bx + expected_length * np.cos(np.radians(fallback_angle)))
        ty = int(by + expected_length * np.sin(np.radians(fallback_angle)))
        best_corner = (max(0, min(width-1, tx)), max(0, min(height-1, ty)))
        print(f"  Warning: Using fallback for {side} top corner")
    else:
        print(f"  Found {side} top corner: {best_corner}, score={best_score:.3f}")

    return best_corner


def detect_court_corners(frame, visualize=True):
    # Main function to detect court corners using baseline-anchored approach
    # Returns corners in order: top-left, top-right, bottom-right, bottom-left
    height, width = frame.shape[:2]

    print("Step 1: Detecting strict white lines...")
    white_mask = detect_strict_white_lines(frame)

    print("Step 2: Finding baseline (bottom horizontal line)...")
    bl_corner, br_corner = find_baseline(white_mask, frame.shape)

    if bl_corner is None or br_corner is None:
        print("ERROR: Could not find baseline!")
        return []

    print(f"  Bottom-left corner: {bl_corner}")
    print(f"  Bottom-right corner: {br_corner}")

    print("Step 3: Finding top corners using court geometry...")
    tl_corner, tr_corner = find_top_corners_from_baseline(
        white_mask, bl_corner, br_corner, frame.shape
    )

    print(f"  Top-left corner: {tl_corner}")
    print(f"  Top-right corner: {tr_corner}")

    # Return corners in order: TL, TR, BR, BL
    court_corners = [tl_corner, tr_corner, br_corner, bl_corner]

    # Visualization
    if visualize:
        vis_white = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        vis_baseline = frame.copy()
        vis_final = frame.copy()

        # Draw baseline
        cv2.line(vis_baseline, bl_corner, br_corner, (0, 255, 0), 3)
        cv2.circle(vis_baseline, bl_corner, 10, (255, 0, 0), -1)
        cv2.circle(vis_baseline, br_corner, 10, (0, 0, 255), -1)
        cv2.putText(vis_baseline, "BL", (bl_corner[0]+15, bl_corner[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(vis_baseline, "BR", (br_corner[0]+15, br_corner[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw final 4 corners
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]

        for i, (x, y) in enumerate(court_corners):
            cv2.circle(vis_final, (x, y), 15, colors[i], -1)
            cv2.putText(vis_final, corner_labels[i], (x+20, y+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)

        # Draw quadrilateral
        pts = np.array(court_corners, dtype=np.int32)
        cv2.polylines(vis_final, [pts], True, (255, 255, 0), 3)

        # Show results
        cv2.imshow('1. Strict White Lines', vis_white)
        cv2.imshow('2. Baseline Detection', vis_baseline)
        cv2.imshow('3. Final 4 Court Corners', vis_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return court_corners


def main():
    parser = argparse.ArgumentParser(description='Detect tennis court corners')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--frame', type=int, default=0,
                       help='Frame number to analyze (default: 0)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')

    args = parser.parse_args()

    # Load video
    cap = cv2.VideoCapture(args.video)

    # Seek to specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {args.frame}")
        return

    print(f"Processing frame {args.frame} from {args.video}")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

    # Detect corners
    corners = detect_court_corners(frame, visualize=not args.no_viz)

    # Print results
    print("\n" + "="*50)
    print("DETECTED CORNERS:")
    print("="*50)
    if len(corners) == 4:
        labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
        for label, (x, y) in zip(labels, corners):
            print(f"{label:12s}: ({x:4d}, {y:4d})")

        print("\nNumPy array format (for code):")
        print("court_corners = np.array([")
        for x, y in corners:
            print(f"    [{x}, {y}],")
        print("], dtype=np.float32)")
    else:
        print(f"Warning: Found {len(corners)} corners instead of 4")
        for i, (x, y) in enumerate(corners):
            print(f"Corner {i}: ({x}, {y})")

    cap.release()


if __name__ == "__main__":
    main()
