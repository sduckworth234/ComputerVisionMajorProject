import cv2
import numpy as np
from collections import defaultdict

def load_and_preprocess(image_path):
    """Load image"""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image from {image_path}")
    return frame

def create_adaptive_region_mask(frame):
    """Create region mask that adapts to image content"""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # More conservative cropping
    y_start = int(h * 0.2)
    y_end = int(h * 0.95)
    x_start = int(w * 0.02)
    x_end = int(w * 0.98)
    
    mask[y_start:y_end, x_start:x_end] = 255
    return mask

def multi_method_line_detection(frame, region_mask):
    """Combine multiple edge/line detection methods"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=region_mask)
    
    # Method 1: Adaptive thresholding for white lines
    adaptive = cv2.adaptiveThreshold(gray_masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, -2)
    
    # Method 2: White color detection (for grass/hard courts)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_mask = cv2.bitwise_and(white_mask, white_mask, mask=region_mask)
    
    # Method 3: Contrast-based edge detection
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_masked)
    _, contrast_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 4: Morphological gradient (detects boundaries)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray_masked, cv2.MORPH_GRADIENT, kernel)
    _, gradient_mask = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
    
    # Combine all methods
    combined = cv2.bitwise_or(adaptive, white_mask)
    combined = cv2.bitwise_or(combined, contrast_mask)
    combined = cv2.bitwise_or(combined, gradient_mask)
    
    # Clean up noise
    kernel_open = np.ones((2, 2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Connect line segments
    kernel_close_h = np.ones((1, 7), np.uint8)
    kernel_close_v = np.ones((7, 1), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close_h, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close_v, iterations=1)
    
    return combined, {
        'adaptive': adaptive,
        'white': white_mask,
        'contrast': contrast_mask,
        'gradient': gradient_mask
    }

def detect_lines_multi_scale(edge_mask, frame_shape):
    """Detect lines at multiple scales and combine"""
    all_lines = []
    
    # Parameters for different scales
    configs = [
        {'threshold': 50, 'minLength': 40, 'maxGap': 20},   # Detect longer lines
        {'threshold': 40, 'minLength': 30, 'maxGap': 15},   # Medium lines
        {'threshold': 30, 'minLength': 25, 'maxGap': 25},   # Shorter but well-connected
    ]
    
    for config in configs:
        lines = cv2.HoughLinesP(
            edge_mask,
            rho=1,
            theta=np.pi/180,
            threshold=config['threshold'],
            minLineLength=config['minLength'],
            maxLineGap=config['maxGap']
        )
        
        if lines is not None:
            all_lines.extend(lines[:, 0, :].tolist())
    
    print(f"Total lines detected: {len(all_lines)}")
    
    if len(all_lines) == 0:
        return [], []
    
    # Classify lines
    vertical_lines = []
    horizontal_lines = []
    
    for x1, y1, x2, y2 in all_lines:
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0:
            angle = 90
        else:
            angle = abs(np.degrees(np.arctan2(dy, dx)))
        
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 20:  # Filter very short segments
            continue
        
        # More generous angle thresholds
        if 65 < angle < 115:
            vertical_lines.append((x1, y1, x2, y2, length))
        elif angle < 25 or angle > 155:
            horizontal_lines.append((x1, y1, x2, y2, length))
    
    print(f"Classified: {len(vertical_lines)} vertical, {len(horizontal_lines)} horizontal")
    return vertical_lines, horizontal_lines

def cluster_and_merge_lines(lines, frame_shape, is_vertical=True):
    """Advanced clustering with DBSCAN-like approach"""
    if len(lines) == 0:
        return []
    
    # Extract position and length
    lines_with_pos = []
    for line in lines:
        x1, y1, x2, y2, length = line
        if is_vertical:
            pos = (x1 + x2) / 2
        else:
            pos = (y1 + y2) / 2
        lines_with_pos.append((line, pos, length))
    
    # Sort by position
    lines_with_pos.sort(key=lambda x: x[1])
    
    # Cluster nearby lines
    clusters = []
    current_cluster = [lines_with_pos[0]]
    
    for i in range(1, len(lines_with_pos)):
        prev_pos = current_cluster[-1][1]
        curr_pos = lines_with_pos[i][1]
        
        # Dynamic threshold based on image size
        threshold = frame_shape[1 if is_vertical else 0] * 0.04  # 4% of dimension
        
        if abs(curr_pos - prev_pos) < threshold:
            current_cluster.append(lines_with_pos[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [lines_with_pos[i]]
    
    clusters.append(current_cluster)
    
    # Merge each cluster by weighted average (weight by length)
    merged_lines = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        
        # Weight by line length
        total_weight = sum(item[2] for item in cluster)
        
        avg_x1 = sum(item[0][0] * item[2] for item in cluster) / total_weight
        avg_y1 = sum(item[0][1] * item[2] for item in cluster) / total_weight
        avg_x2 = sum(item[0][2] * item[2] for item in cluster) / total_weight
        avg_y2 = sum(item[0][3] * item[2] for item in cluster) / total_weight
        
        merged_lines.append((int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)))
    
    # Extend lines to frame boundaries
    extended_lines = []
    for line in merged_lines:
        extended = extend_line_to_boundaries(line, frame_shape)
        extended_lines.append(extended)
    
    return extended_lines

def extend_line_to_boundaries(line, frame_shape):
    """Extend line to image boundaries"""
    x1, y1, x2, y2 = line
    h, w = frame_shape[:2]
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0:  # Vertical line
        return (x1, 0, x1, h-1)
    
    if dy == 0:  # Horizontal line
        return (0, y1, w-1, y1)
    
    # Line equation: y - y1 = m(x - x1)
    m = dy / dx
    
    # Find intersections with boundaries
    points = []
    
    # Left boundary (x=0)
    y = int(y1 + m * (0 - x1))
    if 0 <= y < h:
        points.append((0, y))
    
    # Right boundary (x=w-1)
    y = int(y1 + m * (w-1 - x1))
    if 0 <= y < h:
        points.append((w-1, y))
    
    # Top boundary (y=0)
    x = int(x1 + (0 - y1) / m)
    if 0 <= x < w:
        points.append((x, 0))
    
    # Bottom boundary (y=h-1)
    x = int(x1 + (h-1 - y1) / m)
    if 0 <= x < w:
        points.append((x, h-1))
    
    if len(points) >= 2:
        # Take the two most distant points
        if len(points) > 2:
            points = sorted(points, key=lambda p: p[0] + p[1])
            points = [points[0], points[-1]]
        return (points[0][0], points[0][1], points[1][0], points[1][1])
    
    return line

def line_intersection(line1, line2):
    """Find intersection of two lines"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None
    
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
    
    return (int(px), int(py))

def find_court_corners_robust(vertical_lines, horizontal_lines, frame_shape):
    """Find court corners with multiple validation steps"""
    h, w = frame_shape[:2]
    
    # Get all intersections
    intersections = []
    for vline in vertical_lines:
        for hline in horizontal_lines:
            pt = line_intersection(vline, hline)
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                intersections.append(pt)
    
    if len(intersections) < 4:
        print(f"Only {len(intersections)} intersections found")
        return None, intersections
    
    points = np.array(intersections)
    
    # Method 1: Find extreme corners
    def find_corner_candidates(points, method='extreme'):
        if method == 'extreme':
            tl = points[np.argmin(points[:, 0] + points[:, 1])]
            tr = points[np.argmax(points[:, 0] - points[:, 1])]
            bl = points[np.argmax(points[:, 1] - points[:, 0])]
            br = points[np.argmax(points[:, 0] + points[:, 1])]
            return np.array([tl, tr, bl, br])
        
        elif method == 'convex_hull':
            hull = cv2.convexHull(points)
            hull_points = hull.squeeze()
            
            if len(hull_points) >= 4:
                # Find the 4 most extreme points on the hull
                tl = hull_points[np.argmin(hull_points[:, 0] + hull_points[:, 1])]
                tr = hull_points[np.argmax(hull_points[:, 0] - hull_points[:, 1])]
                bl = hull_points[np.argmax(hull_points[:, 1] - hull_points[:, 0])]
                br = hull_points[np.argmax(hull_points[:, 0] + hull_points[:, 1])]
                return np.array([tl, tr, bl, br])
        
        return None
    
    # Try both methods
    corners_extreme = find_corner_candidates(points, 'extreme')
    corners_hull = find_corner_candidates(points, 'convex_hull')
    
    # Validate corners (should form a reasonable quadrilateral)
    def validate_corners(corners):
        if corners is None or len(corners) != 4:
            return False
        
        # Check if corners form a convex quadrilateral
        area = cv2.contourArea(corners)
        if area < (w * h * 0.05):  # Too small (less than 5% of image)
            return False
        
        # Check aspect ratio (courts are roughly 2.3:1)
        rect = cv2.minAreaRect(corners)
        width, height = rect[1]
        if width == 0 or height == 0:
            return False
        
        aspect = max(width, height) / min(width, height)
        if aspect < 1.5 or aspect > 4.0:  # Too extreme
            return False
        
        return True
    
    # Choose best corners
    if validate_corners(corners_extreme):
        corners = corners_extreme
    elif validate_corners(corners_hull):
        corners = corners_hull
    else:
        # Fallback: use convex hull approach
        hull = cv2.convexHull(points)
        if len(hull) >= 4:
            # Approximate to 4 corners
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) >= 4:
                corners = approx.squeeze()[:4]
            else:
                corners = hull.squeeze()[:4]
        else:
            print("Could not validate corners")
            return None, intersections
    
    # Ensure correct ordering: [TL, TR, BL, BR]
    corners = order_corners(corners)
    
    return corners.astype(np.float32), intersections

def order_corners(corners):
    """Order corners as [top-left, top-right, bottom-left, bottom-right]"""
    # Sort by y-coordinate
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    
    # Top two points
    top = sorted_by_y[:2]
    top = top[np.argsort(top[:, 0])]  # Sort by x
    tl, tr = top[0], top[1]
    
    # Bottom two points
    bottom = sorted_by_y[2:]
    bottom = bottom[np.argsort(bottom[:, 0])]  # Sort by x
    bl, br = bottom[0], bottom[1]
    
    return np.array([tl, tr, bl, br])

def visualize_detection(frame, vertical_lines, horizontal_lines, corners, intersections, methods_debug=None):
    """Comprehensive visualization"""
    result = frame.copy()
    
    # Draw vertical lines (blue)
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw horizontal lines (green)
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw all intersections (yellow, small)
    for pt in intersections:
        cv2.circle(result, pt, 3, (0, 255, 255), -1)
    
    # Draw corners (red, large)
    if corners is not None:
        labels = ['TL', 'TR', 'BL', 'BR']
        for i, pt in enumerate(corners):
            pt_int = tuple(pt.astype(int))
            cv2.circle(result, pt_int, 12, (0, 0, 255), -1)
            cv2.circle(result, pt_int, 16, (0, 0, 255), 3)
            cv2.putText(result, labels[i], (pt_int[0]+20, pt_int[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw court outline
        corners_int = corners.astype(np.int32)
        cv2.polylines(result, [corners_int[[0,1,3,2]]], True, (255, 0, 255), 3)
    
    return result

def warp_court(frame, corners):
    """Warp to top-down view"""
    if corners is None:
        return None
    
    # Tennis court: 23.77m x 10.97m ≈ 2.17:1
    width = 800
    height = 368
    
    dst = np.array([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    
    return warped

def main(image_path):
    """Main pipeline with multiple tests"""
    frame = load_and_preprocess(image_path)
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"Image shape: {frame.shape}")
    print(f"{'='*60}\n")
    
    # Step 1: Region mask
    region_mask = create_adaptive_region_mask(frame)
    
    # Step 2: Multi-method line detection
    edge_mask, debug_masks = multi_method_line_detection(frame, region_mask)
    
    # Show detection methods
    cv2.imshow("A. Region Mask", region_mask)
    cv2.imshow("B. Combined Edge Detection", edge_mask)
    
    # Step 3: Detect lines at multiple scales
    v_lines, h_lines = detect_lines_multi_scale(edge_mask, frame.shape)
    
    if len(v_lines) < 2 or len(h_lines) < 2:
        print("❌ Insufficient lines detected!")
        cv2.waitKey(0)
        return
    
    # Step 4: Cluster and merge
    v_lines_merged = cluster_and_merge_lines(v_lines, frame.shape, is_vertical=True)
    h_lines_merged = cluster_and_merge_lines(h_lines, frame.shape, is_vertical=False)
    
    print(f"\nAfter clustering:")
    print(f"  Vertical lines: {len(v_lines_merged)}")
    print(f"  Horizontal lines: {len(h_lines_merged)}")
    
    # Step 5: Find corners
    corners, intersections = find_court_corners_robust(v_lines_merged, h_lines_merged, frame.shape)
    
    print(f"\nIntersections found: {len(intersections)}")
    
    # Step 6: Visualize
    result = visualize_detection(frame, v_lines_merged, h_lines_merged, corners, intersections)
    cv2.imshow("C. Detection Result", result)
    
    # Step 7: Warp
    if corners is not None:
        print("✅ Successfully found 4 corners!")
        warped = warp_court(frame, corners)
        if warped is not None:
            cv2.imshow("D. Warped Court", warped)
    else:
        print("❌ Could not find valid 4 corners")
    
    print(f"\n{'='*60}\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "data/frame.png"
    main(image_path)