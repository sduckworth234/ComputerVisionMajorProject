import cv2
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations


def detect_white_lines(image):
    """
    Detect white court lines on blue/green court.
    Enhanced to better detect faint/far lines.
    
    Args:
        image: Input image (BGR)
    
    Returns:
        mask: Binary mask of white lines
    """
    lower = np.array([170, 170, 90], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(image, lower, upper)
    
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def get_line_params(line):
    """Get line angle and approximate position."""
    x1, y1, x2, y2 = line
    angle = np.arctan2(y2 - y1, x2 - x1)
    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    return angle, midpoint


def are_lines_similar(line1, line2, angle_threshold=0.1, distance_threshold=20):
    """Check if two lines are similar (collinear/nearby)."""
    angle1, mid1 = get_line_params(line1)
    angle2, mid2 = get_line_params(line2)
    
    angle_diff = abs(angle1 - angle2)
    angle_diff = min(angle_diff, np.pi - angle_diff)
    
    if angle_diff > angle_threshold:
        return False
    
    distance = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
    return distance < distance_threshold


def merge_line_segments(lines):
    """Merge nearby collinear line segments into single lines."""
    if len(lines) == 0:
        return []
    
    merged = []
    used = [False] * len(lines)
    
    for i, line1 in enumerate(lines):
        if used[i]:
            continue
        
        x1, y1, x2, y2 = line1
        points = [(x1, y1), (x2, y2)]
        
        for j, line2 in enumerate(lines):
            if i == j or used[j]:
                continue
            
            if are_lines_similar(line1, line2):
                x3, y3, x4, y4 = line2
                points.extend([(x3, y3), (x4, y4)])
                used[j] = True
        
        points = np.array(points)
        p1_idx = np.argmin(points[:, 0] + points[:, 1])
        p2_idx = np.argmax(points[:, 0] + points[:, 1])
        
        x1, y1 = points[p1_idx]
        x2, y2 = points[p2_idx]
        
        merged.append((int(x1), int(y1), int(x2), int(y2)))
        used[i] = True
    
    return merged


def detect_hough_lines(mask):
    """
    Detect lines using Hough transform with improved parameters.
    
    Args:
        mask: Binary mask of court lines
    
    Returns:
        lines: List of merged lines in format [(x1, y1, x2, y2), ...]
    """
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    hough_lines = cv2.HoughLinesP(
        dilated,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=30,
        maxLineGap=15
    )
    
    if hough_lines is None:
        return []
    
    lines = []
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > 20:
            lines.append((x1, y1, x2, y2))
    
    merged_lines = merge_line_segments(lines)
    
    return merged_lines


def line_intersection(line1, line2):
    """
    Find intersection point of two lines.
    
    Args:
        line1: (x1, y1, x2, y2)
        line2: (x1, y1, x2, y2)
    
    Returns:
        (x, y) intersection point or None
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None
    
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    
    return (px, py)


def find_line_intersections(lines, image_shape):
    """
    Find all intersections between detected lines.
    
    Args:
        lines: List of lines [(x1, y1, x2, y2), ...]
        image_shape: (h, w) to filter out-of-bounds intersections
    
    Returns:
        intersections: List of (x, y) intersection points
    """
    h, w = image_shape[:2]
    intersections = []
    
    for line1, line2 in combinations(lines, 2):
        pt = line_intersection(line1, line2)
        
        if pt is not None:
            x, y = pt
            if 0 <= x < w and 0 <= y < h:
                intersections.append((x, y))
    
    return intersections


def cluster_nearby_points(points, threshold=8):
    """
    Cluster nearby intersection points and return centroids.
    
    Args:
        points: List of (x, y) points
        threshold: Distance threshold for clustering
    
    Returns:
        clustered: List of centroid points
    """
    if len(points) == 0:
        return []
    
    points = np.array(points)
    clustered = []
    used = np.zeros(len(points), dtype=bool)
    
    for i, pt in enumerate(points):
        if used[i]:
            continue
        
        distances = np.sqrt(np.sum((points - pt) ** 2, axis=1))
        nearby = distances < threshold
        
        cluster = points[nearby]
        centroid = cluster.mean(axis=0)
        
        clustered.append(tuple(centroid))
        used[nearby] = True
    
    return clustered


def detect_harris_corners(image):
    """
    Detect corners using Harris corner detection.
    
    Args:
        image: Input image (BGR)
    
    Returns:
        corners: List of (x, y) corner points
    """
    mask = detect_white_lines(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    
    corners = cv2.cornerHarris(gray, blockSize=9, ksize=3, k=0.01)
    corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    dilated = cv2.dilate(thresh, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corner_points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            corner_points.append((cx, cy))
    
    return corner_points


def detect_all_corners(image):
    """
    Detect all court corners using both Hough line intersections and Harris corners.
    
    Args:
        image: Input image (BGR)
    
    Returns:
        corners: List of (x, y) corner points
    """
    mask = detect_white_lines(image)
    lines = detect_hough_lines(mask)
    
    intersections = find_line_intersections(lines, image.shape)
    clustered_intersections = cluster_nearby_points(intersections, threshold=8)
    
    harris_corners = detect_harris_corners(image)
    clustered_harris = cluster_nearby_points(harris_corners, threshold=8)
    
    all_corners = clustered_intersections + clustered_harris
    final_corners = cluster_nearby_points(all_corners, threshold=10)
    
    return final_corners, lines


def filter_court_corners(corners, image_shape):
    """
    Filter to get the 4 main court corners.
    
    Args:
        corners: List of all corner points
        image_shape: (h, w)
    
    Returns:
        Dictionary with 4 main corners
    """
    if len(corners) < 4:
        return {}
    
    sorted_by_y = sorted(corners, key=lambda p: p[1])
    
    top_corners = sorted(sorted_by_y[:len(sorted_by_y)//2], key=lambda p: p[0])
    top_left = top_corners[0]
    top_right = top_corners[-1]
    
    bottom_corners = sorted(sorted_by_y[len(sorted_by_y)//2:], key=lambda p: p[0])
    bottom_left = bottom_corners[0]
    bottom_right = bottom_corners[-1]
    
    return {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right
    }


def map_corners_to_full_image(corners, crop_region):
    """Map corners from cropped to full image coordinates."""
    y1, y2, x1, x2 = crop_region
    
    mapped = {}
    for corner_name, (x, y) in corners.items():
        mapped[corner_name] = (x + x1, y + y1)
    
    return mapped


def expand_corners(corners, pixels=50):
    """Expand corners outward."""
    expanded = {}
    
    tl = corners['top_left']
    expanded['top_left'] = (tl[0] - pixels, tl[1] - pixels)
    
    tr = corners['top_right']
    expanded['top_right'] = (tr[0] + pixels, tr[1] - pixels)
    
    bl = corners['bottom_left']
    expanded['bottom_left'] = (bl[0] - pixels, bl[1] + pixels)
    
    br = corners['bottom_right']
    expanded['bottom_right'] = (br[0] + pixels, br[1] + pixels)
    
    return expanded


def warp_court(image, corners, expand_pixels=50, vertical_padding_ratio=0.4):
    """
    Warp image to bird's eye view.
    Output is slightly taller than wide.
    """
    if expand_pixels > 0:
        corners = expand_corners(corners, expand_pixels)
    
    corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
    src_points = np.array([corners[name] for name in corner_order], dtype=np.float32)
    
    avg_width = (np.linalg.norm(src_points[0] - src_points[1]) + 
                 np.linalg.norm(src_points[3] - src_points[2])) / 2
    
    output_width = int(avg_width)
    base_height = int(output_width * 1.2)
    
    vertical_padding = int(base_height * vertical_padding_ratio)
    output_height = base_height + vertical_padding
    
    top_pad = vertical_padding // 2
    
    dst_points = np.array([
        [0, top_pad],
        [output_width, top_pad],
        [output_width, top_pad + base_height],
        [0, top_pad + base_height]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (output_width, output_height))
    
    return warped


def visualize(original, all_corners, main_corners, lines, warped):
    """Display results with all detected lines and corners."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show all lines
    img_lines = original.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    axes[0].imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Detected Lines ({len(lines)})')
    axes[0].axis('off')
    
    # Show all corners
    img_corners = original.copy()
    for x, y in all_corners:
        cv2.circle(img_corners, (int(x), int(y)), 3, (255, 0, 0), -1)
    
    if len(main_corners) == 4:
        corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        pts = np.array([main_corners[name] for name in corner_order], dtype=np.int32)
        cv2.polylines(img_corners, [pts], True, (0, 255, 255), 2)
        
        for pt in pts:
            cv2.circle(img_corners, tuple(pt), 5, (0, 0, 255), -1)
    
    axes[1].imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'All Corners ({len(all_corners)}) + Main 4')
    axes[1].axis('off')
    
    # Show warped
    if warped is not None:
        axes[2].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Warped View')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_image(image_path, crop_region=None, expand_pixels=100, vertical_padding_ratio=0.4):
    """Process image: detect all lines and corners, warp full image."""
    full_img = cv2.imread(image_path)
    
    if full_img is None:
        print(f"Error: Could not load image from '{image_path}'")
        return None, None
    
    if crop_region:
        y1, y2, x1, x2 = crop_region
        cropped_img = full_img[y1:y2, x1:x2]
    else:
        cropped_img = full_img
        crop_region = (0, full_img.shape[0], 0, full_img.shape[1])
    
    # Detect all corners and lines
    all_corners, lines = detect_all_corners(cropped_img)
    
    # Filter to get 4 main corners
    main_corners = filter_court_corners(all_corners, cropped_img.shape)
    
    if len(main_corners) != 4:
        print(f"Warning: Detected {len(main_corners)} main corners instead of 4")
        visualize(cropped_img, all_corners, main_corners, lines, None)
        return main_corners, None
    
    # Map to full image coordinates
    full_corners = map_corners_to_full_image(main_corners, crop_region)
    
    # Warp full image
    warped = warp_court(full_img, full_corners, expand_pixels, vertical_padding_ratio)
    
    visualize(cropped_img, all_corners, main_corners, lines, warped)
    
    return full_corners, warped


def process_video(video_path, crop_region=None, expand_pixels=50, vertical_padding_ratio=0.3):
    """Process video: detect corners on cropped region, warp full frames."""
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, full_frame = cap.read()
        if not ret:
            break
        
        if crop_region:
            y1, y2, x1, x2 = crop_region
            cropped_frame = full_frame[y1:y2, x1:x2]
        else:
            cropped_frame = full_frame
            crop_region = (0, full_frame.shape[0], 0, full_frame.shape[1])
        
        # Detect corners and lines
        all_corners, lines = detect_all_corners(cropped_frame)
        main_corners = filter_court_corners(all_corners, cropped_frame.shape)
        
        display_frame = cropped_frame.copy()
        
        # Draw lines
        for x1, y1, x2, y2 in lines:
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Draw all corners
        for x, y in all_corners:
            cv2.circle(display_frame, (int(x), int(y)), 2, (255, 0, 0), -1)
        
        if len(main_corners) == 4:
            corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
            pts = np.array([main_corners[name] for name in corner_order], dtype=np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)
            
            for pt in pts:
                cv2.circle(display_frame, tuple(pt), 5, (0, 0, 255), -1)
            
            full_corners = map_corners_to_full_image(main_corners, crop_region)
            warped = warp_court(full_frame, full_corners, expand_pixels, vertical_padding_ratio)
            
            cv2.imshow('Lines & Corners', display_frame)
            cv2.imshow('Warped View', warped)
        else:
            cv2.imshow('Lines & Corners', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_image('data/sample_frames/output.png', 
                  crop_region=(150, 600, 200, 1100), 
                  expand_pixels=100, 
                  vertical_padding_ratio=0.4)
    
    # process_video('data/full_video/tennis_full_video.mp4', 
    #               crop_region=(150, 600, 200, 1100), 
    #               expand_pixels=50, 
    #               vertical_padding_ratio=0.3)
