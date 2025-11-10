import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


def detect_white_lines(image):
    """Detect white court lines."""
    lower = np.array([170, 170, 90], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(image, lower, upper)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def create_edge_map(image):
    """
    Create strong edge map using Sobel + white line detection.
    
    Returns:
        edges: Binary edge map
    """
    mask = detect_white_lines(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel edges
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    
    # Combine Sobel with white mask
    combined = cv2.bitwise_and(sobel, mask)
    _, edges = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)
    
    return edges


def scan_horizontal_lines(edges, kernel_width=15, min_line_length=30):
    """
    Scan horizontally to detect lines and their intersections.
    
    Detects:
    - Line segments (continuous white pixels)
    - Start points (transition from black to white)
    - End points (transition from white to black)
    - Potential intersections (where vertical lines cross)
    
    Returns:
        line_segments: List of (y, x_start, x_end) horizontal segments
        transition_points: List of (x, y) transition points
    """
    h, w = edges.shape
    line_segments = []
    transition_points = []
    
    for y in range(h):
        in_line = False
        line_start = None
        white_count = 0
        
        for x in range(w):
            pixel = edges[y, x]
            
            if pixel > 127:  # White pixel
                if not in_line:
                    # Transition: black -> white (line start)
                    line_start = x
                    in_line = True
                    transition_points.append((x, y))
                white_count += 1
            else:  # Black pixel
                if in_line and white_count >= kernel_width:
                    # Transition: white -> black (line end)
                    line_segments.append((y, line_start, x - 1))
                    transition_points.append((x - 1, y))
                    in_line = False
                    white_count = 0
                elif in_line:
                    white_count = 0
                    in_line = False
        
        # Handle line extending to edge
        if in_line and white_count >= kernel_width:
            line_segments.append((y, line_start, w - 1))
    
    return line_segments, transition_points


def scan_vertical_lines(edges, kernel_height=15, min_line_length=30):
    """
    Scan vertically to detect lines and their intersections.
    
    Returns:
        line_segments: List of (x, y_start, y_end) vertical segments
        transition_points: List of (x, y) transition points
    """
    h, w = edges.shape
    line_segments = []
    transition_points = []
    
    for x in range(w):
        in_line = False
        line_start = None
        white_count = 0
        
        for y in range(h):
            pixel = edges[y, x]
            
            if pixel > 127:  # White pixel
                if not in_line:
                    # Transition: black -> white (line start)
                    line_start = y
                    in_line = True
                    transition_points.append((x, y))
                white_count += 1
            else:  # Black pixel
                if in_line and white_count >= kernel_height:
                    # Transition: white -> black (line end)
                    line_segments.append((x, line_start, y - 1))
                    transition_points.append((x, y - 1))
                    in_line = False
                    white_count = 0
                elif in_line:
                    white_count = 0
                    in_line = False
        
        # Handle line extending to edge
        if in_line and white_count >= kernel_height:
            line_segments.append((x, line_start, h - 1))
    
    return line_segments, transition_points


def detect_intersections_from_segments(h_segments, v_segments, tolerance=5):
    """
    Detect intersections where horizontal and vertical line segments cross.
    
    Args:
        h_segments: Horizontal segments [(y, x_start, x_end), ...]
        v_segments: Vertical segments [(x, y_start, y_end), ...]
        tolerance: Pixel tolerance for intersection detection
    
    Returns:
        intersections: List of (x, y) intersection points
    """
    intersections = []
    
    for y, x_start, x_end in h_segments:
        for x, y_start, y_end in v_segments:
            # Check if horizontal segment crosses vertical segment
            if (x_start - tolerance <= x <= x_end + tolerance and
                y_start - tolerance <= y <= y_end + tolerance):
                intersections.append((x, y))
    
    return intersections


def cluster_points(points, threshold=10):
    """Cluster nearby points and return centroids."""
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


def segments_to_lines(h_segments, v_segments):
    """
    Convert segments to full lines for visualization.
    
    Returns:
        lines: List of (x1, y1, x2, y2) lines
    """
    lines = []
    
    # Horizontal segments
    for y, x_start, x_end in h_segments:
        lines.append((x_start, y, x_end, y))
    
    # Vertical segments
    for x, y_start, y_end in v_segments:
        lines.append((x, y_start, x, y_end))
    
    return lines


def detect_court_lines_and_corners(image, kernel_size=15):
    """
    Novel line and corner detection using kernel scanning.
    
    Args:
        image: Input image (BGR)
        kernel_size: Size of scanning kernel
    
    Returns:
        corners: List of (x, y) corner/intersection points
        lines: List of (x1, y1, x2, y2) line segments
    """
    # Create edge map
    edges = create_edge_map(image)
    
    # Scan horizontally and vertically
    h_segments, h_transitions = scan_horizontal_lines(edges, kernel_width=kernel_size)
    v_segments, v_transitions = scan_vertical_lines(edges, kernel_height=kernel_size)
    
    # Detect intersections
    intersections = detect_intersections_from_segments(h_segments, v_segments, tolerance=8)
    
    # Cluster all transition and intersection points
    all_points = h_transitions + v_transitions + intersections
    corners = cluster_points(all_points, threshold=12)
    
    # Convert segments to lines for visualization
    lines = segments_to_lines(h_segments, v_segments)
    
    return corners, lines, edges


def filter_court_corners(corners, image_shape):
    """Filter to get the 4 main court corners."""
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
    """Warp image to bird's eye view."""
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


def visualize(original, all_corners, main_corners, lines, edges, warped):
    """Display results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Edge map
    axes[0, 0].imshow(edges, cmap='gray')
    axes[0, 0].set_title('Edge Map (Sobel + White Mask)')
    axes[0, 0].axis('off')
    
    # Lines
    img_lines = original.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    axes[0, 1].imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Detected Line Segments ({len(lines)})')
    axes[0, 1].axis('off')
    
    # Corners
    img_corners = original.copy()
    for x, y in all_corners:
        cv2.circle(img_corners, (int(x), int(y)), 3, (255, 0, 0), -1)
    
    if len(main_corners) == 4:
        corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        pts = np.array([main_corners[name] for name in corner_order], dtype=np.int32)
        cv2.polylines(img_corners, [pts], True, (0, 255, 255), 2)
        
        for pt in pts:
            cv2.circle(img_corners, tuple(pt), 5, (0, 0, 255), -1)
    
    axes[1, 0].imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Corners ({len(all_corners)}) + Main 4')
    axes[1, 0].axis('off')
    
    # Warped
    if warped is not None:
        axes[1, 1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Warped Bird\'s Eye View')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_image(image_path, crop_region=None, expand_pixels=100, 
                  vertical_padding_ratio=0.4, kernel_size=15):
    """Process image using novel kernel-based line detection."""
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
    
    # Detect using kernel scanning
    all_corners, lines, edges = detect_court_lines_and_corners(cropped_img, kernel_size)
    
    # Filter to get 4 main corners
    main_corners = filter_court_corners(all_corners, cropped_img.shape)
    
    if len(main_corners) != 4:
        print(f"Warning: Detected {len(main_corners)} main corners instead of 4")
        visualize(cropped_img, all_corners, main_corners, lines, edges, None)
        return main_corners, None
    
    # Map to full image coordinates
    full_corners = map_corners_to_full_image(main_corners, crop_region)
    
    # Warp full image
    warped = warp_court(full_img, full_corners, expand_pixels, vertical_padding_ratio)
    
    visualize(cropped_img, all_corners, main_corners, lines, edges, warped)
    
    return full_corners, warped


def process_video(video_path, crop_region=None, expand_pixels=50, 
                  vertical_padding_ratio=0.3, kernel_size=15):
    """Process video using kernel-based detection."""
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
        
        # Detect using kernel scanning
        all_corners, lines, edges = detect_court_lines_and_corners(cropped_frame, kernel_size)
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
            
            cv2.imshow('Edges', edges)
            cv2.imshow('Lines & Corners', display_frame)
            cv2.imshow('Warped View', warped)
        else:
            cv2.imshow('Edges', edges)
            cv2.imshow('Lines & Corners', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_image('data/sample_frames/output.png', 
                  crop_region=(150, 600, 200, 1100), 
                  expand_pixels=100, 
                  vertical_padding_ratio=0.4,
                  kernel_size=15)
    
    # process_video('data/full_video/tennis_full_video.mp4', 
    #               crop_region=(150, 600, 200, 1100), 
    #               expand_pixels=50, 
    #               vertical_padding_ratio=0.3,
    #               kernel_size=15)
