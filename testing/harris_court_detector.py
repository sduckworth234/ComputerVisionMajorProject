import cv2
import numpy as np
from matplotlib import pyplot as plt


def detect_court_corners_harris(image):
    """
    Detect tennis court corners using Harris corner detection.
    
    Args:
        image: Input image (BGR)
    
    Returns:
        corners: List of (x, y) corner coordinates
    """
    # Color mask for white court lines
    lower = np.array([180, 180, 100], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to grayscale
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    
    # Harris corner detection
    corners = cv2.cornerHarris(gray, blockSize=9, ksize=3, k=0.01)
    
    # Normalize and threshold
    corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Dilate to group nearby corners
    dilated = cv2.dilate(thresh, np.ones((3, 3), np.uint8))
    
    # Find contours and get centers
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corner_points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            corner_points.append((cx, cy))
    
    return corner_points


def filter_court_corners(corners, image_shape):
    """
    Filter and select the 4 main court corners.
    
    Args:
        corners: List of all detected corner points
        image_shape: Shape of the image (h, w)
    
    Returns:
        Dictionary with 4 main corners {top_left, top_right, bottom_left, bottom_right}
    """
    if len(corners) < 4:
        return {}
    
    h, w = image_shape[:2]
    
    # Sort by y (top to bottom)
    sorted_by_y = sorted(corners, key=lambda p: p[1])
    
    # Top 2 corners (smallest y)
    top_corners = sorted(sorted_by_y[:len(sorted_by_y)//2], key=lambda p: p[0])
    top_left = top_corners[0]
    top_right = top_corners[-1]
    
    # Bottom 2 corners (largest y)
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
    
    # Calculate dimensions (slightly taller than wide)
    avg_width = (np.linalg.norm(src_points[0] - src_points[1]) + 
                 np.linalg.norm(src_points[3] - src_points[2])) / 2
    
    output_width = int(avg_width)
    base_height = int(output_width * 1.2)
    
    # Add vertical padding
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


def visualize(original, corners, warped):
    """Display results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    img_with_corners = original.copy()
    
    if len(corners) == 4:
        corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        pts = np.array([corners[name] for name in corner_order], dtype=np.int32)
        cv2.polylines(img_with_corners, [pts], True, (0, 255, 255), 2)
        
        for pt in pts:
            cv2.circle(img_with_corners, tuple(pt), 5, (255, 0, 0), -1)
    
    axes[0].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Detected Corners')
    axes[0].axis('off')
    
    if warped is not None:
        axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Warped View')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_image(image_path, crop_region=None, expand_pixels=100, vertical_padding_ratio=0.4):
    """Process image: detect corners on cropped region, warp full image."""
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
    
    # Detect all corners on cropped image
    all_corners = detect_court_corners_harris(cropped_img)
    
    # Filter to get 4 main corners
    corners = filter_court_corners(all_corners, cropped_img.shape)
    
    if len(corners) != 4:
        print(f"Warning: Detected {len(corners)} corners instead of 4")
        return corners, None
    
    # Map to full image coordinates
    full_corners = map_corners_to_full_image(corners, crop_region)
    
    # Warp full image
    warped = warp_court(full_img, full_corners, expand_pixels, vertical_padding_ratio)
    
    visualize(cropped_img, corners, warped)
    
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
        
        # Detect corners
        all_corners = detect_court_corners_harris(cropped_frame)
        corners = filter_court_corners(all_corners, cropped_frame.shape)
        
        display_frame = cropped_frame.copy()
        
        if len(corners) == 4:
            corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
            pts = np.array([corners[name] for name in corner_order], dtype=np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)
            
            for pt in pts:
                cv2.circle(display_frame, tuple(pt), 5, (255, 0, 0), -1)
            
            full_corners = map_corners_to_full_image(corners, crop_region)
            warped = warp_court(full_frame, full_corners, expand_pixels, vertical_padding_ratio)
            
            cv2.imshow('Cropped - Corner Detection', display_frame)
            cv2.imshow('Full Frame - Warped View', warped)
        else:
            cv2.imshow('Cropped - Corner Detection', display_frame)
        
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
