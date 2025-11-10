import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque


def detect_court_corners(image):
    """
    Detect tennis court corners using perspective-aware quadrant search.
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        corners: Dictionary with corner coordinates {top_left, top_right, bottom_left, bottom_right}
    """
    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Divide into quadrants
    h, w = edges.shape
    mid_h, mid_w = h // 2, w // 2
    
    quadrants = {
        'top_left': edges[0:mid_h, 0:mid_w],
        'top_right': edges[0:mid_h, mid_w:w],
        'bottom_left': edges[mid_h:h, 0:mid_w],
        'bottom_right': edges[mid_h:h, mid_w:w]
    }
    
    # Find corners in each quadrant
    corners = {}
    corner_types = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    
    for (quad_name, quad_edges), corner_type in zip(quadrants.items(), corner_types):
        corner = _find_corner(quad_edges, corner_type)
        
        if corner is not None:
            x, y = corner
            # Adjust to full image coordinates
            if 'right' in quad_name:
                x += mid_w
            if 'bottom' in quad_name:
                y += mid_h
            corners[quad_name] = (x, y)
    
    return corners


def _find_corner(quad_edges, corner_type):
    """Find corner in quadrant based on perspective-aware search regions."""
    h, w = quad_edges.shape
    
    if corner_type == 'top-left':
        search_h = int(h * 0.3)
        search_w_start, search_w_end = int(w * 0.2), int(w * 0.8)
        region = quad_edges[0:search_h, search_w_start:search_w_end]
        offset_y, offset_x = 0, search_w_start
        
    elif corner_type == 'top-right':
        search_h = int(h * 0.3)
        search_w_start, search_w_end = int(w * 0.2), int(w * 0.8)
        region = quad_edges[0:search_h, search_w_start:search_w_end]
        offset_y, offset_x = 0, search_w_start
        
    elif corner_type == 'bottom-left':
        search_h, search_w = int(h * 0.4), int(w * 0.4)
        region = quad_edges[-search_h:, 0:search_w]
        offset_y, offset_x = h - search_h, 0
        
    elif corner_type == 'bottom-right':
        search_h, search_w = int(h * 0.4), int(w * 0.4)
        region = quad_edges[-search_h:, -search_w:]
        offset_y, offset_x = h - search_h, w - search_w
    
    # Find edge points
    y_coords, x_coords = np.where(region > 0)
    if len(x_coords) == 0:
        return None
    
    # Score based on position
    reg_h, reg_w = region.shape
    if 'top-left' in corner_type:
        scores = (reg_h - y_coords) * 3 + (reg_w - x_coords) * 1
    elif 'top-right' in corner_type:
        scores = (reg_h - y_coords) * 3 + x_coords * 1
    elif 'bottom-left' in corner_type:
        scores = y_coords * 2 + (reg_w - x_coords) * 2
    elif 'bottom-right' in corner_type:
        scores = y_coords * 2 + x_coords * 2
    
    best_idx = np.argmax(scores)
    return (x_coords[best_idx] + offset_x, y_coords[best_idx] + offset_y)


def expand_court_boundaries(corners, pixels=50):
    """
    Expand court corners outward by fixed pixel amounts.
    
    Args:
        corners: Dictionary with 4 corner coordinates
        pixels: Number of pixels to expand on each side
    
    Returns:
        expanded_corners: Dictionary with expanded corner coordinates
    """
    expanded = {}
    
    # Top left: move left and up
    tl = corners['top_left']
    expanded['top_left'] = (tl[0] - pixels, tl[1] - pixels)
    
    # Top right: move right and up
    tr = corners['top_right']
    expanded['top_right'] = (tr[0] + pixels, tr[1] - pixels)
    
    # Bottom left: move left and down
    bl = corners['bottom_left']
    expanded['bottom_left'] = (bl[0] - pixels, bl[1] + pixels)
    
    # Bottom right: move right and down
    br = corners['bottom_right']
    expanded['bottom_right'] = (br[0] + pixels, br[1] + pixels)
    
    return expanded


def map_corners_to_full_image(corners, crop_region):
    """
    Map corners from cropped image coordinates to full image coordinates.
    
    Args:
        corners: Dictionary with corner coordinates in cropped image
        crop_region: Tuple (y1, y2, x1, x2) defining the crop region
    
    Returns:
        mapped_corners: Dictionary with corners in full image coordinates
    """
    y1, y2, x1, x2 = crop_region
    
    mapped = {}
    for corner_name, (x, y) in corners.items():
        # Add crop offset to map to full image
        mapped[corner_name] = (x + x1, y + y1)
    
    return mapped


def warp_perspective(image, corners, expand_pixels=50, vertical_padding_ratio=0.4):
    """
    Apply perspective transformation to get bird's eye view.
    Output is slightly taller than wide (aspect ratio ~0.85).
    
    Args:
        image: Input image (full size)
        corners: Dictionary with 4 corner coordinates (in full image coordinates)
        expand_pixels: Pixels to expand boundaries outward (especially vertical for out-of-bounds)
        vertical_padding_ratio: Additional vertical space ratio (e.g., 0.4 = 40% extra vertical space)
    
    Returns:
        warped: Bird's eye view image
    """
    # Expand corners outward
    if expand_pixels > 0:
        corners = expand_court_boundaries(corners, expand_pixels)
    
    # Order corners: top_left, top_right, bottom_right, bottom_left
    corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
    src_points = np.array([corners[name] for name in corner_order], dtype=np.float32)
    
    # Calculate output dimensions - slightly taller than wide
    avg_width = (np.linalg.norm(src_points[0] - src_points[1]) + 
                 np.linalg.norm(src_points[3] - src_points[2])) / 2
    
    output_width = int(avg_width)
    base_height = int(output_width * 1.2)  # Slightly taller than wide
    
    # Add extra vertical padding for out-of-bounds coverage
    vertical_padding = int(base_height * vertical_padding_ratio)
    output_height = base_height + vertical_padding
    
    # Distribute vertical padding (top and bottom for baselines)
    top_pad = vertical_padding // 2
    bottom_pad = vertical_padding - top_pad
    
    # Destination points
    dst_points = np.array([
        [0, top_pad],
        [output_width, top_pad],
        [output_width, top_pad + base_height],
        [0, top_pad + base_height]
    ], dtype=np.float32)
    
    # Apply perspective transformation
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (output_width, output_height))
    
    return warped

def visualize_results(original, corners, warped):
    """Display original with corners and warped perspective."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Image with corners
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
    
    # Warped perspective
    if warped is not None:
        axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Warped View')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_image(image_path, crop_region=None, expand_pixels=100, vertical_padding_ratio=0.4):
    """
    Process a single image: detect corners on cropped region, warp full image.
    
    Args:
        image_path: Path to image file
        crop_region: (y1, y2, x1, x2) crop region for corner detection
        expand_pixels: Pixels to expand court boundaries outward
        vertical_padding_ratio: Extra vertical space ratio for out-of-bounds (e.g., 0.4 = 40%)
    """
    # Load full image
    full_img = cv2.imread(image_path)
    
    if full_img is None:
        print(f"Error: Could not load image from '{image_path}'")
        return None, None
    
    # Crop for corner detection
    if crop_region:
        y1, y2, x1, x2 = crop_region
        cropped_img = full_img[y1:y2, x1:x2]
    else:
        cropped_img = full_img
        crop_region = (0, full_img.shape[0], 0, full_img.shape[1])
    
    # Detect corners on cropped image
    corners = detect_court_corners(cropped_img)
    
    if len(corners) != 4:
        print("Warning: Could not detect all 4 corners")
        visualize_results(cropped_img, corners, None)
        return corners, None
    
    # Map corners to full image coordinates
    full_corners = map_corners_to_full_image(corners, crop_region)
    
    # Warp full image using mapped corners
    warped = warp_perspective(full_img, full_corners, expand_pixels, vertical_padding_ratio)
    
    # Visualize: show cropped image with detected corners, and warped full image
    visualize_results(cropped_img, corners, warped)
    
    return full_corners, warped


def process_video(video_path, crop_region=None, expand_pixels=50, vertical_padding_ratio=0.3):
    """
    Process video: detect corners on cropped region, track players, warp full frames.
    
    Args:
        video_path: Path to video file
        crop_region: (y1, y2, x1, x2) crop region for corner detection
        expand_pixels: Pixels to expand court boundaries
        vertical_padding_ratio: Extra vertical space ratio for out-of-bounds
    """
    cap = cv2.VideoCapture(video_path)
    
    # Initialize player tracker
    tracker = PlayerTracker(history=500, var_threshold=40)
    
    # Detect corners from first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read video")
        return
    
    if crop_region:
        y1, y2, x1, x2 = crop_region
        cropped_frame = first_frame[y1:y2, x1:x2]
    else:
        cropped_frame = first_frame
        crop_region = (0, first_frame.shape[0], 0, first_frame.shape[1])
    
    # Detect corners once
    corners = detect_court_corners(cropped_frame)
    
    if len(corners) != 4:
        print("Warning: Could not detect all 4 corners")
        return
    
    # Map to full image coordinates
    full_corners = map_corners_to_full_image(corners, crop_region)
    
    # Create court mask with buffer for tracking
    court_mask = create_court_mask(first_frame.shape, full_corners, buffer_pixels=200)
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, full_frame = cap.read()
        if not ret:
            break
        
        # Apply court mask to focus tracking
        masked_frame = cv2.bitwise_and(full_frame, full_frame, mask=court_mask)
        
        # Track players
        players, fg_mask = tracker.track(masked_frame)
        
        # Draw players on original frame
        display_frame = draw_players(full_frame, players, full_corners)
        
        # Warp full frame
        warped = warp_perspective(full_frame, full_corners, expand_pixels, vertical_padding_ratio)
        
        # Draw players on warped view
        # Map player positions to warped coordinates
        warped_display = warped.copy()
        if len(players) > 0:
            corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
            src_points = np.array([full_corners[name] for name in corner_order], dtype=np.float32)
            
            # Calculate warped dimensions
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
            
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            for idx, (player_id, player_data) in enumerate(players.items()):
                centroid = player_data['centroid']
                color = colors[idx % len(colors)]
                
                # Transform centroid to warped coordinates
                pt = np.array([[[centroid[0], centroid[1]]]], dtype=np.float32)
                warped_pt = cv2.perspectiveTransform(pt, matrix)
                wx, wy = int(warped_pt[0][0][0]), int(warped_pt[0][0][1])
                
                # Draw on warped view
                cv2.circle(warped_display, (wx, wy), 8, color, -1)
                cv2.putText(warped_display, f"P{player_id}", (wx + 10, wy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display
        cv2.imshow('Original - Player Tracking', display_frame)
        cv2.imshow('Warped - Bird\'s Eye View', warped_display)
        cv2.imshow('Foreground Mask', fg_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


class PlayerTracker:
    """Track tennis players using background subtraction and centroid tracking."""
    
    def __init__(self, history=300, var_threshold=40):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True
        )
        self.players = {}  # {player_id: {'centroid': (x,y), 'bbox': (x,y,w,h), 'history': deque}}
        self.next_id = 0
        self.max_distance = 100  # Max distance for matching centroids
        
    def get_centroid(self, bbox):
        """Calculate centroid of bounding box."""
        x, y, w, h = bbox
        return (int(x + w/2), int(y + h/2))
    
    def detect_blobs(self, frame):
        """Detect moving blobs (players) using background subtraction."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (gray pixels)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        detected_blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            print("AREA", area)
            
            # Filter by area (players are typically 1000-20000 pixels)
            if 250 < area < 25000:
                x, y, w, h = cv2.boundingRect(c)
                if w > 0:
                    aspect = h / w
                else:
                    aspect = 0

                # Filter by aspect ratio (players are taller than wide)
                if 1.1 < aspect < 4.0:
                    detected_blobs.append((x, y, w, h))
        
        return detected_blobs, fg_mask
    
    def update_tracks(self, detected_blobs):
        """Update player tracks using centroid matching."""
        current_centroids = [self.get_centroid(bbox) for bbox in detected_blobs]
        if len(self.players) == 0:
            # Initialize tracks
            for i, bbox in enumerate(detected_blobs):
                centroid = current_centroids[i]
                self.players[self.next_id] = {
                    'centroid': centroid,
                    'bbox': bbox,
                    'history': deque(maxlen=30)
                }
                self.players[self.next_id]['history'].append(centroid)
                self.next_id += 1
        else:
            # Match detected blobs to existing tracks
            used_blobs = set()
            updated_ids = set()
            
            for player_id, player_data in list(self.players.items()):
                old_centroid = player_data['centroid']
                
                # Find nearest new centroid
                min_dist = float('inf')
                best_idx = -1
                
                for i, new_centroid in enumerate(current_centroids):
                    if i in used_blobs:
                        continue
                    
                    dist = np.sqrt((old_centroid[0] - new_centroid[0])**2 + 
                                   (old_centroid[1] - new_centroid[1])**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                
                # Update if match found
                if best_idx != -1 and min_dist < self.max_distance:
                    self.players[player_id]['centroid'] = current_centroids[best_idx]
                    self.players[player_id]['bbox'] = detected_blobs[best_idx]
                    self.players[player_id]['history'].append(current_centroids[best_idx])
                    used_blobs.add(best_idx)
                    updated_ids.add(player_id)
                else:
                    # Lost track - remove player
                    del self.players[player_id]
            
            # Add new tracks for unmatched blobs
            for i, bbox in enumerate(detected_blobs):
                if i not in used_blobs:
                    centroid = current_centroids[i]
                    self.players[self.next_id] = {
                        'centroid': centroid,
                        'bbox': bbox,
                        'history': deque(maxlen=30)
                    }
                    self.players[self.next_id]['history'].append(centroid)
                    self.next_id += 1
    
    def track(self, frame):
        """Main tracking function."""
        detected_blobs, fg_mask = self.detect_blobs(frame)
        self.update_tracks(detected_blobs)
        return self.players, fg_mask


def draw_players(frame, players, corners=None):
    """Draw player bounding boxes and tracks on frame."""
    display = frame.copy()
    
    # Draw court boundaries if available
    if corners and len(corners) == 4:
        corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        pts = np.array([corners[name] for name in corner_order], dtype=np.int32)
        cv2.polylines(display, [pts], True, (0, 255, 255), 2)
    
    # Draw players
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for idx, (player_id, player_data) in enumerate(players.items()):
        x, y, w, h = player_data['bbox']
        centroid = player_data['centroid']
        color = colors[idx % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        
        # Draw centroid
        cv2.circle(display, centroid, 5, color, -1)
        
        # Draw ID
        cv2.putText(display, f"P{player_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw track history
        history = list(player_data['history'])
        for i in range(1, len(history)):
            if history[i - 1] is None or history[i] is None:
                continue
            cv2.line(display, history[i - 1], history[i], color, 2)
    
    return display


def create_court_mask(image_shape, corners, buffer_pixels=100):
    """
    Create a mask of the court area with buffer for out-of-bounds tracking.
    
    Args:
        image_shape: (h, w) of the image
        corners: Dictionary with 4 corner coordinates
        buffer_pixels: Extra pixels around court boundaries
    
    Returns:
        mask: Binary mask of court + buffer area
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if len(corners) != 4:
        return mask
    
    # Expand corners for buffer
    expanded = expand_court_boundaries(corners, buffer_pixels)
    
    corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
    pts = np.array([expanded[name] for name in corner_order], dtype=np.int32)
    
    # Fill polygon
    cv2.fillPoly(mask, [pts], 255)
    
    return mask


if __name__ == "__main__":
    # Process image: detect corners on cropped image, warp full image
    # process_image('data/sample_frames/ss.png', 
    #               crop_region=(200, 900, 200, 1750), 
    #               expand_pixels=100, 
    #               vertical_padding_ratio=0.4)
    
    # Process video: warped view with vertical padding for out-of-bounds + player tracking
    process_video('data/full_video/tennis_full_video.mp4', 
                  crop_region=(150, 600, 200, 1100), 
                  expand_pixels=50, 
                  vertical_padding_ratio=0.5)