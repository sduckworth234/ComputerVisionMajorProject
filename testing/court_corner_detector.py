"""
Tennis Court Corner Detection & Player Tracking System

PIPELINE:
1. Detect court corners using edge detection and quadrant search
2. Map corners from cropped to full image coordinates
3. Create court mask with buffer zone
4. Track players using background subtraction
5. Warp image to bird's eye view
6. Visualize player positions in real-time

Pure computer vision - no machine learning required!
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================================================================
# STAGE 1: COURT CORNER DETECTION
# =============================================================================

def detect_court_corners(image):
    """
    Detect 4 tennis court corners using perspective-aware quadrant search.
    
    Pipeline:
    - Convert to grayscale and detect edges
    - Divide image into 4 quadrants (one corner per quadrant)
    - Search for corner in each quadrant with perspective awareness
    - Adjust coordinates to full image
    
    Returns: (corner_tl, corner_tr, corner_bl, corner_br)
             Each corner is a tuple (x, y)
    """
    # STEP 1: Convert to grayscale and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Detect edges (white lines on court)
    
    # STEP 2: Divide image into 4 quadrants
    # Each quadrant should contain ONE court corner
    h, w = edges.shape
    mid_h, mid_w = h // 2, w // 2
    
    quad_top_left = edges[0:mid_h, 0:mid_w]
    quad_top_right = edges[0:mid_h, mid_w:w]
    quad_bottom_left = edges[mid_h:h, 0:mid_w]
    quad_bottom_right = edges[mid_h:h, mid_w:w]
    
    # STEP 3: Find corner in each quadrant
    corner_tl = find_corner_in_quadrant(quad_top_left, 'top-left')
    corner_tr = find_corner_in_quadrant(quad_top_right, 'top-right')
    corner_bl = find_corner_in_quadrant(quad_bottom_left, 'bottom-left')
    corner_br = find_corner_in_quadrant(quad_bottom_right, 'bottom-right')
    
    # STEP 4: Adjust coordinates from quadrant to full image
    # Top-left corner is already at (0, 0) origin - no adjustment needed
    # Other corners need offsets added
    if corner_tr:
        corner_tr = (corner_tr[0] + mid_w, corner_tr[1])
    if corner_bl:
        corner_bl = (corner_bl[0], corner_bl[1] + mid_h)
    if corner_br:
        corner_br = (corner_br[0] + mid_w, corner_br[1] + mid_h)
    
    return corner_tl, corner_tr, corner_bl, corner_br


def find_corner_in_quadrant(quad_edges, corner_type):
    """
    Find a specific corner within its quadrant.
    
    Uses perspective-aware search:
    - Far corners (top) appear smaller due to camera angle
      -> Search in smaller region (top 30%, middle 60% horizontally)
    - Near corners (bottom) are larger and clearer
      -> Search in larger region (bottom 40%, full 40% from edge)
    
    Scoring system:
    - Weights pixels based on position
    - Wants pixels at the extreme of the quadrant (actual corner location)
    """
    h, w = quad_edges.shape
    
    # Define search region based on corner type and perspective
    if corner_type == 'top-left':
        # Far corner: appears smaller, search top 30% and middle 60% horizontally
        search_h = int(h * 0.3)
        search_w_start, search_w_end = int(w * 0.2), int(w * 0.8)
        region = quad_edges[0:search_h, search_w_start:search_w_end]
        offset_x, offset_y = search_w_start, 0
        
    elif corner_type == 'top-right':
        # Far corner: same as top-left
        search_h = int(h * 0.3)
        search_w_start, search_w_end = int(w * 0.2), int(w * 0.8)
        region = quad_edges[0:search_h, search_w_start:search_w_end]
        offset_x, offset_y = search_w_start, 0
        
    elif corner_type == 'bottom-left':
        # Near corner: larger and clearer, search bottom 40% and left 40%
        search_h, search_w = int(h * 0.4), int(w * 0.4)
        region = quad_edges[-search_h:, 0:search_w]
        offset_x, offset_y = 0, h - search_h
        
    elif corner_type == 'bottom-right':
        # Near corner: search bottom 40% and right 40%
        search_h, search_w = int(h * 0.4), int(w * 0.4)
        region = quad_edges[-search_h:, -search_w:]
        offset_x, offset_y = w - search_w, h - search_h
    
    # Find all edge pixels (white pixels from Canny) in search region
    y_coords, x_coords = np.where(region > 0)
    if len(x_coords) == 0:
        return None  # No edges found
    
    # Score each pixel based on how "corner-like" its position is
    # Want pixels at the extremes (actual corner location)
    reg_h, reg_w = region.shape
    
    if 'top-left' in corner_type:
        # Want top-left extreme: prioritize top (high y score), then left (high x score)
        scores = (reg_h - y_coords) * 3 + (reg_w - x_coords) * 1
    elif 'top-right' in corner_type:
        # Want top-right extreme: prioritize top (high y score), then right (low x = far right)
        scores = (reg_h - y_coords) * 3 + x_coords * 1
    elif 'bottom-left' in corner_type:
        # Want bottom-left extreme: equal weight to bottom and left
        scores = y_coords * 2 + (reg_w - x_coords) * 2
    elif 'bottom-right' in corner_type:
        # Want bottom-right extreme: equal weight to bottom and right
        scores = y_coords * 2 + x_coords * 2
    
    # Return pixel with highest score (most corner-like position)
    best_idx = np.argmax(scores)
    corner_x = x_coords[best_idx] + offset_x
    corner_y = y_coords[best_idx] + offset_y
    
    return (corner_x, corner_y)


# =============================================================================
# STAGE 2: COORDINATE MAPPING & EXPANSION
# =============================================================================

def expand_corners(corner_tl, corner_tr, corner_bl, corner_br, pixels=50):
    """
    Expand court corners outward to capture out-of-bounds area.
    
    Each corner moves away from the court center:
    - Top-left: move left AND up
    - Top-right: move right AND up
    - Bottom-left: move left AND down
    - Bottom-right: move right AND down
    
    This creates a larger boundary to track players who run outside court lines.
    """
    exp_tl = (corner_tl[0] - pixels, corner_tl[1] - pixels)  # Left + up
    exp_tr = (corner_tr[0] + pixels, corner_tr[1] - pixels)  # Right + up
    exp_bl = (corner_bl[0] - pixels, corner_bl[1] + pixels)  # Left + down
    exp_br = (corner_br[0] + pixels, corner_br[1] + pixels)  # Right + down
    
    return exp_tl, exp_tr, exp_bl, exp_br


def map_corners_to_full_image(corner_tl, corner_tr, corner_bl, corner_br, crop_region):
    """
    Map corners from cropped image coordinates to full image coordinates.
    
    When we detect corners on a cropped region, the coordinates are relative
    to the crop. This function adds the crop offset to get full image coordinates.
    
    Example:
    - Crop region: (y1=150, y2=600, x1=200, x2=1100)
    - Corner detected at (100, 50) in crop
    - Full image coordinate: (100 + 200, 50 + 150) = (300, 200)
    """
    y1, y2, x1, x2 = crop_region
    
    # Add crop offset to each corner
    full_tl = (corner_tl[0] + x1, corner_tl[1] + y1)
    full_tr = (corner_tr[0] + x1, corner_tr[1] + y1)
    full_bl = (corner_bl[0] + x1, corner_bl[1] + y1)
    full_br = (corner_br[0] + x1, corner_br[1] + y1)
    
    return full_tl, full_tr, full_bl, full_br


def warp_to_birds_eye(image, corner_tl, corner_tr, corner_bl, corner_br, 
                       expand_pixels=50, vertical_padding_ratio=0.4):
    """
    Apply perspective transformation to get bird's eye view of the court.
    
    Pipeline:
    1. Expand corners outward (capture out-of-bounds area)
    2. Calculate output dimensions (slightly taller than wide)
    3. Add vertical padding (extra space for baseline players)
    4. Define source (oblique quad) and destination (perfect rectangle) points
    5. Compute transformation matrix
    6. Apply warp
    
    Math: Uses homography (projective transformation) to map quadrilateral -> rectangle
    """
    # STEP 1: Expand corners to capture out-of-bounds
    if expand_pixels > 0:
        corner_tl, corner_tr, corner_bl, corner_br = expand_corners(
            corner_tl, corner_tr, corner_bl, corner_br, expand_pixels
        )
    
    # STEP 2: Define source points (oblique quadrilateral in original image)
    src_points = np.array([corner_tl, corner_tr, corner_br, corner_bl], dtype=np.float32)
    
    # STEP 3: Calculate output dimensions
    # Measure top and bottom baseline lengths
    top_width = np.sqrt((corner_tr[0] - corner_tl[0])**2 + (corner_tr[1] - corner_tl[1])**2)
    bottom_width = np.sqrt((corner_br[0] - corner_bl[0])**2 + (corner_br[1] - corner_bl[1])**2)
    avg_width = (top_width + bottom_width) / 2
    
    output_width = int(avg_width)
    base_height = int(output_width * 1.2)  # Make output slightly taller than wide
    
    # STEP 4: Add vertical padding for out-of-bounds players
    vertical_padding = int(base_height * vertical_padding_ratio)
    output_height = base_height + vertical_padding
    top_pad = vertical_padding // 2  # Split padding between top and bottom
    
    # STEP 5: Define destination points (perfect rectangle in output image)
    dst_points = np.array([
        [0, top_pad],                           # Top-left
        [output_width, top_pad],                # Top-right
        [output_width, top_pad + base_height],  # Bottom-right
        [0, top_pad + base_height]              # Bottom-left
    ], dtype=np.float32)
    
    # STEP 6: Compute perspective transformation matrix
    # This 3x3 matrix maps oblique quadrilateral -> perfect rectangle
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # STEP 7: Apply transformation
    warped = cv2.warpPerspective(image, matrix, (output_width, output_height))
    
    return warped, matrix


def create_court_mask(image_shape, corner_tl, corner_tr, corner_bl, corner_br, buffer_pixels=150):
    """
    Create binary mask of court area + buffer zone.
    
    This mask is used to focus player tracking only on the court region,
    ignoring audience, advertisements, scoreboard, etc.
    
    Returns: Binary mask (white inside court+buffer, black outside)
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Expand corners by buffer pixels
    exp_tl, exp_tr, exp_bl, exp_br = expand_corners(
        corner_tl, corner_tr, corner_bl, corner_br, buffer_pixels
    )
    
    # Create polygon from expanded corners
    pts = np.array([exp_tl, exp_tr, exp_br, exp_bl], dtype=np.int32)
    
    # Fill polygon with white
    cv2.fillPoly(mask, [pts], 255)
    
    return mask



# =============================================================================
# STAGE 3: PLAYER TRACKING (PURE COMPUTER VISION)
# =============================================================================

class PlayerTracker:
    """
    Track tennis players using background subtraction and centroid matching.
    
    NO MACHINE LEARNING - Pure computer vision approach:
    1. Background subtraction (MOG2) - learns what doesn't move (court)
    2. Blob detection - finds moving objects (players)
    3. Centroid tracking - matches players frame-to-frame by position
    
    Attributes:
    - bg_subtractor: MOG2 background subtractor
    - player_list: List of [centroid, bbox, history] for each player
    - next_id: Counter for assigning IDs to new players
    - max_distance: Maximum distance for matching centroids (pixels)
    """
    
    def __init__(self, history=500, var_threshold=25):
        """
        Initialize player tracker.
        
        Args:
            history: Number of frames to learn background (500 = ~16 seconds at 30fps)
            var_threshold: Sensitivity to motion (lower = more sensitive)
        """
        # Create background subtractor
        # detectShadows=True marks shadows as gray (127) instead of white (255)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True
        )
        
        # Store players as list 
        # Each player: [player_id, centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h, history_list]
        self.player_list = []
        self.next_id = 0
        self.max_distance = 150  # Increased for far-end players who move faster in pixels
    
    def get_centroid(self, x, y, w, h):
        """Calculate center point of bounding box."""
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        return cx, cy
    
    def detect_blobs(self, frame):
        """
        Detect moving blobs (players) using background subtraction.
        
        Pipeline:
        1. Apply background subtraction
        2. Remove shadows (threshold gray pixels)
        3. Morphological operations (clean noise)
        4. Find contours
        5. Filter by area and aspect ratio
        
        Returns: List of bounding boxes [(x, y, w, h), ...], foreground mask
        """
        # STEP 1: Apply background subtraction
        # Compares each pixel to learned background model
        # Output: white=foreground (moving), black=background (static), gray=shadow
        fg_mask = self.bg_subtractor.apply(frame)
        
        # STEP 2: Remove shadows
        # MOG2 marks shadows as 127 (gray), we only want pure white (255) moving objects
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # STEP 3: Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)   # Remove small dots
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        
        # STEP 4: Find contours (outlines of white blobs)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # STEP 5: Filter contours by area and aspect ratio
        detected_blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            
            # Filter by area
            # Far-end players appear smaller due to perspective
            if 200 < area < 30000:
                x, y, w, h = cv2.boundingRect(c)
                
                # Calculate aspect ratio (height / width)
                if w > 0:
                    aspect = h / w
                else:
                    continue
                
                # Filter by aspect ratio (players are taller than wide)
                # Far-end players may appear more compressed
                if 0.8 < aspect < 5.0:
                    detected_blobs.append((x, y, w, h))
        
        return detected_blobs, fg_mask
    
    def update_tracks(self, detected_blobs):
        """
        Update player tracks using centroid matching.
        
        Pipeline:
        1. If first frame: initialize all tracks
        2. Otherwise: match new detections to existing players
           - For each existing player, find nearest new detection
           - Update position if match found (distance < max_distance)
           - Delete player if no match (left screen)
        3. Add new players for unmatched detections
        """
        # Calculate centroids for all detected blobs
        current_centroids = []
        for x, y, w, h in detected_blobs:
            cx, cy = self.get_centroid(x, y, w, h)
            current_centroids.append((cx, cy))
        
        # CASE 1: First frame or no existing players - initialize tracks
        if len(self.player_list) == 0:
            for i, bbox in enumerate(detected_blobs):
                x, y, w, h = bbox
                cx, cy = current_centroids[i]
                
                # Create new player: [id, cx, cy, x, y, w, h, history]
                # History stores last 30 positions for drawing trails
                new_player = [self.next_id, cx, cy, x, y, w, h, [(cx, cy)]]
                self.player_list.append(new_player)
                self.next_id += 1
        
        # CASE 2: Match detections to existing players
        else:
            used_blobs = set()  # Track which detections have been matched
            players_to_remove = []  # Track which players to delete
            
            # For each existing player, find best matching new detection
            for player_idx, player in enumerate(self.player_list):
                player_id, old_cx, old_cy = player[0], player[1], player[2]
                
                # Find nearest new detection
                min_dist = float('inf')
                best_blob_idx = -1
                
                for blob_idx, (new_cx, new_cy) in enumerate(current_centroids):
                    if blob_idx in used_blobs:
                        continue  # Already matched to another player
                    
                    # Calculate Euclidean distance
                    dist = np.sqrt((old_cx - new_cx)**2 + (old_cy - new_cy)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_blob_idx = blob_idx
                
                # Update if match found within max_distance
                if best_blob_idx != -1 and min_dist < self.max_distance:
                    x, y, w, h = detected_blobs[best_blob_idx]
                    new_cx, new_cy = current_centroids[best_blob_idx]
                    
                    # Update player data
                    player[1] = new_cx  # Update centroid x
                    player[2] = new_cy  # Update centroid y
                    player[3] = x       # Update bbox x
                    player[4] = y       # Update bbox y
                    player[5] = w       # Update bbox w
                    player[6] = h       # Update bbox h
                    player[7].append((new_cx, new_cy))  # Add to history
                    
                    # Keep only last 30 positions
                    if len(player[7]) > 30:
                        player[7].pop(0)
                    
                    used_blobs.add(best_blob_idx)
                else:
                    # No match found - player left screen
                    players_to_remove.append(player_idx)
            
            # Remove players who left screen (reverse order to maintain indices)
            for idx in reversed(players_to_remove):
                self.player_list.pop(idx)
            
            # Add new players for unmatched detections
            for blob_idx, bbox in enumerate(detected_blobs):
                if blob_idx not in used_blobs:
                    x, y, w, h = bbox
                    cx, cy = current_centroids[blob_idx]
                    
                    new_player = [self.next_id, cx, cy, x, y, w, h, [(cx, cy)]]
                    self.player_list.append(new_player)
                    self.next_id += 1
    
    def track(self, frame):
        """
        Main tracking function.
        
        Returns: player_list, foreground_mask
        """
        detected_blobs, fg_mask = self.detect_blobs(frame)
        self.update_tracks(detected_blobs)
        return self.player_list, fg_mask


def draw_players(frame, player_list, corner_tl, corner_tr, corner_bl, corner_br):
    """
    Draw player bounding boxes, IDs, and motion trails on frame.
    
    Args:
        frame: Image to draw on
        player_list: List of players from tracker
        corner_tl, corner_tr, corner_bl, corner_br: Court corners (optional, for drawing court)
    
    Returns: Frame with visualizations drawn
    """
    display = frame.copy()
    
    # Draw court boundaries
    if corner_tl and corner_tr and corner_bl and corner_br:
        pts = np.array([corner_tl, corner_tr, corner_br, corner_bl], dtype=np.int32)
        cv2.polylines(display, [pts], True, (0, 255, 255), 2)
    
    # Define colors for different players
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # Draw each player
    for idx, player in enumerate(player_list):
        player_id, cx, cy, x, y, w, h, history = player
        color = colors[idx % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        
        # Draw centroid
        cv2.circle(display, (cx, cy), 5, color, -1)
        
        # Draw ID label
        cv2.putText(display, f"P{player_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw motion trail (last 30 positions)
        for i in range(1, len(history)):
            cv2.line(display, history[i - 1], history[i], color, 2)
    
    return display


# =============================================================================
# STAGE 5: REAL-TIME VISUALIZATION WITH XY PLOT
# =============================================================================

def process_video(video_path, crop_region, expand_pixels=50, vertical_padding_ratio=0.5, 
                  output_path='output_tracking.mp4', show_plot=True):
    """
    Complete pipeline: Detect court, track players, warp to bird's eye, visualize in real-time.
    
    Pipeline:
    1. Read video frame-by-frame
    2. Detect court corners (once, from first frame)
    3. Create court mask for tracking
    4. Track players using background subtraction
    5. Warp frame to bird's eye view
    6. Display warped view + player positions XY plot
    
    Args:
        video_path: Path to input video
        crop_region: (top, bottom, left, right) pixel bounds to crop before corner detection
        expand_pixels: Pixels to expand corners for out-of-bounds tracking
        vertical_padding_ratio: Extra vertical space in warped view (0.5 = 50% padding)
        output_path: Where to save output video (optional)
        show_plot: Whether to show real-time XY position plot
    """
    print(f"[1/6] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return
    
    # Read first frame for court detection
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Could not read first frame")
        return
    
    print("[2/6] Detecting court corners on first frame...")
    # Crop region for corner detection
    top, bottom, left, right = crop_region
    cropped = first_frame[top:bottom, left:right]
    
    # Detect corners in cropped region
    corner_tl, corner_tr, corner_bl, corner_br = detect_court_corners(cropped)
    
    if not all([corner_tl, corner_tr, corner_bl, corner_br]):
        print("ERROR: Could not detect all 4 court corners")
        return
    
    print(f"   Corners detected (cropped): TL={corner_tl}, TR={corner_tr}, BL={corner_bl}, BR={corner_br}")
    
    # Map corners to full image
    corner_tl, corner_tr, corner_bl, corner_br = map_corners_to_full_image(
        corner_tl, corner_tr, corner_bl, corner_br, crop_region
    )
    
    print(f"   Corners mapped (full image): TL={corner_tl}, TR={corner_tr}, BL={corner_bl}, BR={corner_br}")
    
    print("[3/6] Creating court mask for player tracking...")
    # Create mask for focusing tracking on court area
    court_mask = create_court_mask(first_frame.shape, corner_tl, corner_tr, corner_bl, corner_br, 
                                    buffer_pixels=150)
    
    print("[4/6] Initializing player tracker...")
    # Initialize tracker with tuned parameters for better far-end detection
    tracker = PlayerTracker(history=500, var_threshold=25)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[5/6] Processing {total_frames} frames at {fps} FPS...")
    
    # Setup matplotlib for real-time XY plot
    if show_plot:
        plt.ion()  # Interactive mode
        fig, (ax_frame, ax_plot) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Tennis Player Tracking', fontsize=16)
        
        # Calculate court dimensions for plot limits
        top_width = np.sqrt((corner_tr[0] - corner_tl[0])**2 + (corner_tr[1] - corner_tl[1])**2)
        bottom_width = np.sqrt((corner_br[0] - corner_bl[0])**2 + (corner_br[1] - corner_bl[1])**2)
        avg_width = (top_width + bottom_width) / 2
        base_height = int(avg_width * 1.2)
        vertical_padding = int(base_height * vertical_padding_ratio)
        total_height = base_height + vertical_padding
        
        ax_plot.set_xlim(0, avg_width)
        ax_plot.set_ylim(total_height, 0)  # Flip Y axis (image coordinates)
        ax_plot.set_xlabel('Court Width (pixels)')
        ax_plot.set_ylabel('Court Length (pixels)')
        ax_plot.set_title('Player Positions (Bird\'s Eye View)')
        ax_plot.grid(True, alpha=0.3)
        ax_plot.set_aspect('equal')
        
        # Store plot elements
        player_scatters = []
        player_trails = []
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    
    print("[6/6] Starting main processing loop...")
    print("   Press 'q' to quit, 'p' to pause")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Apply court mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=court_mask)
        
        # Track players in masked frame
        player_list, fg_mask = tracker.track(masked_frame)
        
        # Draw players on original frame
        display_frame = draw_players(frame, player_list, corner_tl, corner_tr, corner_bl, corner_br)
        
        # Warp to bird's eye view
        warped, warp_matrix = warp_to_birds_eye(
            display_frame, corner_tl, corner_tr, corner_bl, corner_br,
            expand_pixels=expand_pixels,
            vertical_padding_ratio=vertical_padding_ratio
        )
        
        # Update visualization
        if show_plot:
            # Clear previous plot
            ax_frame.clear()
            ax_plot.clear()
            
            # Show warped frame
            ax_frame.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            ax_frame.set_title(f'Warped View (Frame {frame_count}/{total_frames})')
            ax_frame.axis('off')
            
            # Plot setup
            ax_plot.set_xlim(0, avg_width)
            ax_plot.set_ylim(total_height, 0)
            ax_plot.set_xlabel('Court Width (pixels)')
            ax_plot.set_ylabel('Court Length (pixels)')
            ax_plot.set_title('Player Positions (Bird\'s Eye View)')
            ax_plot.grid(True, alpha=0.3)
            ax_plot.set_aspect('equal')
            
            # Plot each player's position and trail
            colors = ['blue', 'green', 'red', 'yellow']
            for idx, player in enumerate(player_list):
                player_id, cx, cy, x, y, w, h, history = player
                color = colors[idx % len(colors)]
                
                # Transform centroid to warped coordinates
                point = np.array([[[cx, cy]]], dtype=np.float32)
                warped_point = cv2.perspectiveTransform(point, warp_matrix)
                wx, wy = warped_point[0][0]
                
                # Plot current position
                ax_plot.scatter(wx, wy, c=color, s=200, marker='o', 
                               edgecolors='white', linewidths=2, label=f'Player {player_id}')
                
                # Plot trail
                if len(history) > 1:
                    trail_x, trail_y = [], []
                    for hist_pt in history:
                        hist_point = np.array([[[hist_pt[0], hist_pt[1]]]], dtype=np.float32)
                        warped_hist = cv2.perspectiveTransform(hist_point, warp_matrix)
                        trail_x.append(warped_hist[0][0][0])
                        trail_y.append(warped_hist[0][0][1])
                    
                    ax_plot.plot(trail_x, trail_y, c=color, alpha=0.6, linewidth=2)
            
            if player_list:
                ax_plot.legend(loc='upper right')
            
            plt.pause(0.001)
        else:
            # Show warped view in OpenCV window
            cv2.imshow('Warped View', warped)
            cv2.imshow('Foreground Mask', fg_mask)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('p'):
            print("\nPaused. Press any key to continue...")
            cv2.waitKey(0)
        
        # Progress indicator
        if frame_count % 30 == 0:
            print(f"   Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%) - {len(player_list)} players tracked")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if show_plot:
        plt.close('all')
    
    print(f"\n✓ Processing complete! Processed {frame_count} frames.")


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