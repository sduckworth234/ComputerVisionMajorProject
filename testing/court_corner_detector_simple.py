"""
Tennis Court Corner Detection & Player Tracking System

PIPELINE:
1. Detect court corners using edge detection and quadrant search
2. Map corners from cropped to full image coordinates  
3. Create court mask with buffer zone
4. Track players using background subtraction
5. Warp image to bird's eye view
6. Visualize player positions in real-time + plot tracking

Pure computer vision - no machine learning required!
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


# =============================================================================
# STAGE 1: COURT CORNER DETECTION
# =============================================================================

def detect_court_corners(image):
    """
    Detect 4 tennis court corners using perspective-aware quadrant search.
    
    Returns: 4 corner points as (top_left, top_right, bottom_left, bottom_right)
    """
    # Step 1: Convert to grayscale and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Detect edges
    
    # Step 2: Divide image into 4 quadrants (one corner per quadrant)
    h, w = edges.shape
    mid_h, mid_w = h // 2, w // 2
    
    # Extract quadrants
    quad_top_left = edges[0:mid_h, 0:mid_w]
    quad_top_right = edges[0:mid_h, mid_w:w]
    quad_bottom_left = edges[mid_h:h, 0:mid_w]
    quad_bottom_right = edges[mid_h:h, mid_w:w]
    
    # Step 3: Find corner in each quadrant
    corner_tl = find_corner_in_quadrant(quad_top_left, 'top-left')
    corner_tr = find_corner_in_quadrant(quad_top_right, 'top-right')
    corner_bl = find_corner_in_quadrant(quad_bottom_left, 'bottom-left')
    corner_br = find_corner_in_quadrant(quad_bottom_right, 'bottom-right')
    
    # Step 4: Adjust coordinates to full image
    if corner_tr: corner_tr = (corner_tr[0] + mid_w, corner_tr[1])
    if corner_bl: corner_bl = (corner_bl[0], corner_bl[1] + mid_h)
    if corner_br: corner_br = (corner_br[0] + mid_w, corner_br[1] + mid_h)
    
    return corner_tl, corner_tr, corner_bl, corner_br


def find_corner_in_quadrant(quad_edges, corner_type):
    """
    Find a specific corner within its quadrant.
    Uses perspective-aware search regions (far corners appear smaller).
    """
    h, w = quad_edges.shape
    
    # Define search region based on perspective
    if corner_type == 'top-left':
        # Far corner: search top 30%, middle 60% horizontally
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
        # Near corner: search bottom 40%, left 40%
        search_h, search_w = int(h * 0.4), int(w * 0.4)
        region = quad_edges[-search_h:, 0:search_w]
        offset_x, offset_y = 0, h - search_h
        
    elif corner_type == 'bottom-right':
        # Near corner: search bottom 40%, right 40%
        search_h, search_w = int(h * 0.4), int(w * 0.4)
        region = quad_edges[-search_h:, -search_w:]
        offset_x, offset_y = w - search_w, h - search_h
    
    # Find all edge pixels in search region
    y_coords, x_coords = np.where(region > 0)
    if len(x_coords) == 0:
        return None
    
    # Score pixels based on position (want corners at extremes)
    reg_h, reg_w = region.shape
    if 'top-left' in corner_type:
        # Want top-left: high y score (top), high x score (left)
        scores = (reg_h - y_coords) * 3 + (reg_w - x_coords) * 1
    elif 'top-right' in corner_type:
        # Want top-right: high y score (top), low x score (right)
        scores = (reg_h - y_coords) * 3 + x_coords * 1
    elif 'bottom-left' in corner_type:
        # Want bottom-left: low y score (bottom), high x score (left)
        scores = y_coords * 2 + (reg_w - x_coords) * 2
    elif 'bottom-right' in corner_type:
        # Want bottom-right: low y score (bottom), low x score (right)
        scores = y_coords * 2 + x_coords * 2
    
    # Return pixel with best score
    best_idx = np.argmax(scores)
    return (x_coords[best_idx] + offset_x, y_coords[best_idx] + offset_y)


# =============================================================================
# STAGE 2: COORDINATE MAPPING & EXPANSION
# =============================================================================

def expand_corners(corner_tl, corner_tr, corner_bl, corner_br, pixels=50):
    """
    Expand court corners outward to capture out-of-bounds area.
    Each corner moves away from the court center.
    """
    # Expand each corner outward
    exp_tl = (corner_tl[0] - pixels, corner_tl[1] - pixels)  # Left + up
    exp_tr = (corner_tr[0] + pixels, corner_tr[1] - pixels)  # Right + up
    exp_bl = (corner_bl[0] - pixels, corner_bl[1] + pixels)  # Left + down
    exp_br = (corner_br[0] + pixels, corner_br[1] + pixels)  # Right + down
    
    return exp_tl, exp_tr, exp_bl, exp_br


def map_corners_to_full_image(corner_tl, corner_tr, corner_bl, corner_br, crop_region):
    """
    Map corners from cropped image coordinates to full image coordinates.
    Adds the crop offset to each corner.
    """
    y1, y2, x1, x2 = crop_region
    
    # Add crop offset to each corner
    full_tl = (corner_tl[0] + x1, corner_tl[1] + y1)
    full_tr = (corner_tr[0] + x1, corner_tr[1] + y1)
    full_bl = (corner_bl[0] + x1, corner_bl[1] + y1)
    full_br = (corner_br[0] + x1, corner_br[1] + y1)
    
    return full_tl, full_tr, full_bl, full_br


# =============================================================================
# STAGE 3: PERSPECTIVE WARP TO BIRD'S EYE VIEW
# =============================================================================

def warp_to_birds_eye(image, corner_tl, corner_tr, corner_bl, corner_br, 
                      expand_pixels=50, vertical_padding_ratio=0.4):
    """
    Warp oblique court view to perfect bird's eye rectangle.
    Uses perspective transformation (homography).
    """
    # Step 1: Expand corners if requested
    if expand_pixels > 0:
        corner_tl, corner_tr, corner_bl, corner_br = expand_corners(
            corner_tl, corner_tr, corner_bl, corner_br, expand_pixels
        )
    
    # Step 2: Define source points (oblique quadrilateral)
    src_points = np.array([corner_tl, corner_tr, corner_br, corner_bl], dtype=np.float32)
    
    # Step 3: Calculate output dimensions
    # Measure top and bottom baseline widths
    top_width = np.linalg.norm(np.array(corner_tr) - np.array(corner_tl))
    bottom_width = np.linalg.norm(np.array(corner_br) - np.array(corner_bl))
    avg_width = (top_width + bottom_width) / 2
    
    output_width = int(avg_width)
    base_height = int(output_width * 1.2)  # Slightly taller than wide
    
    # Step 4: Add vertical padding for out-of-bounds coverage
    vertical_padding = int(base_height * vertical_padding_ratio)
    output_height = base_height + vertical_padding
    top_pad = vertical_padding // 2
    
    # Step 5: Define destination points (perfect rectangle)
    dst_points = np.array([
        [0, top_pad],  # Top-left
        [output_width, top_pad],  # Top-right
        [output_width, top_pad + base_height],  # Bottom-right
        [0, top_pad + base_height]  # Bottom-left
    ], dtype=np.float32)
    
    # Step 6: Calculate perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Step 7: Apply transformation
    warped = cv2.warpPerspective(image, matrix, (output_width, output_height))
    
    return warped, matrix


# =============================================================================
# STAGE 4: COURT MASK FOR FOCUSED TRACKING
# =============================================================================

def create_court_mask(image_shape, corner_tl, corner_tr, corner_bl, corner_br, buffer_pixels=150):
    """
    Create binary mask of court area + buffer.
    Only track players inside this region (ignore audience, etc).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Expand corners to include buffer zone
    exp_tl, exp_tr, exp_bl, exp_br = expand_corners(
        corner_tl, corner_tr, corner_bl, corner_br, buffer_pixels
    )
    
    # Create polygon and fill it
    pts = np.array([exp_tl, exp_tr, exp_br, exp_bl], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    return mask


# =============================================================================
# STAGE 5: PLAYER TRACKING (BACKGROUND SUBTRACTION)
# =============================================================================

class PlayerTracker:
    """
    Track tennis players using background subtraction and centroid matching.
    No machine learning - just motion detection and geometry.
    """
    
    def __init__(self, history=500, var_threshold=25):
        # Background subtractor (learns background over time)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,  # Number of frames to learn background
            varThreshold=var_threshold,  # Sensitivity (lower = more sensitive)
            detectShadows=True  # Detect and remove shadows
        )
        
        # Tracking data
        self.player_positions = []  # List of (x, y) centroids
        self.player_bboxes = []  # List of (x, y, w, h) bounding boxes
        self.player_histories = []  # List of position histories
        self.max_distance = 150  # Max pixels for matching player between frames
        
    def track(self, frame):
        """
        Main tracking function - detects and tracks players.
        Returns: player_positions, player_bboxes, player_histories, foreground_mask
        """
        # Step 1: Background subtraction - find moving objects
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Step 2: Remove shadows (MOG2 marks shadows as gray ~127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Step 3: Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        
        # Step 4: Find contours (blobs)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 5: Filter contours to find players
        detected_bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust these if tracking fails)
            if 300 < area < 30000:  # Lowered min area for far player
                x, y, w, h = cv2.boundingRect(contour)
                aspect = h / w if w > 0 else 0
                
                # Filter by aspect ratio (players are taller than wide)
                if 1.0 < aspect < 5.0:  # More lenient for perspective
                    detected_bboxes.append((x, y, w, h))
        
        # Step 6: Update player tracks
        self._update_tracks(detected_bboxes)
        
        return self.player_positions, self.player_bboxes, self.player_histories, fg_mask
    
    def _update_tracks(self, detected_bboxes):
        """
        Match detected bboxes to existing players using nearest neighbor.
        """
        # Calculate centroids of new detections
        new_centroids = []
        for bbox in detected_bboxes:
            x, y, w, h = bbox
            centroid = (int(x + w/2), int(y + h/2))
            new_centroids.append(centroid)
        
        # If first frame, initialize all players
        if len(self.player_positions) == 0:
            for i in range(len(detected_bboxes)):
                self.player_positions.append(new_centroids[i])
                self.player_bboxes.append(detected_bboxes[i])
                self.player_histories.append([new_centroids[i]])
            return
        
        # Match new detections to existing players
        used_detections = set()
        matched_players = set()
        
        for player_idx, old_pos in enumerate(self.player_positions):
            # Find nearest new detection
            min_dist = float('inf')
            best_det_idx = -1
            
            for det_idx, new_pos in enumerate(new_centroids):
                if det_idx in used_detections:
                    continue
                
                # Calculate distance
                dist = np.sqrt((old_pos[0] - new_pos[0])**2 + (old_pos[1] - new_pos[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_det_idx = det_idx
            
            # Update if match found within threshold
            if best_det_idx != -1 and min_dist < self.max_distance:
                self.player_positions[player_idx] = new_centroids[best_det_idx]
                self.player_bboxes[player_idx] = detected_bboxes[best_det_idx]
                self.player_histories[player_idx].append(new_centroids[best_det_idx])
                
                # Keep history limited to last 30 positions
                if len(self.player_histories[player_idx]) > 30:
                    self.player_histories[player_idx] = self.player_histories[player_idx][-30:]
                
                used_detections.add(best_det_idx)
                matched_players.add(player_idx)
        
        # Remove unmatched players (lost track)
        for player_idx in sorted(range(len(self.player_positions)), reverse=True):
            if player_idx not in matched_players:
                del self.player_positions[player_idx]
                del self.player_bboxes[player_idx]
                del self.player_histories[player_idx]
        
        # Add new players (new detections that weren't matched)
        for det_idx in range(len(detected_bboxes)):
            if det_idx not in used_detections:
                self.player_positions.append(new_centroids[det_idx])
                self.player_bboxes.append(detected_bboxes[det_idx])
                self.player_histories.append([new_centroids[det_idx]])


# =============================================================================
# STAGE 6: VISUALIZATION
# =============================================================================

def draw_players(frame, player_positions, player_bboxes, player_histories, 
                 corner_tl=None, corner_tr=None, corner_bl=None, corner_br=None):
    """
    Draw player bounding boxes, centroids, IDs, and trails on frame.
    """
    display = frame.copy()
    
    # Draw court boundaries if available
    if corner_tl and corner_tr and corner_bl and corner_br:
        pts = np.array([corner_tl, corner_tr, corner_br, corner_bl], dtype=np.int32)
        cv2.polylines(display, [pts], True, (0, 255, 255), 2)
    
    # Colors for different players
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # Draw each player
    for idx in range(len(player_positions)):
        x, y, w, h = player_bboxes[idx]
        centroid = player_positions[idx]
        color = colors[idx % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        
        # Draw centroid
        cv2.circle(display, centroid, 5, color, -1)
        
        # Draw ID
        cv2.putText(display, f"P{idx}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw trail (last 30 positions)
        history = player_histories[idx]
        for i in range(1, len(history)):
            cv2.line(display, history[i - 1], history[i], color, 2)
    
    return display


# =============================================================================
# STAGE 7: POSITION PLOT
# =============================================================================

def create_position_plot(player_positions, court_width, court_height, buffer=100):
    """
    Create matplotlib plot showing player positions from above.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    
    # Set plot limits (court + buffer)
    ax.set_xlim(0, court_width)
    ax.set_ylim(court_height + buffer, -buffer)  # Inverted Y (image coordinates)
    
    # Draw court outline
    court_rect = plt.Rectangle((0, 0), court_width, court_height, 
                                fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(court_rect)
    
    # Draw center line
    ax.plot([0, court_width], [court_height/2, court_height/2], 'b--', linewidth=1)
    
    # Plot player positions
    colors = ['red', 'green', 'blue', 'yellow']
    for idx, pos in enumerate(player_positions):
        color = colors[idx % len(colors)]
        circle = Circle(pos, radius=15, color=color, alpha=0.7)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1] - 30, f"P{idx}", ha='center', fontsize=12, 
                color=color, weight='bold')
    
    ax.set_title('Player Positions (Bird\'s Eye View)')
    ax.set_xlabel('Court Width (pixels)')
    ax.set_ylabel('Court Length (pixels)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return fig


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_video(video_path, crop_region=None, expand_pixels=50, 
                  vertical_padding_ratio=0.5, show_plot=True):
    """
    Complete pipeline: detect court, track players, warp to bird's eye view.
    
    Args:
        video_path: Path to tennis video
        crop_region: (y1, y2, x1, x2) to crop for corner detection
        expand_pixels: Pixels to expand court boundaries
        vertical_padding_ratio: Extra vertical space for out-of-bounds
        show_plot: Whether to show position plot
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # STAGE 1: Detect corners from first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read video")
        return
    
    # Crop for corner detection
    if crop_region:
        y1, y2, x1, x2 = crop_region
        cropped_frame = first_frame[y1:y2, x1:x2]
    else:
        cropped_frame = first_frame
        crop_region = (0, first_frame.shape[0], 0, first_frame.shape[1])
    
    # Detect corners
    corner_tl, corner_tr, corner_bl, corner_br = detect_court_corners(cropped_frame)
    
    if not (corner_tl and corner_tr and corner_bl and corner_br):
        print("Error: Could not detect all 4 corners")
        return
    
    # Map to full image coordinates
    corner_tl, corner_tr, corner_bl, corner_br = map_corners_to_full_image(
        corner_tl, corner_tr, corner_bl, corner_br, crop_region
    )
    
    print(f"Detected corners:")
    print(f"  Top-left: {corner_tl}")
    print(f"  Top-right: {corner_tr}")
    print(f"  Bottom-left: {corner_bl}")
    print(f"  Bottom-right: {corner_br}")
    
    # STAGE 2: Create court mask
    court_mask = create_court_mask(first_frame.shape, corner_tl, corner_tr, 
                                    corner_bl, corner_br, buffer_pixels=200)
    
    # STAGE 3: Initialize player tracker
    tracker = PlayerTracker(history=500, var_threshold=25)
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Setup position plot if requested
    if show_plot:
        plt.ion()  # Interactive mode
        fig_plot = None
    
    # STAGE 4: Process video frames
    print("\nProcessing video...")
    print("Press 'q' to quit")
    
    while True:
        ret, full_frame = cap.read()
        if not ret:
            break
        
        # Apply court mask to focus tracking
        masked_frame = cv2.bitwise_and(full_frame, full_frame, mask=court_mask)
        
        # Track players
        player_positions, player_bboxes, player_histories, fg_mask = tracker.track(masked_frame)
        
        # Draw on original frame
        display_frame = draw_players(full_frame, player_positions, player_bboxes, 
                                     player_histories, corner_tl, corner_tr, corner_bl, corner_br)
        
        # Warp to bird's eye view
        warped, warp_matrix = warp_to_birds_eye(full_frame, corner_tl, corner_tr, 
                                                 corner_bl, corner_br, expand_pixels, 
                                                 vertical_padding_ratio)
        
        # Map player positions to warped view
        warped_display = warped.copy()
        for idx, pos in enumerate(player_positions):
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            color = colors[idx % len(colors)]
            
            # Transform position to warped coordinates
            pt = np.array([[[pos[0], pos[1]]]], dtype=np.float32)
            warped_pt = cv2.perspectiveTransform(pt, warp_matrix)
            wx, wy = int(warped_pt[0][0][0]), int(warped_pt[0][0][1])
            
            # Draw on warped view
            cv2.circle(warped_display, (wx, wy), 8, color, -1)
            cv2.putText(warped_display, f"P{idx}", (wx + 10, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Update position plot
        if show_plot and len(player_positions) > 0:
            if fig_plot:
                plt.close(fig_plot)
            
            # Get warped dimensions for plot
            h_warped, w_warped = warped.shape[:2]
            fig_plot = create_position_plot(
                [(pt[0][0][0], pt[0][0][1]) for pt in 
                 [cv2.perspectiveTransform(np.array([[[pos[0], pos[1]]]], dtype=np.float32), warp_matrix) 
                  for pos in player_positions]],
                w_warped, h_warped, buffer=100
            )
            plt.pause(0.001)
        
        # Display windows
        cv2.imshow('Original - Player Tracking', display_frame)
        cv2.imshow('Bird\'s Eye View - Warped', warped_display)
        cv2.imshow('Foreground Mask (Debug)', fg_mask)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if show_plot:
        plt.close('all')
    
    print("\nVideo processing complete!")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Process tennis video
    process_video(
        video_path='data/full_video/tennis_full_video.mp4',
        crop_region=(150, 600, 200, 1100),  # Crop for corner detection
        expand_pixels=50,  # Expand court boundaries
        vertical_padding_ratio=0.5,  # Extra vertical space (50%)
        show_plot=True  # Show position plot
    )
