
import cv2
import numpy as np

def detect_court_corners(image):
    # Detect court corners using Canny edge detection and quadrant search
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Divide into 4 quadrants
    h, w = edges.shape
    mid_h, mid_w = h // 2, w // 2
    
    quad_top_left = edges[0:mid_h, 0:mid_w]
    quad_top_right = edges[0:mid_h, mid_w:w]
    quad_bottom_left = edges[mid_h:h, 0:mid_w]
    quad_bottom_right = edges[mid_h:h, mid_w:w]
    
    # Find corner in each quadrant
    corner_tl = find_corner_in_quadrant(quad_top_left, 'top-left')
    corner_tr = find_corner_in_quadrant(quad_top_right, 'top-right')
    corner_bl = find_corner_in_quadrant(quad_bottom_left, 'bottom-left')
    corner_br = find_corner_in_quadrant(quad_bottom_right, 'bottom-right')
    
    # Adjust coordinates to full image
    if corner_tr:
        corner_tr = (corner_tr[0] + mid_w, corner_tr[1])
    if corner_bl:
        corner_bl = (corner_bl[0], corner_bl[1] + mid_h)
    if corner_br:
        corner_br = (corner_br[0] + mid_w, corner_br[1] + mid_h)
    
    return corner_tl, corner_tr, corner_bl, corner_br


def find_corner_in_quadrant(quad_edges, corner_type):
    # Find best corner candidate in quadrant using position-based scoring
    h, w = quad_edges.shape
    
    if corner_type == 'top-left':
        search_h = int(h * 0.3)
        search_w_start, search_w_end = int(w * 0.2), int(w * 0.8)
        region = quad_edges[0:search_h, search_w_start:search_w_end]
        offset_x, offset_y = search_w_start, 0
    elif corner_type == 'top-right':
        search_h = int(h * 0.3)
        search_w_start, search_w_end = int(w * 0.2), int(w * 0.8)
        region = quad_edges[0:search_h, search_w_start:search_w_end]
        offset_x, offset_y = search_w_start, 0
    elif corner_type == 'bottom-left':
        search_h, search_w = int(h * 0.4), int(w * 0.4)
        region = quad_edges[-search_h:, 0:search_w]
        offset_x, offset_y = 0, h - search_h
    elif corner_type == 'bottom-right':
        search_h, search_w = int(h * 0.4), int(w * 0.4)
        region = quad_edges[-search_h:, -search_w:]
        offset_x, offset_y = w - search_w, h - search_h
    
    # Find edge pixels
    y_coords, x_coords = np.where(region > 0)
    if len(x_coords) == 0:
        return None
    
    # Score pixels based on corner position
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
    corner_x = x_coords[best_idx] + offset_x
    corner_y = y_coords[best_idx] + offset_y
    
    return (corner_x, corner_y)


def map_corners_to_full_image(corner_tl, corner_tr, corner_bl, corner_br, crop_region):
    # Transform corner coordinates from crop to full image space
    y1, y2, x1, x2 = crop_region
    
    full_tl = (corner_tl[0] + x1, corner_tl[1] + y1)
    full_tr = (corner_tr[0] + x1, corner_tr[1] + y1)
    full_bl = (corner_bl[0] + x1, corner_bl[1] + y1)
    full_br = (corner_br[0] + x1, corner_br[1] + y1)
    
    return full_tl, full_tr, full_bl, full_br


def expand_corners(corner_tl, corner_tr, corner_bl, corner_br, pixels=50):
    # Inflate court boundaries outward by specified pixels
    exp_tl = (corner_tl[0] - pixels, corner_tl[1] - pixels)
    exp_tr = (corner_tr[0] + pixels, corner_tr[1] - pixels)
    exp_bl = (corner_bl[0] - pixels, corner_bl[1] + pixels)
    exp_br = (corner_br[0] + pixels, corner_br[1] + pixels)
    
    return exp_tl, exp_tr, exp_bl, exp_br


def create_court_mask(image_shape, corner_tl, corner_tr, corner_bl, corner_br, buffer_pixels=150):
    # Generate binary mask for court region with buffer zone
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    exp_tl, exp_tr, exp_bl, exp_br = expand_corners(
        corner_tl, corner_tr, corner_bl, corner_br, buffer_pixels
    )
    
    pts = np.array([exp_tl, exp_tr, exp_br, exp_bl], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    return mask


def warp_to_birds_eye(image, corner_tl, corner_tr, corner_bl, corner_br,
                       expand_pixels=50, vertical_padding_ratio=0.4):
    # Apply perspective transformation to bird's eye view with padding
    if expand_pixels > 0:
        corner_tl, corner_tr, corner_bl, corner_br = expand_corners(
            corner_tl, corner_tr, corner_bl, corner_br, expand_pixels
        )
    
    src_points = np.array([corner_tl, corner_tr, corner_br, corner_bl], dtype=np.float32)
    
    # Calculate output dimensions
    top_width = np.sqrt((corner_tr[0] - corner_tl[0])**2 + (corner_tr[1] - corner_tl[1])**2)
    bottom_width = np.sqrt((corner_br[0] - corner_bl[0])**2 + (corner_br[1] - corner_bl[1])**2)
    avg_width = (top_width + bottom_width) / 2
    
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
    
    return warped, matrix