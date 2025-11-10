import numpy as np
import cv2


class BallTracker:
    # Simple ball tracker: Find right-sized blob, use optical flow as backup
    # No physics, no predictions - just robust blob + motion detection
    
    def __init__(self):
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,
            detectShadows=True
        )
        
        # Detection parameters - based on observations
        # Ball: ~15x15, area ~225
        self.min_area = 50
        self.max_area = 500
        
        # Optical flow backup
        self.prev_gray = None
        
        # Debug visualization
        self.debug = True
    
    # No sharpening needed for best MOG2+3x3x2
    
    def detect_ball_blobs(self, frame, player_list=None):
        # No sharpening, just use raw frame
        fg_mask_raw = self.bg_subtractor.apply(frame)
        # Remove shadows and adaptive threshold for better blob separation
        _, fg_mask = cv2.threshold(fg_mask_raw, 200, 255, cv2.THRESH_BINARY)
        # 3x3 kernel, 2 open iterations (no close)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        fg_mask_open = fg_mask_clean.copy()
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Build player exclusion regions
        player_regions = []
        if player_list is not None:
            for player in player_list:
                # player format: [id, cx, cy, x, y, w, h, history]
                _, _, _, px, py, pw, ph, _ = player
                # Bigger margin to exclude player motion area
                margin = 20
                player_regions.append({
                    'x1': max(0, px - margin),
                    'y1': max(0, py - margin),
                    'x2': px + pw + margin,
                    'y2': py + ph + margin
                })
        
        # Filter for small circular objects (ball, not players)
        detected_blobs = []
        
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            
            # Calculate metrics
            aspect = h / w if w > 0 else 0
            
            # Check if blob overlaps with any player region
            cx = x + w // 2
            cy = y + h // 2
            in_player_region = False
            
            for region in player_regions:
                if (region['x1'] <= cx <= region['x2'] and 
                    region['y1'] <= cy <= region['y2']):
                    in_player_region = True
                    break
            
            # Skip if in player region
            if in_player_region:
                continue
            
            # Filter criteria - tighter for ball consistency
            # Ball: aspect ~1.0 (circular), area 150-300
            # Players: aspect 1.3-2.0, area 1800+ (28x65 minimum)
            size_ok = self.min_area < area < self.max_area
            aspect_ok = 0.5 < aspect < 2.0  # Ball should be roughly circular
            
            if size_ok and aspect_ok:
                detected_blobs.append((x, y, w, h, area, aspect))
        
        # Sort by quality: prefer circular blobs (aspect ~1.0) with area ~225
        # Score = abs(aspect - 1.0) + abs(area - 225) / 100
        detected_blobs.sort(key=lambda b: abs(b[5] - 1.0) + abs(b[4] - 225) / 100.0)
        
        # Debug visualization
        if self.debug:
            # Create visualization with original, sharpened, and masks
            h, w = frame.shape[:2]
            
            # Create player mask visualization
            player_mask_vis = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
            for region in player_regions:
                cv2.rectangle(player_mask_vis, 
                            (region['x1'], region['y1']), 
                            (region['x2'], region['y2']), 
                            (0, 0, 255), -1)  # Red exclusion zones
            
            # Convert grayscale masks to BGR for visualization
            vis_raw = cv2.cvtColor(fg_mask_raw, cv2.COLOR_GRAY2BGR)
            vis_thresh = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            vis_open = cv2.cvtColor(fg_mask_open, cv2.COLOR_GRAY2BGR)
            vis_clean = cv2.cvtColor(fg_mask_clean, cv2.COLOR_GRAY2BGR)
            
            # Draw detected blobs on clean mask (GREEN)
            for x, y, w_blob, h_blob, area, aspect_ratio in detected_blobs:
                cv2.rectangle(vis_clean, (x, y), (x+w_blob, y+h_blob), (0, 255, 0), 2)
                # Area on top, size and aspect below
                cv2.putText(vis_clean, f"A:{area:.0f}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                cv2.putText(vis_clean, f"{w_blob}x{h_blob} R:{aspect_ratio:.2f}", (x, y+h_blob+12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            
            # Draw all contours on a separate view (YELLOW)
            vis_contours = vis_clean.copy()
            cv2.drawContours(vis_contours, contours, -1, (0, 255, 255), 1)
            
            # Row 1: Original, Player Exclusion Zones, Sharpened, Raw Mask
            row1 = np.hstack([
                cv2.resize(frame, (w//2, h//2)),
                cv2.resize(player_mask_vis, (w//2, h//2)),
                cv2.resize(vis_raw, (w//2, h//2)),
                cv2.resize(vis_raw, (w//2, h//2))
            ])
            # Note: Third panel was 'sharpened', now just duplicate raw mask for layout
            
            # Row 2: Threshold, Open, Clean+Detected, Contours
            row2 = np.hstack([
                cv2.resize(vis_thresh, (w//2, h//2)),
                cv2.resize(vis_open, (w//2, h//2)),
                cv2.resize(vis_clean, (w//2, h//2)),
                cv2.resize(vis_contours, (w//2, h//2))
            ])
            
            # Stack rows
            debug_img = np.vstack([row1, row2])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(debug_img, "ORIGINAL", (10, 25), font, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_img, "EXCLUSION ZONES", (w//2+10, 25), font, 0.6, (0, 0, 255), 2)
            cv2.putText(debug_img, "SHARPENED", (w+10, 25), font, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_img, "RAW MASK", (3*w//2+10, 25), font, 0.6, (255, 255, 255), 2)
            
            cv2.putText(debug_img, "THRESHOLD", (10, h//2+25), font, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_img, "MORPH OPEN", (w//2+10, h//2+25), font, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_img, "DETECTED", (w+10, h//2+25), font, 0.6, (0, 255, 0), 2)
            cv2.putText(debug_img, "ALL CONTOURS", (3*w//2+10, h//2+25), font, 0.6, (0, 255, 255), 2)
            
            # Add stats
            stats = f"Blobs Found: {len(detected_blobs)} | Ball Tracked: {'YES' if len(detected_blobs) > 0 else 'NO'}"
            cv2.putText(debug_img, stats, (10, debug_img.shape[0]-10), 
                       font, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Ball Detection Pipeline', debug_img)
        
        return detected_blobs, fg_mask_clean, fg_mask_open
    
    
    def detect_with_optical_flow(self, frame, morph_open_mask):
        """Use optical flow on high-motion areas to find ball"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return []
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Get magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Threshold for motion
        motion_mask = (mag > 2.0).astype(np.uint8) * 255
        
        # Combine with morphology mask (areas that pass background subtraction)
        combined = cv2.bitwise_and(motion_mask, morph_open_mask)
        
        # Find contours in combined
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        flow_blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(c)
                aspect = h / w if w > 0 else 0
                if 0.5 < aspect < 2.0:
                    flow_blobs.append((x, y, w, h, area, aspect))
        
        self.prev_gray = gray
        return flow_blobs
    
    def track(self, frame, player_list=None):
        """Simple tracking: find best blob, return single ball. No history."""
        detected_blobs, mask_clean, mask_open = self.detect_ball_blobs(frame, player_list)
        
        # Try optical flow as backup
        flow_blobs = self.detect_with_optical_flow(frame, mask_open)
        
        # Combine both methods - prefer blob detection, use flow as backup
        all_candidates = detected_blobs if len(detected_blobs) > 0 else flow_blobs
        
        if len(all_candidates) > 0:
            # Take ONLY the best blob (first after sorting)
            x, y, w, h, area, aspect = all_candidates[0]
            cx = x + w // 2
            cy = y + h // 2
            
            # Return single ball - NO HISTORY
            # Format: [id, cx, cy, x, y, w, h, empty_history]
            return [[0, cx, cy, x, y, w, h, []]], mask_clean
        
        # No detection - return empty
        return [], mask_clean


