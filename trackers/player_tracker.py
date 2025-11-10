import numpy as np
import cv2


class PlayerTracker:
    # MOG2-based player tracker with Kalman filtering

    def __init__(self, history=500, var_threshold=50, max_players=2):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True
        )

        self.player_list = []
        self.next_id = 0
        self.max_players = max_players
        self.max_frames_lost = 15
        self.bbox_alpha = 0.5
        self.debug = True

    def get_centroid(self, x, y, w, h):
        # Calculate bottom-center of bounding box
        return int(x + w / 2), int(y + h)

    def create_kalman_filter(self):
        # Create Kalman filter
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.3
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 8
        return kf

    def calculate_iou(self, bbox1, bbox2):
        # Calculate Intersection over Union
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / (union + 1e-6)

    def smooth_bbox(self, old_bbox, new_bbox, alpha):
        # Exponential moving average smoothing
        if old_bbox is None:
            return new_bbox

        smooth_x = alpha * new_bbox[0] + (1 - alpha) * old_bbox[0]
        smooth_y = alpha * new_bbox[1] + (1 - alpha) * old_bbox[1]
        smooth_w = alpha * new_bbox[2] + (1 - alpha) * old_bbox[2]
        smooth_h = alpha * new_bbox[3] + (1 - alpha) * old_bbox[3]

        return (int(smooth_x), int(smooth_y), int(smooth_w), int(smooth_h))

    def detect_blobs(self, frame):
        # Apply background subtraction and morphological operations
        fg_mask_raw = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask_raw, 200, 255, cv2.THRESH_BINARY)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel_medium, iterations=3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if 750 < area < 30000:
                x, y, w, h = cv2.boundingRect(c)
                aspect = h / w if w > 0 else 0
                if 0.75 < aspect < 10.0:
                    detected_blobs.append((x, y, w, h))

        if self.debug:
            vis_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            for x, y, w, h in detected_blobs:
                cv2.rectangle(vis_mask, (x, y), (x+w, y+h), (255, 0, 0), 2)

            stats = f"Detected: {len(detected_blobs)} | Tracked: {len(self.player_list)}"
            cv2.putText(vis_mask, stats, (10, vis_mask.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Player Detection', vis_mask)

        return detected_blobs, fg_mask

    def update_tracks(self, detected_blobs):
        # Match detections to existing tracks
        current_centroids = [self.get_centroid(*bbox) for bbox in detected_blobs]

        if len(self.player_list) == 0:
            for i, bbox in enumerate(detected_blobs[:self.max_players]):
                x, y, w, h = bbox
                cx, cy = current_centroids[i]
                kf = self.create_kalman_filter()
                kf.statePost = np.array([[np.float32(cx)], [np.float32(cy)],
                                        [np.float32(0)], [np.float32(0)]], dtype=np.float32)
                new_player = [self.next_id, cx, cy, x, y, w, h, [(cx, cy)], kf, (x, y, w, h), 0]
                self.player_list.append(new_player)
                self.next_id += 1
            return

        used_blobs = set()

        for player in self.player_list:
            _, _, _, old_x, old_y, old_w, old_h, _, kf, old_smoothed, _ = player
            old_bbox = (old_x, old_y, old_w, old_h)

            predicted = kf.predict()
            pred_cx, pred_cy = int(predicted[0]), int(predicted[1])

            best_score = -1
            best_blob_idx = -1

            for blob_idx, bbox in enumerate(detected_blobs):
                if blob_idx in used_blobs:
                    continue

                x, y, w, h = bbox
                new_cx, new_cy = current_centroids[blob_idx]

                centroid_dist = np.sqrt((pred_cx - new_cx)**2 + (pred_cy - new_cy)**2)
                iou = self.calculate_iou(old_bbox, bbox)

                distance_score = max(0, 150 - centroid_dist)
                iou_score = iou * 300
                combined_score = iou_score + distance_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_blob_idx = blob_idx

            if best_blob_idx != -1 and best_score > 50:
                x, y, w, h = detected_blobs[best_blob_idx]
                new_cx, new_cy = current_centroids[best_blob_idx]

                measured = np.array([[np.float32(new_cx)], [np.float32(new_cy)]], dtype=np.float32)
                kf.correct(measured)
                corrected = kf.statePost
                smooth_cx, smooth_cy = int(corrected[0]), int(corrected[1])

                smoothed_bbox = self.smooth_bbox(old_smoothed, (x, y, w, h), self.bbox_alpha)
                smooth_x, smooth_y, smooth_w, smooth_h = smoothed_bbox

                player[1], player[2] = smooth_cx, smooth_cy
                player[3], player[4], player[5], player[6] = smooth_x, smooth_y, smooth_w, smooth_h
                player[7].append((smooth_cx, smooth_cy))
                player[9] = smoothed_bbox
                player[10] = 0

                if len(player[7]) > 30:
                    player[7].pop(0)

                used_blobs.add(best_blob_idx)
            else:
                player[10] += 1

        self.player_list = [p for p in self.player_list if p[10] < self.max_frames_lost]

        if len(self.player_list) < self.max_players:
            for blob_idx, bbox in enumerate(detected_blobs):
                if blob_idx not in used_blobs and len(self.player_list) < self.max_players:
                    x, y, w, h = bbox
                    cx, cy = current_centroids[blob_idx]
                    kf = self.create_kalman_filter()
                    kf.statePost = np.array([[np.float32(cx)], [np.float32(cy)],
                                            [np.float32(0)], [np.float32(0)]], dtype=np.float32)
                    new_player = [self.next_id, cx, cy, x, y, w, h, [(cx, cy)], kf, (x, y, w, h), 0]
                    self.player_list.append(new_player)
                    self.next_id += 1

    def track(self, frame, warp_matrix=None):
        # Main tracking function
        detected_blobs, fg_mask = self.detect_blobs(frame)
        self.update_tracks(detected_blobs)

        if len(self.player_list) > 0:
            sorted_players = sorted(self.player_list, key=lambda p: p[2])
            for i, player in enumerate(sorted_players):
                player[0] = i

        return self.player_list, fg_mask
