import numpy as np
import cv2
from ultralytics import YOLO


class PlayerTrackerYOLO:
    # YOLO-based player tracker with feet position tracking

    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3):
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.debug = True

    def detect_players(self, frame):
        # Detect players using YOLO
        results = self.model(frame, conf=self.conf_threshold, classes=[0], verbose=False)
        boxes = results[0].boxes

        detections = []
        if len(boxes) > 0:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(boxes.conf[i])

                feet_x = (x1 + x2) // 2
                feet_y = y2

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'feet': (feet_x, feet_y),
                    'conf': conf
                })

        return detections

    def assign_player_ids(self, detections):
        # Assign IDs based on position
        if len(detections) == 0:
            return None, None

        detections_sorted = sorted(detections, key=lambda d: d['conf'], reverse=True)
        top_candidates = detections_sorted[:2]

        if len(top_candidates) == 1:
            return top_candidates[0], None

        by_position = sorted(top_candidates, key=lambda d: d['feet'][1])
        return by_position[0], by_position[1]

    def track(self, frame, warp_matrix=None):
        # Main tracking function
        detections = self.detect_players(frame)
        player1, player2 = self.assign_player_ids(detections)

        p1_x = p1_y = p2_x = p2_y = None
        player1_bbox = player2_bbox = None

        if player1:
            p1_x, p1_y = player1['feet']
            player1_bbox = player1['bbox']

        if player2:
            p2_x, p2_y = player2['feet']
            player2_bbox = player2['bbox']

        return player1_bbox, player2_bbox, (p1_x, p1_y), (p2_x, p2_y)
