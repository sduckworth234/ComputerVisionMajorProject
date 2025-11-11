import cv2
import numpy as np

class BallTracker:
    # MOG2-based ball tracker with Kalman filtering

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 2.5
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 6

        self.prev_gray = None
        self.kalman_initialized = False
        self.detection_history = []
        self.max_history = 10
        self.debug = True
        self.court_polygon = None

    def set_court_boundaries(self, inflated_corners):
        # Set court polygon for boundary checking
        self.court_polygon = np.array(inflated_corners, dtype=np.int32)

    def is_inside_court(self, x, y):
        # Check if point is inside court polygon
        if self.court_polygon is None:
            return True
        result = cv2.pointPolygonTest(self.court_polygon, (float(x), float(y)), False)
        return result >= 0

    def track(self, frame, player_list=None, frame_idx=0):
        # Main tracking interface
        position = self.update(frame, player_list)

        if position:
            bx, by = position
            ball_size = 20
            x = bx - ball_size // 2
            y = by - ball_size // 2
            ball_list = [[0, bx, by, x, y, ball_size, ball_size, []]]
        else:
            ball_list = []

        return ball_list, frame

    def detect_ball(self, frame, gray, prev_gray, player_list=None, predicted_pos=None):
        # Detect ball using frame differencing and morphology
        if prev_gray is None:
            return None

        diff = cv2.absdiff(prev_gray, gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_tiny, iterations=2)
        motion_mask = cv2.dilate(motion_mask, kernel_small, iterations=2)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

        ball_mask = motion_mask.copy()
        if player_list:
            for player in player_list:
                x, y, w, h = player[3], player[4], player[5], player[6]
                margin = 30
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(ball_mask.shape[1], x + w + margin), min(ball_mask.shape[0], y + h + margin)
                cv2.rectangle(ball_mask, (x1, y1), (x2, y2), 0, -1)

        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_score = -1

        for c in contours:
            area = cv2.contourArea(c)
            if area < 8 or area > 200:
                continue

            (cx, cy), _ = cv2.minEnclosingCircle(c)
            cx, cy = int(cx), int(cy)
            x, y, w, h = cv2.boundingRect(c)

            aspect = max(w, h) / (min(w, h) + 1e-6)
            if aspect > 3.0:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.4:
                    continue
            else:
                circularity = 0.4

            compactness = area / (w * h + 1e-6)
            if compactness < 0.5:
                continue

            score = circularity * 40 + compactness * 30
            if predicted_pos is not None:
                dist = np.sqrt((cx - predicted_pos[0])**2 + (cy - predicted_pos[1])**2)
                score += max(0, 100 - dist) * 0.3

            if score > best_score:
                best_score = score
                best_candidate = (cx, cy)

        if self.debug:
            debug_vis = cv2.cvtColor(ball_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(debug_vis, "Ball Detection", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("bball Detection Mask", debug_vis)

        return best_candidate

    def update(self, frame, player_list=None):
        # Main tracking loop with Kalman filtering
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        predicted_pos = None
        if self.kalman_initialized:
            predicted = self.kf.predict()
            px, py = int(predicted[0]), int(predicted[1])
            predicted_pos = (px, py)

        detected = self.detect_ball(frame, gray, self.prev_gray, player_list, predicted_pos)

        if detected:
            self.detection_history.append(detected)
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)

            if not self.kalman_initialized and len(self.detection_history) >= 2:
                self.kalman_initialized = True
                self.kf.statePost = np.array([[np.float32(detected[0])],
                                             [np.float32(detected[1])],
                                             [np.float32(0)],
                                             [np.float32(0)]], dtype=np.float32)

            if self.kalman_initialized:
                measured = np.array([[np.float32(detected[0])], [np.float32(detected[1])]], dtype=np.float32)
                corrected = self.kf.correct(measured)
                bx, by = int(corrected[0]), int(corrected[1])
            else:
                bx, by = detected
        else:
            if self.kalman_initialized and predicted_pos and len(self.detection_history) >= 3:
                bx, by = predicted_pos
            else:
                bx, by = None, None
                self.detection_history.clear()
                self.kalman_initialized = False

        self.prev_gray = gray

        if self.debug and (bx is not None):
            vis = frame.copy()

            if player_list:
                for player in player_list:
                    x, y, w, h = player[3], player[4], player[5], player[6]
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if detected:
                cv2.circle(vis, detected, 7, (0, 255, 0), 2)
            if predicted_pos:
                cv2.circle(vis, predicted_pos, 5, (255, 0, 255), 2)

            cv2.circle(vis, (bx, by), 5, (0, 0, 255), -1)

            status = "TRACKING" if self.kalman_initialized else f"INIT ({len(self.detection_history)}/2)"
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Ball Tracking", vis)

        if bx is not None and by is not None:
            if self.is_inside_court(bx, by):
                return (bx, by)
            else:
                return None
        return None
