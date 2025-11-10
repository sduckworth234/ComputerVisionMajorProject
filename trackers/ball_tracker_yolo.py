from ultralytics import YOLO


class BallTrackerYOLO:
    # YOLO-based ball tracker using trained model

    def __init__(self, model_path='yolo/training/runs/detect/train3/weights/best.pt', conf_threshold=0.2):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def track(self, frame):
        # Detect ball and return center coordinates
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        boxes = results[0].boxes

        ball_x = ball_y = None

        if len(boxes) > 0:
            best_idx = boxes.conf.argmax()
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
            ball_x = int((x1 + x2) / 2)
            ball_y = int((y1 + y2) / 2)

        return ball_x, ball_y
