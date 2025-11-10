
import cv2
import numpy as np
from trackers.player_tracker import PlayerTracker

# COCO model (18 keypoints)
protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"

# POSE_PAIRS for COCO model
POSE_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
]

video_path = 'data/full_video/video.mp4'
cap = cv2.VideoCapture(video_path)

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
threshold = 0.1

player_tracker = PlayerTracker()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frameCopy = np.copy(frame)
    # Get player bounding boxes
    player_list, _ = player_tracker.track(frame)

    for player in player_list:
        _, _, _, x, y, w, h, _ = player
        # Crop player region
        player_roi = frame[y:y+h, x:x+w]
        if player_roi.size == 0:
            continue
        roi_h, roi_w = player_roi.shape[:2]
        inpBlob = cv2.dnn.blobFromImage(player_roi, 1.0 / 255, (roi_w, roi_h), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]
        points = []
        nPoints = output.shape[1]
        for i in range(nPoints):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            px = int((roi_w * point[0]) / W)
            py = int((roi_h * point[1]) / H)
            if prob > threshold:
                # Offset by bbox position
                cv2.circle(frameCopy, (x + px, y + py), 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frameCopy, str(i), (x + px, y + py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                points.append((x + px, y + py))
            else:
                points.append(None)
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                cv2.line(frameCopy, points[partA], points[partB], (0, 255, 0), 2)

    cv2.imshow('Pose Estimation', frameCopy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()