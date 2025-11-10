import cv2
import numpy as np
import torch
import time
import csv
from frame_predict.predict import load_knn_model_pickle, extract_features, _std_apply, _knn_predict_batch, _vectorise_features_from_names
from court_detect.extract_court_corners import BallTrackerNet, extract_corners_from_frame
from trackers.player_tracker import PlayerTracker
from trackers.player_tracker_yolo import PlayerTrackerYOLO
from trackers.ball_tracker import BallTracker
from trackers.ball_tracker_yolo import BallTrackerYOLO
from utils.visualisation import draw_players, setup_plot, update_plot


def predict_frame_validity(frame, classifier_model):
    # Check if frame is valid gameplay footage
    start_time = time.perf_counter()
    feats = extract_features(frame)
    x = _vectorise_features_from_names(feats, classifier_model["feature_names"])
    x_std = _std_apply(x, classifier_model["mu"], classifier_model["sigma"])
    yhat = int(_knn_predict_batch(classifier_model["Xtr"], classifier_model["ytr"], x_std,
                                   k=classifier_model["k"], weights=classifier_model["weights"])[0])
    inference_time = (time.perf_counter() - start_time) * 1000
    return bool(yhat == 1), inference_time


def inflate_corners(corners, buffer_ratio=0.15):
    # Extrapolate court sidelines outward to capture out-of-bounds movement
    corner_tl, corner_tr, corner_bl, corner_br = corners

    tl = np.array(corner_tl, dtype=np.float32)
    tr = np.array(corner_tr, dtype=np.float32)
    bl = np.array(corner_bl, dtype=np.float32)
    br = np.array(corner_br, dtype=np.float32)

    left_vec = bl - tl
    right_vec = br - tr

    new_tl = tl - left_vec * buffer_ratio
    new_tr = tr - right_vec * buffer_ratio
    new_bl = bl + left_vec * buffer_ratio
    new_br = br + right_vec * buffer_ratio

    return (tuple(new_tl.astype(int)), tuple(new_tr.astype(int)),
            tuple(new_bl.astype(int)), tuple(new_br.astype(int)))


def create_placeholder_frame(frame_width, frame_height, message, color=(50, 50, 50)):
    # Create placeholder frame for invalid footage
    placeholder = np.full((frame_height, frame_width, 3), color, dtype=np.uint8)
    text_color = (0, 0, 255)
    cv2.putText(placeholder, message, (50, frame_height // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 4)
    return placeholder


from collections import deque

class CutDetector:
    # Detect hard cuts and fades in video footage

    def __init__(self, hard_diff_threshold=12.0, window_size=8, avg_diff_threshold=6.0,
                 cum_diff_threshold=60.0, corr_threshold=0.90, trend_len=4):
        self.hard_diff_threshold = hard_diff_threshold
        self.window_size = window_size
        self.avg_diff_threshold = avg_diff_threshold
        self.cum_diff_threshold = cum_diff_threshold
        self.corr_threshold = corr_threshold
        self.trend_len = trend_len
        self.prev = None
        self.diff_win = deque(maxlen=window_size)
        self.corr_win = deque(maxlen=window_size)
        self.rolling_diff_window = deque(maxlen=5)

    def _hist_corr(self, a, b):
        # Calculate histogram correlation
        hist_a = cv2.calcHist([a], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        return float(corr)

    def update(self, curr_gray_small):
        # Update cut detector and return True if cut detected
        if self.prev is None:
            self.prev = curr_gray_small
            self.diff_win.clear()
            self.corr_win.clear()
            self.rolling_diff_window.clear()
            return True

        diff = cv2.absdiff(self.prev, curr_gray_small)
        mean_diff = float(np.mean(diff))
        corr = self._hist_corr(self.prev, curr_gray_small)
        self.prev = curr_gray_small

        self.diff_win.append(mean_diff)
        self.corr_win.append(corr)
        self.rolling_diff_window.append(mean_diff)
        avg_recent_diff = np.mean(self.rolling_diff_window)

        # Hard-cut detection
        if avg_recent_diff > self.hard_diff_threshold:
            self.diff_win.clear()
            self.corr_win.clear()
            self.rolling_diff_window.clear()
            return True

        # Fade detection
        if len(self.diff_win) >= self.diff_win.maxlen:
            avg_diff = sum(self.diff_win) / len(self.diff_win)
            cum_diff = sum(self.diff_win)
            min_corr = min(self.corr_win)

            trend_ok = False
            if len(self.corr_win) >= self.trend_len:
                recent = list(self.corr_win)[-self.trend_len:]
                trend_ok = all(recent[i] >= recent[i+1] for i in range(len(recent)-1))

            if ((avg_diff >= self.avg_diff_threshold and min_corr <= self.corr_threshold) or
                (cum_diff >= self.cum_diff_threshold and trend_ok)):
                self.diff_win.clear()
                self.corr_win.clear()
                self.rolling_diff_window.clear()
                return True

        return False


def process_video(video_path, frame_classifier_path, court_model_path,
                  show_visualization=True, save_output=False, output_path='output_tracking.mp4',
                  csv_path='tracking_unified.csv',
                  yolo_model_path='yolo/yolov8n.pt',
                  ball_yolo_model_path='yolo/training/runs/detect/train3/weights/best.pt'):
    # Main processing pipeline

    print("Loading models...")
    classifier = load_knn_model_pickle(frame_classifier_path)

    court_model = BallTrackerNet(out_channels=15)
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    court_model.load_state_dict(torch.load(court_model_path, map_location='cpu'))
    court_model = court_model.to(device)
    court_model.eval()
    print(f"Using device: {device}")

    N_VALIDATION = 10
    MIN_FRAMES_BETWEEN_CUTS = 10

    sampling_active = True
    samples_collected = 0
    valid_votes = 0
    is_valid_cached = False
    last_cut_frame_idx = -10**9

    gray_small_size = (160, 90)
    cut_detector = CutDetector(
        hard_diff_threshold=15.0,
        window_size=8,
        avg_diff_threshold=6.0,
        cum_diff_threshold=40.0,
        corr_threshold=0.90,
        trend_len=10,
    )

    player_tracker_mog2 = PlayerTracker(history=300, var_threshold=25)
    player_tracker_yolo = PlayerTrackerYOLO(model_path=yolo_model_path, conf_threshold=0.3)
    ball_tracker_mog2 = BallTracker()
    ball_tracker_yolo = BallTrackerYOLO(model_path=ball_yolo_model_path, conf_threshold=0.2)

    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame',
                         'p1_x_mog2', 'p1_y_mog2', 'p2_x_mog2', 'p2_y_mog2', 'ball_x_mog2', 'ball_y_mog2',
                         'p1_x_yolo', 'p1_y_yolo', 'p2_x_yolo', 'p2_y_yolo', 'ball_x_yolo', 'ball_y_yolo'])
    print(f"Unified tracking CSV: {csv_path}")

    ball_csv_path = 'ball_tracking_raw.csv'
    ball_csv_file = open(ball_csv_path, 'w', newline='')
    ball_csv_writer = csv.writer(ball_csv_file)
    ball_csv_writer.writerow(['frame', 'ball_x_mog2', 'ball_y_mog2', 'ball_x_yolo', 'ball_y_yolo'])
    print(f"Ball tracking CSV: {ball_csv_path}")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")
    print("Press 'q' to quit\n")

    corners = None
    plot_setup = None
    video_writer = None
    frame_idx = 0
    prev_valid = False

    frame_times = []
    max_frame_times = 30
    processing_fps = 0.0
    classifier_inference_time = 0.0

    while True:
        frame_start_time = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Cut detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, gray_small_size, interpolation=cv2.INTER_AREA)
        cut_detected = cut_detector.update(gray_small)
        if cut_detected and (frame_idx - last_cut_frame_idx) > MIN_FRAMES_BETWEEN_CUTS:
            last_cut_frame_idx = frame_idx
            sampling_active = True
            samples_collected = 0
            valid_votes = 0
            corners = None

        # Run classifier during sampling
        if sampling_active:
            is_valid, classifier_inference_time = predict_frame_validity(frame, classifier)
            samples_collected += 1
            valid_votes += 1 if is_valid else 0
            if samples_collected >= N_VALIDATION:
                is_valid_cached = (valid_votes >= ((N_VALIDATION + 1) // 2))
                sampling_active = False
        else:
            is_valid = is_valid_cached
            classifier_inference_time = 0.0

        raw_debug = frame.copy()
        status_text = "VALID" if is_valid else "INVALID"
        status_color = (0, 255, 0) if is_valid else (0, 0, 255)
        status_mode = "SAMPLE" if sampling_active else "CACHED"
        cv2.putText(raw_debug, f"{status_text} [{status_mode}]", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 4)
        cv2.putText(raw_debug, f"Classifier: {classifier_inference_time:.2f}ms", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Raw Debug', raw_debug)

        if not is_valid:
            if prev_valid:
                print(f"Frame {frame_idx}: Invalid footage - pausing tracking")
                corners = None

            status_note = "(sampling)" if sampling_active else "(cached)"
            placeholder = create_placeholder_frame(frame_width, frame_height, f"INVALID FRAME {status_note}")
            cv2.imshow('Overall Tracking', placeholder)

            if plot_setup is not None:
                fig, ax_frame, ax_plot, court_ref_img, orig_corners = plot_setup
                ax_frame.clear()
                ax_plot.clear()
                ax_frame.text(0.5, 0.5, 'INVALID FRAME', ha='center', va='center',
                            fontsize=20, color='red', transform=ax_frame.transAxes)
                ax_frame.axis('off')
                ax_plot.text(0.5, 0.5, 'TRACKING PAUSED', ha='center', va='center',
                            fontsize=20, color='red', transform=ax_plot.transAxes)
                ax_plot.axis('off')
                import matplotlib.pyplot as plt
                plt.pause(0.001)

            prev_valid = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if not prev_valid and is_valid:
            print(f"Frame {frame_idx}: Valid footage resumed - re-detecting corners")
            corners = None

        if corners is None:
            print(f"Detecting court corners on frame {frame_idx}...")
            _, homography_points, _ = extract_corners_from_frame(court_model, frame, device, refine=True)

            if homography_points is not None:
                corners = [
                    (int(homography_points[0][0]), int(homography_points[0][1])),
                    (int(homography_points[1][0]), int(homography_points[1][1])),
                    (int(homography_points[2][0]), int(homography_points[2][1])),
                    (int(homography_points[3][0]), int(homography_points[3][1]))
                ]
                print(f"Corners detected: TL={corners[0]}, TR={corners[1]}, BL={corners[2]}, BR={corners[3]}")

                inflated_corners = inflate_corners(corners, buffer_ratio=0.15)
                print(f"Inflated corners: TL={inflated_corners[0]}, TR={inflated_corners[1]}, BL={inflated_corners[2]}, BR={inflated_corners[3]}\n")

                ball_tracker_mog2.set_court_boundaries(inflated_corners)

                if show_visualization and plot_setup is None:
                    fig, ax_frame, ax_plot, court_ref_img, orig_corners = setup_plot(*inflated_corners, original_corners=corners)
                    plot_setup = (fig, ax_frame, ax_plot, court_ref_img, orig_corners)
            else:
                prev_valid = is_valid
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        inflated_corners = inflate_corners(corners, buffer_ratio=0.15)

        corner_tl, corner_tr, corner_bl, corner_br = inflated_corners
        top_width = np.sqrt((corner_tr[0] - corner_tl[0])**2 + (corner_tr[1] - corner_tl[1])**2)
        bottom_width = np.sqrt((corner_br[0] - corner_bl[0])**2 + (corner_br[1] - corner_bl[1])**2)
        avg_width = (top_width + bottom_width) / 2
        court_height = avg_width * 2.17
        vertical_padding = court_height * 0.5
        total_height = court_height + vertical_padding
        expand_pixels = 50
        dst_width = int(avg_width + 2 * expand_pixels)
        dst_height = int(total_height + expand_pixels)

        src_pts = np.float32([corner_tl, corner_tr, corner_bl, corner_br])
        dst_pts = np.float32([
            [expand_pixels, expand_pixels],
            [dst_width - expand_pixels, expand_pixels],
            [expand_pixels, dst_height - expand_pixels],
            [dst_width - expand_pixels, dst_height - expand_pixels]
        ])
        warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        player_list_mog2, _ = player_tracker_mog2.track(frame)
        _, _, p1_feet_yolo, p2_feet_yolo = player_tracker_yolo.track(frame)

        ball_tracker_mog2.processing_fps = processing_fps
        ball_tracker_mog2.video_fps = fps
        ball_list_mog2, _ = ball_tracker_mog2.track(frame, player_list_mog2, frame_idx)

        ball_x_yolo_raw, ball_y_yolo_raw = ball_tracker_yolo.track(frame)

        sorted_players_mog2 = sorted(player_list_mog2, key=lambda p: p[2])
        p1_x_mog2 = p1_y_mog2 = p2_x_mog2 = p2_y_mog2 = None
        if len(sorted_players_mog2) >= 1:
            p1_x_mog2, p1_y_mog2 = sorted_players_mog2[0][1], sorted_players_mog2[0][2]
        if len(sorted_players_mog2) >= 2:
            p2_x_mog2, p2_y_mog2 = sorted_players_mog2[1][1], sorted_players_mog2[1][2]

        ball_x_mog2 = ball_y_mog2 = None
        if len(ball_list_mog2) > 0:
            ball_x_mog2, ball_y_mog2 = ball_list_mog2[0][1], ball_list_mog2[0][2]

        p1_x_yolo = p1_y_yolo = p2_x_yolo = p2_y_yolo = None
        if p1_feet_yolo[0] is not None:
            p1_x_yolo, p1_y_yolo = p1_feet_yolo
        if p2_feet_yolo[0] is not None:
            p2_x_yolo, p2_y_yolo = p2_feet_yolo

        ball_x_yolo = ball_y_yolo = None
        if ball_x_yolo_raw is not None:
            ball_x_yolo, ball_y_yolo = ball_x_yolo_raw, ball_y_yolo_raw

        ball_csv_writer.writerow([frame_idx, ball_x_mog2, ball_y_mog2, ball_x_yolo, ball_y_yolo])

        if warp_matrix is not None:
            if p1_x_mog2 is not None and p1_y_mog2 is not None:
                pts = np.array([[[p1_x_mog2, p1_y_mog2]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(pts, warp_matrix)
                p1_x_mog2, p1_y_mog2 = float(warped[0][0][0]), float(warped[0][0][1])

            if p2_x_mog2 is not None and p2_y_mog2 is not None:
                pts = np.array([[[p2_x_mog2, p2_y_mog2]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(pts, warp_matrix)
                p2_x_mog2, p2_y_mog2 = float(warped[0][0][0]), float(warped[0][0][1])

            if ball_x_mog2 is not None and ball_y_mog2 is not None:
                pts = np.array([[[ball_x_mog2, ball_y_mog2]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(pts, warp_matrix)
                ball_x_mog2, ball_y_mog2 = float(warped[0][0][0]), float(warped[0][0][1])

            if p1_x_yolo is not None and p1_y_yolo is not None:
                pts = np.array([[[p1_x_yolo, p1_y_yolo]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(pts, warp_matrix)
                p1_x_yolo, p1_y_yolo = float(warped[0][0][0]), float(warped[0][0][1])

            if p2_x_yolo is not None and p2_y_yolo is not None:
                pts = np.array([[[p2_x_yolo, p2_y_yolo]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(pts, warp_matrix)
                p2_x_yolo, p2_y_yolo = float(warped[0][0][0]), float(warped[0][0][1])

            if ball_x_yolo is not None and ball_y_yolo is not None:
                pts = np.array([[[ball_x_yolo, ball_y_yolo]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(pts, warp_matrix)
                ball_x_yolo, ball_y_yolo = float(warped[0][0][0]), float(warped[0][0][1])

        csv_writer.writerow([frame_idx,
                            p1_x_mog2, p1_y_mog2, p2_x_mog2, p2_y_mog2, ball_x_mog2, ball_y_mog2,
                            p1_x_yolo, p1_y_yolo, p2_x_yolo, p2_y_yolo, ball_x_yolo, ball_y_yolo])

        player_list = player_list_mog2
        ball_list = ball_list_mog2

        display_frame = draw_players(frame, player_list, ball_list, corners, inflated_corners)
        status_mode = "SAMPLE" if sampling_active else "CACHED"
        cv2.putText(display_frame, f"VALID [{status_mode}]", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        warped = cv2.warpPerspective(display_frame, warp_matrix, (dst_width, dst_height))

        cv2.imshow('Overall Tracking', display_frame)

        if show_visualization and plot_setup is not None:
            fig, ax_frame, ax_plot, court_ref_img, orig_corners = plot_setup
            update_plot(ax_frame, ax_plot, warped, player_list, ball_list, warp_matrix,
                       frame_idx, total_frames, court_ref_img, orig_corners, processing_fps, fps)

        if save_output and is_valid:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = warped.shape[:2]
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                print(f"Saving to: {output_path} ({w}x{h} @ {fps} FPS)")
            video_writer.write(warped)

        prev_valid = is_valid

        frame_end_time = time.perf_counter()
        frame_time = frame_end_time - frame_start_time
        frame_times.append(frame_time)
        if len(frame_times) > max_frame_times:
            frame_times.pop(0)

        if len(frame_times) > 0:
            avg_frame_time = sum(frame_times) / len(frame_times)
            processing_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved: {output_path}")

    csv_file.close()
    ball_csv_file.close()
    cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames")
    print(f"Unified tracking CSV saved: {csv_path}")
    print(f"Ball tracking CSV saved: {ball_csv_path}")


if __name__ == "__main__":
    process_video(
        video_path='data/full_video/tennis_full_video.mp4',
        frame_classifier_path='BALL_TRACKING/court_validity_knn(92946_images).pkl',
        court_model_path='court_detect/models/tennis_court.pth',
        show_visualization=True,
        save_output=False,
        output_path='output_tracking.mp4',
        csv_path='tracking_unified.csv'
    )
