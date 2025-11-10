import os
import json
import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps, court_ref
import argparse

def read_video(path_video):
    """ Read video file
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def write_video(imgs_new, fps, path_output_video):
    height, width = imgs_new[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))
    for num in range(len(imgs_new)):
        frame = imgs_new[num]
        out.write(frame)
    out.release() 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--input_path', type=str, help='path to input video')
    parser.add_argument('--output_path', type=str, help='path to output video')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    parser.add_argument('--coords_path', type=str, default=None, help='optional path to save per-frame keypoints (JSON)')
    parser.add_argument('--warp_video_path', type=str, default=None, help='optional path to save bird-eye video')
    parser.add_argument('--warp_keep_last', action='store_true',
                        help='reuse last valid warp when homography is unavailable for a frame')
    parser.add_argument('--lock_homography', action='store_true',
                        help='freeze homography to the first valid matrix to reduce jitter')
    args = parser.parse_args()

    model = BallTrackerNet(out_channels=15)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360
    INPUT_WIDTH = OUTPUT_WIDTH * 2
    INPUT_HEIGHT = OUTPUT_HEIGHT * 2
    
    frames, fps = read_video(args.input_path)
    if len(frames) == 0:
        raise RuntimeError(f'Failed to read any frames from {args.input_path}. Check that the path is correct and the video is not corrupted.')
    frames_upd = []
    warp_frames = [] if args.warp_video_path is not None else None
    coords_records = [] if args.coords_path is not None else None
    court_width = int(court_ref.court_total_width)
    court_height = int(court_ref.court_total_height)
    last_warp = None
    locked_matrix = None
    locked_inverse = None

    for frame_idx, image in enumerate(tqdm(frames)):
        img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        orig_height, orig_width = image.shape[:2]
        scale_x = orig_width / INPUT_WIDTH
        scale_y = orig_height / INPUT_HEIGHT

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            if x_pred is not None and y_pred is not None:
                x_pred *= scale_x
                y_pred *= scale_y
                if args.use_refine_kps and kps_num not in [8, 12, 9]:
                    x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            points.append((x_pred, y_pred))

        matrix_trans = get_trans_matrix(points)
        if args.lock_homography:
            if matrix_trans is not None and locked_matrix is None:
                locked_matrix = matrix_trans
                try:
                    locked_inverse = np.linalg.inv(locked_matrix)
                except np.linalg.LinAlgError:
                    locked_inverse = None
            matrix_trans = locked_matrix
        if args.use_homography and matrix_trans is not None:
            homography_points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            points = [np.squeeze(x) for x in homography_points]

        if coords_records is not None:
            coords_records.append({
                'frame': frame_idx,
                'points': [
                    {'x': float(pt[0]) if pt[0] is not None else None,
                     'y': float(pt[1]) if pt[1] is not None else None}
                    for pt in points
                ],
                'homography': matrix_trans.tolist() if matrix_trans is not None else None
            })

        if warp_frames is not None:
            warp_matrix = None
            if args.lock_homography and locked_inverse is not None:
                warp_matrix = locked_inverse
            elif matrix_trans is not None:
                try:
                    warp_matrix = np.linalg.inv(matrix_trans)
                except np.linalg.LinAlgError:
                    warp_matrix = None
            if warp_matrix is not None:
                bird_eye = cv2.warpPerspective(image, warp_matrix, (court_width, court_height))
                warp_frames.append(bird_eye)
                last_warp = bird_eye
            elif args.warp_keep_last and last_warp is not None:
                warp_frames.append(last_warp.copy())

        for j in range(len(points)):
            if points[j][0] is not None:
                image = cv2.circle(image, (int(points[j][0]), int(points[j][1])),
                                  radius=0, color=(0, 0, 255), thickness=10)
        frames_upd.append(image)

    write_video(frames_upd, fps, args.output_path)

    if args.warp_video_path is not None and warp_frames and len(warp_frames) > 0:
        write_video(warp_frames, fps, args.warp_video_path)

    if args.coords_path is not None and coords_records is not None:
        with open(args.coords_path, 'w') as f:
            json.dump(coords_records, f, indent=2)
    
