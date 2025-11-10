import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps, court_ref
import argparse
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--input_path', type=str, help='path to input image')
    parser.add_argument('--output_path', type=str, help='path to output image')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    parser.add_argument('--coords_path', type=str, default=None, help='optional path to save keypoints (JSON)')
    parser.add_argument('--warp_output_path', type=str, default=None, help='optional path to save bird-eye view image')
    args = parser.parse_args()

    model = BallTrackerNet(out_channels=15)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360
    INPUT_WIDTH = OUTPUT_WIDTH * 2
    INPUT_HEIGHT = OUTPUT_HEIGHT * 2

    image = cv2.imread(args.input_path)
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
        heatmap = (pred[kps_num]*255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if x_pred is not None and y_pred is not None:
            x_pred *= scale_x
            y_pred *= scale_y
            if args.use_refine_kps and kps_num not in [8, 12, 9]:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))

    matrix_trans = get_trans_matrix(points)
    if args.use_homography and matrix_trans is not None:
        points = cv2.perspectiveTransform(refer_kps, matrix_trans)
        points = [np.squeeze(x) for x in points]

    if args.coords_path is not None:
        with open(args.coords_path, 'w') as f:
            json.dump({
                'points': [
                    {'x': float(pt[0]) if pt[0] is not None else None,
                     'y': float(pt[1]) if pt[1] is not None else None}
                    for pt in points
                ],
                'homography': matrix_trans.tolist() if matrix_trans is not None else None
            }, f, indent=2)

    if args.warp_output_path is not None and matrix_trans is not None:
        try:
            warp_matrix = np.linalg.inv(matrix_trans)
            bird_eye = cv2.warpPerspective(image, warp_matrix,
                                           (int(court_ref.court_total_width), int(court_ref.court_total_height)))
            cv2.imwrite(args.warp_output_path, bird_eye)
        except np.linalg.LinAlgError:
            pass

    for j in range(len(points)):
        if points[j][0] is not None:
            image = cv2.circle(image, (int(points[j][0]), int(points[j][1])),
                               radius=0, color=(0, 0, 255), thickness=10)

    cv2.imwrite(args.output_path, image)
