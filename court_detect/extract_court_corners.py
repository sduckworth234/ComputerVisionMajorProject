import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance
from sympy import Line
import sympy


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=14):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def postprocess(heatmap, scale=2, low_thresh=155, min_radius=10, max_radius=30):
    x_pred, y_pred = None, None
    _, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    if circles is not None:
        x_pred = circles[0][0][0] * scale
        y_pred = circles[0][0][1] * scale
    return x_pred, y_pred


def line_intersection(line1, line2):
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))
    intersection = l1.intersection(l2)
    point = None
    if len(intersection) > 0:
        if isinstance(intersection[0], sympy.geometry.point.Point2D):
            point = intersection[0].coordinates
    return point


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    lines = np.squeeze(lines)
    if len(getattr(lines, 'shape', ())) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines


def merge_lines(lines):
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 20 and dist2 < 20:
                        line = np.array(
                            [
                                int((x1 + x3) / 2),
                                int((y1 + y3) / 2),
                                int((x2 + x4) / 2),
                                int((y2 + y4) / 2)
                            ],
                            dtype=np.int32
                        )
                        mask[i + j + 1] = False
            new_lines.append(line)
    return new_lines


def refine_kps(img, x_ct, y_ct, crop_size=40):
    refined_x_ct, refined_y_ct = x_ct, y_ct

    img_height, img_width = img.shape[:2]
    x_min = max(x_ct - crop_size, 0)
    x_max = min(img_height, x_ct + crop_size)
    y_min = max(y_ct - crop_size, 0)
    y_max = min(img_width, y_ct + crop_size)

    img_crop = img[x_min:x_max, y_min:y_max]
    lines = detect_lines(img_crop)

    if len(lines) > 1:
        lines = merge_lines(lines)
        if len(lines) == 2:
            inters = line_intersection(lines[0], lines[1])
            if inters:
                new_x_ct = int(inters[1])
                new_y_ct = int(inters[0])
                if 0 < new_x_ct < img_crop.shape[0] and 0 < new_y_ct < img_crop.shape[1]:
                    refined_x_ct = x_min + new_x_ct
                    refined_y_ct = y_min + new_y_ct
    return refined_y_ct, refined_x_ct


class CourtReference:
    def __init__(self):
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))
        self.top_extra_part = (832.5, 580)
        self.bottom_extra_part = (832.5, 2910)

        self.key_points = [
            *self.baseline_top,
            *self.baseline_bottom,
            *self.left_inner_line,
            *self.right_inner_line,
            *self.top_inner_line,
            *self.bottom_inner_line,
            *self.middle_line
        ]

        self.court_conf = {
            1: [*self.baseline_top, *self.baseline_bottom],
            2: [
                self.left_inner_line[0],
                self.right_inner_line[0],
                self.left_inner_line[1],
                self.right_inner_line[1]
            ],
            3: [
                self.left_inner_line[0],
                self.right_court_line[0],
                self.left_inner_line[1],
                self.right_court_line[1]
            ],
            4: [
                self.left_court_line[0],
                self.right_inner_line[0],
                self.left_court_line[1],
                self.right_inner_line[1]
            ],
            5: [*self.top_inner_line, *self.bottom_inner_line],
            6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
            7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
            8: [
                self.right_inner_line[0],
                self.right_court_line[0],
                self.right_inner_line[1],
                self.right_court_line[1]
            ],
            9: [
                self.left_court_line[0],
                self.left_inner_line[0],
                self.left_court_line[1],
                self.left_inner_line[1]
            ],
            10: [
                self.top_inner_line[0],
                self.middle_line[0],
                self.bottom_inner_line[0],
                self.middle_line[1]
            ],
            11: [
                self.middle_line[0],
                self.top_inner_line[1],
                self.middle_line[1],
                self.bottom_inner_line[1]
            ],
            12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]
        }


court_ref = CourtReference()
refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

court_conf_ind = {}
for i in range(len(court_ref.court_conf)):
    conf = court_ref.court_conf[i + 1]
    inds = []
    for j in range(4):
        inds.append(court_ref.key_points.index(conf[j]))
    court_conf_ind[i + 1] = inds


def get_trans_matrix(points):
    matrix_trans = None
    dist_max = np.inf
    for conf_ind in range(1, 13):
        conf = court_ref.court_conf[conf_ind]

        inds = court_conf_ind[conf_ind]
        inters = [points[inds[0]], points[inds[1]], points[inds[2]], points[inds[3]]]
        if not any([None in x for x in inters]):
            matrix, _ = cv2.findHomography(np.float32(conf), np.float32(inters), method=0)
            trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
            trans_kps = np.squeeze(trans_kps, axis=1)
            dists = []
            for i in range(12):
                if i not in inds and points[i][0] is not None:
                    dists.append(distance.euclidean(points[i], trans_kps[i]))
            if dists:
                dist_median = np.mean(dists)
                if dist_median < dist_max:
                    matrix_trans = matrix
                    dist_max = dist_median
    return matrix_trans


def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def extract_corners_from_frame(model, frame, device, refine=True):
    output_width = 640
    output_height = 360
    input_width = output_width * 2
    input_height = output_height * 2

    resized = cv2.resize(frame, (output_width, output_height))
    inp = (resized.astype(np.float32) / 255.0)
    inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0)

    with torch.no_grad():
        out = model(inp.float().to(device))[0]
    pred = F.sigmoid(out).detach().cpu().numpy()

    orig_height, orig_width = frame.shape[:2]
    scale_x = orig_width / input_width
    scale_y = orig_height / input_height

    points = []
    for kps_num in range(14):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if x_pred is not None and y_pred is not None:
            x_pred *= scale_x
            y_pred *= scale_y
            if refine and kps_num not in [8, 12, 9]:
                x_pred, y_pred = refine_kps(frame, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))

    matrix_trans = get_trans_matrix(points)
    transformed_points = None
    if matrix_trans is not None:
        transformed = cv2.perspectiveTransform(refer_kps, matrix_trans)
        transformed_points = [tuple(np.squeeze(x)) for x in transformed]
    return points, transformed_points, matrix_trans


def extract_initial_court_corners(model_path, input_path,
                                  footage_type='gameplay', use_refine_kps=False):
    model = BallTrackerNet(out_channels=15)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Process frames one at a time instead of loading all into memory
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video from {input_path}')

    corner_indices = [0, 1, 2, 3]
    initial_inverse = None
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_points, homography_points, homography = extract_corners_from_frame(
            model, frame, device, refine=use_refine_kps
        )

        # choose best available points (homography-adjusted preferred)
        points_source = homography_points if homography_points is not None else image_points
        corners = []
        for ci in corner_indices:
            pt = points_source[ci]
            if pt[0] is None or pt[1] is None:
                corners.append({'x': None, 'y': None})
            else:
                corners.append({'x': float(pt[0]), 'y': float(pt[1])})

        warped_corners = None
        if footage_type == 'gameplay':
            if initial_inverse is None and homography is not None:
                try:
                    initial_inverse = np.linalg.inv(homography)
                except np.linalg.LinAlgError:
                    initial_inverse = None
            if initial_inverse is not None:
                valid = all(c['x'] is not None and c['y'] is not None for c in corners)
                if valid:
                    pts = np.array([[ [c['x'], c['y']] for c in corners ]], dtype=np.float32)
                    warped = cv2.perspectiveTransform(pts, initial_inverse)
                    warped_corners = [
                        {'x': float(p[0]), 'y': float(p[1])} for p in warped[0]
                    ]

        valid_standard = all(c['x'] is not None and c['y'] is not None for c in corners)
        if valid_standard:
            cap.release()
            return {
                'frame_index': idx,
                'corners': corners,
                'warped_corners': warped_corners,
                'homography': homography.tolist() if homography is not None else None
            }

        idx += 1

    cap.release()
    raise RuntimeError('No frame with complete corner detections found.')


def main():
    # Set the values below to match your environment.
    config = {
        'model_path': 'models/tennis_court.pth',
        'input_path': 'clips/video.mp4',
        'footage_type': 'gameplay',  # or 'auxiliary'
        'use_refine_kps': True,
    }
    result = extract_initial_court_corners(
        model_path=config['model_path'],
        input_path=config['input_path'],
        footage_type=config['footage_type'],
        use_refine_kps=config['use_refine_kps']
    )
    print('First detected corners frame:', result['frame_index'])
    print('Corners:', result['corners'])
    if result['warped_corners'] is not None:
        print('Warped corners:', result['warped_corners'])
    if result['homography'] is not None:
        print('Homography matrix:', result['homography'])

    # Visualize the corners on the frame
    cap = cv2.VideoCapture(config['input_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, result['frame_index'])
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_height, frame_width = frame.shape[:2]
        print(f'\nFrame dimensions: {frame_width}x{frame_height}')

        # Draw original detected corners (solid circles)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Cyan
        for idx, corner in enumerate(result['corners']):
            if corner['x'] is not None and corner['y'] is not None:
                x, y = int(corner['x']), int(corner['y'])
                cv2.circle(frame, (x, y), 10, colors[idx], -1)
                cv2.putText(frame, f"C{idx}", (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[idx], 2)

        # Draw warped corners (hollow circles with X)
        # Note: Warped corners are in reference court space, may be outside frame bounds
        if result['warped_corners'] is not None:
            print('\nWarped corners status:')
            for idx, corner in enumerate(result['warped_corners']):
                if corner['x'] is not None and corner['y'] is not None:
                    x, y = int(corner['x']), int(corner['y'])
                    in_bounds = (0 <= x < frame_width and 0 <= y < frame_height)
                    print(f"  W{idx}: ({x}, {y}) - {'VISIBLE' if in_bounds else 'OUT OF FRAME BOUNDS'}")

                    if in_bounds:
                        # Hollow circle
                        cv2.circle(frame, (x, y), 12, colors[idx], 3)
                        # Draw X marker
                        cv2.drawMarker(frame, (x, y), colors[idx],
                                     cv2.MARKER_CROSS, 20, 3)
                        cv2.putText(frame, f"W{idx}", (x + 15, y + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[idx], 2)
                    else:
                        # Draw arrow pointing to off-screen corner
                        arrow_x = max(20, min(x, frame_width - 20))
                        arrow_y = max(20, min(y, frame_height - 20))
                        cv2.putText(frame, f"W{idx} ->", (arrow_x - 60, arrow_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2)

        # Add legend
        legend_y_start = 30
        legend_x = 20
        cv2.putText(frame, "Legend:", (legend_x, legend_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Original corners legend
        cv2.circle(frame, (legend_x + 10, legend_y_start + 35), 8, (255, 255, 255), -1)
        cv2.putText(frame, "Original Corners (C0-C3)", (legend_x + 30, legend_y_start + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Warped corners legend
        cv2.circle(frame, (legend_x + 10, legend_y_start + 70), 8, (255, 255, 255), 2)
        cv2.drawMarker(frame, (legend_x + 10, legend_y_start + 70), (255, 255, 255),
                      cv2.MARKER_CROSS, 15, 2)
        cv2.putText(frame, "Warped Corners (W0-W3)", (legend_x + 30, legend_y_start + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save and display
        output_path = 'court_corners_visualization.jpg'
        cv2.imwrite(output_path, frame)
        print(f'\nVisualization saved to: {output_path}')

        # Optional: display the image (comment out if running headless)
        cv2.imshow('Court Corners', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()