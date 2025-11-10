import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def draw_players(frame, player_list, ball_list, corners, inflated_corners=None):
    # Draw player/ball bounding boxes and court corners on frame
    display = frame.copy()
    corner_tl, corner_tr, corner_bl, corner_br = corners

    # Draw court corners (yellow)
    cv2.circle(display, corner_tl, 10, (0, 255, 255), -1)
    cv2.circle(display, corner_tr, 10, (0, 255, 255), -1)
    cv2.circle(display, corner_bl, 10, (0, 255, 255), -1)
    cv2.circle(display, corner_br, 10, (0, 255, 255), -1)

    # Draw inflated corners (magenta)
    if inflated_corners is not None:
        inf_tl, inf_tr, inf_bl, inf_br = inflated_corners
        cv2.circle(display, inf_tl, 8, (255, 0, 255), 2)
        cv2.circle(display, inf_tr, 8, (255, 0, 255), 2)
        cv2.circle(display, inf_bl, 8, (255, 0, 255), 2)
        cv2.circle(display, inf_br, 8, (255, 0, 255), 2)

        cv2.line(display, corner_tl, inf_tl, (255, 0, 255), 2)
        cv2.line(display, corner_tr, inf_tr, (255, 0, 255), 2)
        cv2.line(display, corner_bl, inf_bl, (255, 0, 255), 2)
        cv2.line(display, corner_br, inf_br, (255, 0, 255), 2)

    # Draw players (blue)
    for player in player_list:
        player_id, cx, cy, x, y, w, h, history = player[0], player[1], player[2], player[3], player[4], player[5], player[6], player[7]

        cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        area = w * h
        aspect = h / w if w > 0 else 0
        cv2.putText(display, f"P{player_id}", (x, y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(display, f"{w}x{h} A:{area}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(display, f"R:{aspect:.2f}", (x, y + h + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Draw trail
        if len(history) > 1:
            for i in range(1, len(history)):
                cv2.line(display, history[i-1], history[i], (255, 0, 0), 2)

    # Draw ball (green)
    for ball in ball_list:
        ball_id, cx, cy, x, y, w, h, history = ball

        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)

        area = w * h
        cv2.putText(display, f"BALL", (x, y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, f"{w}x{h} A:{area}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return display


def setup_plot(corner_tl, corner_tr, corner_bl, corner_br, original_corners, vertical_padding_ratio=0.5, expand_pixels=50):
    # Initialize matplotlib figure with frame and bird's eye view subplots
    plt.ion()
    fig, (ax_frame, ax_plot) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Tennis Player Tracking', fontsize=16)

    ax_plot.set_xlabel('Court Width (pixels)')
    ax_plot.set_ylabel('Court Length (pixels)')
    ax_plot.set_title('Player Positions (Bird\'s Eye View)')
    ax_plot.grid(True, alpha=0.3)
    ax_plot.set_aspect('equal')

    # Load court reference image
    court_ref_img = None
    court_ref_path = 'court_configurations/court_reference.png'
    if os.path.exists(court_ref_path):
        court_ref_img = cv2.imread(court_ref_path, cv2.IMREAD_GRAYSCALE)

    return fig, ax_frame, ax_plot, court_ref_img, original_corners


def update_plot(ax_frame, ax_plot, warped, player_list, ball_list, warp_matrix,
                frame_count, total_frames, court_ref_img=None, original_corners=None,
                processing_fps=0.0, video_fps=0):
    # Update matplotlib plots with warped frame and tracked positions
    ax_frame.clear()
    ax_plot.clear()

    # Show warped frame
    ax_frame.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    fps_ratio = (processing_fps / video_fps * 100) if video_fps > 0 else 0.0
    title = f'{processing_fps:.1f} FPS)'
    ax_frame.set_title(title, fontsize=10)
    ax_frame.axis('off')

    h, w = warped.shape[:2]

    # Plot setup
    ax_plot.set_xlim(0, w)
    ax_plot.set_ylim(h, 0)
    ax_plot.set_xlabel('Court Width (pixels)')
    ax_plot.set_ylabel('Court Length (pixels)')
    ax_plot.set_title('Tracking (Bird\'s Eye View)')
    ax_plot.set_aspect('equal')

    # Overlay court reference
    if court_ref_img is not None and original_corners is not None:
        # Reference corners
        ref_corners = np.float32([
            [284, 559],
            [1381, 559],
            [284, 2937],
            [1381, 2937]
        ])

        # Transform to warped space
        orig_pts = np.array([original_corners], dtype=np.float32)
        warped_corners = cv2.perspectiveTransform(orig_pts, warp_matrix)[0]

        ref_to_warped = cv2.getPerspectiveTransform(ref_corners, warped_corners)
        warped_ref = cv2.warpPerspective(court_ref_img, ref_to_warped, (w, h))

        ax_plot.imshow(warped_ref, extent=[0, w, h, 0],
                      cmap='gray', alpha=0.25, aspect='auto', zorder=0)
    else:
        ax_plot.grid(True, alpha=0.3)

    # Plot players (blue)
    for idx, player in enumerate(player_list):
        player_id, cx, cy, x, y, w, h, history = player[0], player[1], player[2], player[3], player[4], player[5], player[6], player[7]

        # Transform to warped coordinates
        point = np.array([[[cx, cy]]], dtype=np.float32)
        warped_point = cv2.perspectiveTransform(point, warp_matrix)
        wx, wy = warped_point[0][0]

        ax_plot.scatter(wx, wy, c='blue', s=200, marker='o',
                       edgecolors='white', linewidths=2, label=f'P{player_id}' if idx == 0 else '')

        # Plot trail
        if len(history) > 1:
            trail_x, trail_y = [], []
            for hist_pt in history:
                hist_point = np.array([[[hist_pt[0], hist_pt[1]]]], dtype=np.float32)
                warped_hist = cv2.perspectiveTransform(hist_point, warp_matrix)
                trail_x.append(warped_hist[0][0][0])
                trail_y.append(warped_hist[0][0][1])

            ax_plot.plot(trail_x, trail_y, c='blue', alpha=0.4, linewidth=2)

    # Plot ball (green)
    for ball in ball_list:
        ball_id, cx, cy, x, y, w, h, history = ball

        point = np.array([[[cx, cy]]], dtype=np.float32)
        warped_point = cv2.perspectiveTransform(point, warp_matrix)
        wx, wy = warped_point[0][0]

        ax_plot.scatter(wx, wy, c='lime', s=150, marker='o',
                       edgecolors='white', linewidths=3, label='Ball', zorder=10)

    if player_list or ball_list:
        ax_plot.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.pause(0.001)
