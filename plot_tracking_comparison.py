import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'tracking_unified.csv'
df = pd.read_csv(csv_path)

all_x = pd.concat([df['p1_x_mog2'], df['p1_x_yolo'], df['p2_x_mog2'], df['p2_x_yolo'],
                   df['ball_x_mog2'], df['ball_x_yolo']]).dropna()
all_y = pd.concat([df['p1_y_mog2'], df['p1_y_yolo'], df['p2_y_mog2'], df['p2_y_yolo'],
                   df['ball_y_mog2'], df['ball_y_yolo']]).dropna()

if len(all_x) > 0 and len(all_y) > 0:
    max_x = all_x.max()
    max_y = all_y.max()
    expand_pixels = 50
    court_x_min, court_x_max = expand_pixels, max_x - expand_pixels
    court_y_min, court_y_max = expand_pixels, max_y - expand_pixels
else:
    expand_pixels = 50
    court_x_min, court_x_max = 50, 500
    court_y_min, court_y_max = 50, 1000

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

ax = axes[0]
ax.plot(df['p1_x_mog2'], df['p1_y_mog2'], 'o', linewidth=0.8, markersize=2, label='Player 1 MOG2', alpha=0.7, color='blue')
ax.plot(df['p1_x_yolo'], df['p1_y_yolo'], 's', linewidth=0.8, markersize=2, label='Player 1 YOLO', alpha=0.7, color='cyan')
ax.plot(df['p2_x_mog2'], df['p2_y_mog2'], 'o', linewidth=0.8, markersize=2, label='Player 2 MOG2', alpha=0.7, color='red')
ax.plot(df['p2_x_yolo'], df['p2_y_yolo'], 's', linewidth=0.8, markersize=2, label='Player 2 YOLO', alpha=0.7, color='orange')

ax.set_xlim(0, max_x if len(all_x) > 0 else 600)
ax.set_ylim(0, max_y if len(all_y) > 0 else 1200)
ax.invert_yaxis()
ax.set_xlabel('X (warped pixels)')
ax.set_ylabel('Y (warped pixels)')
ax.set_title('Player Tracking Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

ax = axes[1]
ax.plot(df['ball_x_mog2'], df['ball_y_mog2'], 'o', linewidth=0.8, markersize=2, label='MOG2', alpha=0.7)
ax.plot(df['ball_x_yolo'], df['ball_y_yolo'], 's', linewidth=0.8, markersize=2, label='YOLO', alpha=0.7)

ax.set_xlim(0, max_x if len(all_x) > 0 else 600)
ax.set_ylim(0, max_y if len(all_y) > 0 else 1200)
ax.invert_yaxis()
ax.set_xlabel('X (warped pixels)')
ax.set_ylabel('Y (warped pixels)')
ax.set_title('Ball Tracking Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('tracking_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
