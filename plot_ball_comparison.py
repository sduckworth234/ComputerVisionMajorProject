import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Load ball tracking data
csv_path = 'ball_tracking_raw.csv'
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} frames from {csv_path}")

# Drop rows where both methods have no detection
df_valid = df.dropna(subset=['ball_x_mog2', 'ball_x_yolo'], how='all')
print(f"Valid frames (at least one method detected): {len(df_valid)}")

# Frames where both detected
df_both = df.dropna(subset=['ball_x_mog2', 'ball_y_mog2', 'ball_x_yolo', 'ball_y_yolo'])
print(f"Frames where both methods detected: {len(df_both)}")

# Calculate error metrics for frames where both detected
if len(df_both) > 0:
    errors = []
    for _, row in df_both.iterrows():
        mog2_pos = (row['ball_x_mog2'], row['ball_y_mog2'])
        yolo_pos = (row['ball_x_yolo'], row['ball_y_yolo'])
        error = euclidean(mog2_pos, yolo_pos)
        errors.append(error)

    errors = np.array(errors)

    print("\n=== Error Metrics ===")
    print(f"Mean Error: {np.mean(errors):.2f} pixels")
    print(f"Median Error: {np.median(errors):.2f} pixels")
    print(f"Std Dev: {np.std(errors):.2f} pixels")
    print(f"Min Error: {np.min(errors):.2f} pixels")
    print(f"Max Error: {np.max(errors):.2f} pixels")
    print(f"95th Percentile: {np.percentile(errors, 95):.2f} pixels")

# Detection rates
mog2_detections = df['ball_x_mog2'].notna().sum()
yolo_detections = df['ball_x_yolo'].notna().sum()
total_frames = len(df)

print("\n=== Detection Rates ===")
print(f"MOG2 Detection Rate: {mog2_detections}/{total_frames} ({mog2_detections/total_frames*100:.1f}%)")
print(f"YOLO Detection Rate: {yolo_detections}/{total_frames} ({yolo_detections/total_frames*100:.1f}%)")

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. XY Trajectory Plot
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(df['ball_x_mog2'], df['ball_y_mog2'], s=10, alpha=0.5, label='MOG2', c='blue')
ax1.scatter(df['ball_x_yolo'], df['ball_y_yolo'], s=10, alpha=0.5, label='YOLO', c='red')
ax1.set_xlabel('X Position (pixels)')
ax1.set_ylabel('Y Position (pixels)')
ax1.set_title('Ball Trajectories (Raw Pixel Space)')
ax1.legend()
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 2. X Position over time
ax2 = plt.subplot(2, 3, 2)
ax2.plot(df['frame'], df['ball_x_mog2'], linewidth=1, alpha=0.7, label='MOG2')
ax2.plot(df['frame'], df['ball_x_yolo'], linewidth=1, alpha=0.7, label='YOLO')
ax2.set_xlabel('Frame')
ax2.set_ylabel('X Position (pixels)')
ax2.set_title('X Position Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Y Position over time
ax3 = plt.subplot(2, 3, 3)
ax3.plot(df['frame'], df['ball_y_mog2'], linewidth=1, alpha=0.7, label='MOG2')
ax3.plot(df['frame'], df['ball_y_yolo'], linewidth=1, alpha=0.7, label='YOLO')
ax3.set_xlabel('Frame')
ax3.set_ylabel('Y Position (pixels)')
ax3.set_title('Y Position Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Error over time (for frames where both detected)
if len(df_both) > 0:
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(df_both['frame'], errors, linewidth=1, color='purple')
    ax4.axhline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.1f}px')
    ax4.axhline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.1f}px')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Euclidean Distance (pixels)')
    ax4.set_title('Position Error Between Methods')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# 5. Error histogram
if len(df_both) > 0:
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax5.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.1f}px')
    ax5.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.1f}px')
    ax5.set_xlabel('Error (pixels)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

# 6. Detection overlap
ax6 = plt.subplot(2, 3, 6)
only_mog2 = (df['ball_x_mog2'].notna()) & (df['ball_x_yolo'].isna())
only_yolo = (df['ball_x_mog2'].isna()) & (df['ball_x_yolo'].notna())
both = (df['ball_x_mog2'].notna()) & (df['ball_x_yolo'].notna())
neither = (df['ball_x_mog2'].isna()) & (df['ball_x_yolo'].isna())

categories = ['Both', 'Only MOG2', 'Only YOLO', 'Neither']
counts = [both.sum(), only_mog2.sum(), only_yolo.sum(), neither.sum()]
colors = ['green', 'blue', 'red', 'gray']

ax6.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Number of Frames')
ax6.set_title('Detection Overlap')
ax6.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (cat, count) in enumerate(zip(categories, counts)):
    ax6.text(i, count, f'{count}\n({count/total_frames*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('ball_tracking_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: ball_tracking_comparison.png")
plt.show()
