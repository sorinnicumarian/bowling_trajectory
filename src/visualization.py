"""
visualization.py
----------------
Plots and video overlays for the bowling ball analysis pipeline.

Outputs:
  - 2D trajectory overlay on video frames (annotated MP4)
  - 2D top-down lane view of the ball trajectory
  - 3D trajectory plot (X, Y, Z vs time)
  - Spin RPM timeseries plot
  - Summary figure
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional
import os


# ---------------------------------------------------------------------------
# Video overlay
# ---------------------------------------------------------------------------

def draw_trajectory_overlay(frame_bgr: np.ndarray,
                             points: List[dict],
                             current_frame: int,
                             det: Optional[dict] = None,
                             tail_len: int = 30) -> np.ndarray:
    """
    Draw the ball trail and current detection on a video frame.
    Returns annotated copy.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # Draw trail (fading dots)
    trail = [p for p in points if p['frame'] <= current_frame][-tail_len:]
    for i, p in enumerate(trail):
        alpha = (i + 1) / len(trail)
        color = (int(0 * alpha), int(200 * alpha), int(255 * alpha))
        cv2.circle(out, (int(p['u']), int(p['v'])), 3, color, -1)

    # Connect trail with thin lines
    for i in range(1, len(trail)):
        cv2.line(out,
                 (int(trail[i-1]['u']), int(trail[i-1]['v'])),
                 (int(trail[i  ]['u']), int(trail[i  ]['v'])),
                 (0, 255, 200), 1)

    # Draw current detection circle
    if det is not None:
        cv2.circle(out, (det['x'], det['y']), det['r'], (0, 255, 0), 2)
        cv2.circle(out, (det['x'], det['y']), 3, (0, 0, 255), -1)
        # Speed readout if available
        cur_pts = [p for p in points if p['frame'] == current_frame]
        if cur_pts:
            sp = cur_pts[0].get('speed_3d', 0)
            label = f"v={sp:.1f} m/s"
            cv2.putText(out, label,
                        (det['x'] + det['r'] + 5, det['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    return out


def write_annotated_video(video_path: str, output_path: str,
                          detections: List[Optional[dict]],
                          trajectory_points: List[dict],
                          spin_results: List[Optional[dict]] = None):
    """Write the full annotated video to output_path."""
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        det = detections[frame_idx] if frame_idx < len(detections) else None
        annotated = draw_trajectory_overlay(frame, trajectory_points,
                                            frame_idx, det)

        # Spin overlay
        if spin_results and frame_idx < len(spin_results) and spin_results[frame_idx]:
            sp = spin_results[frame_idx]
            rpm_text = f"RPM: {sp['rpm']:.0f}"
            cv2.putText(annotated, rpm_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 100), 2)

        out_vid.write(annotated)
        frame_idx += 1

    cap.release()
    out_vid.release()
    print(f"Annotated video saved to {output_path}")


# ---------------------------------------------------------------------------
# 2D top-down lane view
# ---------------------------------------------------------------------------

def plot_lane_topdown(trajectory_points: List[dict],
                      calibrator=None,
                      output_path: Optional[str] = None):
    """
    Plot the ball trajectory as a top-down view of the lane.
    X = lane width (0 = left gutter, 1.05 m = right gutter)
    Y = distance from foul line (metres)
    """
    if not trajectory_points:
        return

    Xs = [p['X_smooth'] for p in trajectory_points]
    Ys = [p['Y_smooth'] for p in trajectory_points]
    times = [p['t'] for p in trajectory_points]

    fig, ax = plt.subplots(figsize=(4, 10))

    # Draw lane outline
    lane_w = 1.05
    lane_l = 18.29
    ax.add_patch(mpatches.Rectangle((0, 0), lane_w, lane_l,
                                    fill=False, edgecolor='saddlebrown', lw=2))

    # Arrow markers at 4.57 m
    for xi in np.linspace(0.15, 0.90, 7):
        ax.plot(xi, 4.57, 'v', color='saddlebrown', ms=8)

    # Approach dots at 1.37 m
    for xi in np.linspace(0.10, 0.95, 7):
        ax.plot(xi, 1.37, '.', color='saddlebrown', ms=6)

    # Trajectory (coloured by time)
    sc = ax.scatter(Xs, Ys, c=times, cmap='plasma', s=20, zorder=5)
    ax.plot(Xs, Ys, color='steelblue', lw=1.5, alpha=0.7)

    # Start / end markers
    ax.plot(Xs[0],  Ys[0],  'go', ms=10, label='Release', zorder=6)
    ax.plot(Xs[-1], Ys[-1], 'rs', ms=10, label='End',     zorder=6)

    plt.colorbar(sc, ax=ax, label='Time (s)')
    ax.set_xlim(-0.1, lane_w + 0.1)
    ax.set_ylim(-0.5, lane_l + 0.5)
    ax.set_xlabel('Lane width (m)')
    ax.set_ylabel('Distance from foul line (m)')
    ax.set_title('Ball Trajectory – Top-Down Lane View')
    ax.legend(loc='upper right')
    ax.invert_yaxis()   # foul line at bottom, pins at top
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Lane top-down plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 3D trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory_3d(trajectory_points: List[dict],
                       output_path: Optional[str] = None):
    """3-D scatter/line plot of (X, Y, Z) trajectory."""
    if not trajectory_points:
        return

    Xs = [p['X_smooth'] for p in trajectory_points]
    Ys = [p['Y_smooth'] for p in trajectory_points]
    Zs = [p['Z']        for p in trajectory_points]

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection='3d')

    ax.plot(Xs, Ys, Zs, 'o-', color='steelblue', lw=2, ms=4, alpha=0.8)
    ax.scatter([Xs[0]], [Ys[0]], [Zs[0]], color='green', s=80, label='Release', zorder=5)
    ax.scatter([Xs[-1]], [Ys[-1]], [Zs[-1]], color='red', s=80, label='End', zorder=5)

    # Lane floor
    lane_w, lane_l = 1.05, 18.29
    xx, yy = np.meshgrid([0, lane_w], [0, min(lane_l, max(Ys) * 1.1)])
    ax.plot_surface(xx, yy, np.zeros_like(xx),
                    alpha=0.15, color='peru')

    ax.set_xlabel('X – lane width (m)')
    ax.set_ylabel('Y – distance from foul line (m)')
    ax.set_zlabel('Z – height (m)')
    ax.set_title('Ball 3-D Trajectory')
    ax.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"3D trajectory plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Spin timeseries
# ---------------------------------------------------------------------------

def plot_spin_timeseries(spin_results: List[dict],
                         fps: float,
                         output_path: Optional[str] = None):
    """Plot RPM and angular velocity over time."""
    valid = [(i / fps, r) for i, r in enumerate(spin_results)
             if r is not None and r.get('rpm', 0) > 0]
    if not valid:
        print("No spin data to plot.")
        return

    ts, rs = zip(*valid)
    rpms   = [r['rpm'] for r in rs]
    omegas = [r['omega_rad_s'] for r in rs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(ts, rpms, color='darkorange', lw=2)
    ax1.set_ylabel('Spin rate (RPM)')
    ax1.set_title('Bowling Ball Spin Over Time')
    ax1.axhline(np.mean(rpms), color='grey', ls='--', label=f'Mean: {np.mean(rpms):.0f} RPM')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ts, omegas, color='purple', lw=2)
    ax2.set_ylabel('ω (rad/s)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Spin timeseries plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------

def plot_summary(trajectory_points: List[dict],
                 spin_summary: dict,
                 traj_summary: dict,
                 output_path: Optional[str] = None):
    """Single-page summary figure with key metrics."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Bowling Ball Analysis – Summary', fontsize=16, fontweight='bold')

    # Left: top-down lane view (re-use function on subplot)
    ax1 = fig.add_subplot(1, 3, 1)
    if trajectory_points:
        Xs = [p['X_smooth'] for p in trajectory_points]
        Ys = [p['Y_smooth'] for p in trajectory_points]
        ax1.plot(Xs, Ys, 'o-', color='steelblue', ms=3)
        ax1.plot(Xs[0],  Ys[0],  'go', ms=8, label='Release')
        ax1.plot(Xs[-1], Ys[-1], 'rs', ms=8, label='End')
    ax1.set_xlim(-0.1, 1.2)
    ax1.set_xlabel('Lane X (m)'); ax1.set_ylabel('Lane Y (m)')
    ax1.set_title('Top-down'); ax1.legend(fontsize=8)
    ax1.invert_yaxis()

    # Middle: speed profile
    ax2 = fig.add_subplot(1, 3, 2)
    if trajectory_points:
        ts = [p['t'] for p in trajectory_points]
        sp = [p.get('speed_3d', 0) for p in trajectory_points]
        ax2.plot(ts, sp, color='teal', lw=2)
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Ball Speed')
        ax2.grid(alpha=0.3)

    # Right: text summary
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off')
    lines = [
        f"Frames tracked:   {traj_summary.get('n_frames', '–')}",
        f"Duration:         {traj_summary.get('duration_s', 0):.2f} s",
        f"Distance:         {traj_summary.get('distance_m', 0):.2f} m",
        f"Avg speed:        {traj_summary.get('avg_speed_ms', 0):.2f} m/s",
        f"Max speed:        {traj_summary.get('max_speed_ms', 0):.2f} m/s",
        f"Hook angle:       {traj_summary.get('hook_angle_deg', 0):.1f}°",
        f"Peak height:      {traj_summary.get('peak_height_m', 0):.3f} m",
        "",
        f"Avg spin:         {spin_summary.get('avg_rpm', 0):.0f} RPM",
        f"Max spin:         {spin_summary.get('max_rpm', 0):.0f} RPM",
        f"ω avg:            {spin_summary.get('avg_omega_rads', 0):.2f} rad/s",
    ]
    for i, line in enumerate(lines):
        ax3.text(0.05, 0.95 - i * 0.08, line,
                 transform=ax3.transAxes, fontsize=10,
                 fontfamily='monospace', va='top')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Summary plot saved to {output_path}")
    else:
        plt.show()
    plt.close()
