"""
trajectory_3d.py
----------------
3-D ball trajectory reconstruction from per-frame detections.

Algorithm (IACV course, lecture G):
  1. Each frame gives a ball centre (u, v) in image coords.
  2. Back-project (u, v) through K^{-1} to get ray direction d.
  3. Intersect ray with lane plane (Z_world = 0) using the homography
     computed by lane_calibration.py  →  (X, Y) in metres on the lane.
  4. Height Z above the lane cannot be recovered from a single image;
     we fit a parabola Z(t) = Z0 + vz*t - 0.5*g*t²  to the sequence,
     using the constraint that Z=0 at the foul line and Z≈0 at impact.
  5. Smooth the trajectory with a Savitzky-Golay filter to reduce noise.
  6. Compute speed and side-drift (hook angle) from the smoothed positions.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from typing import List, Optional, Tuple
import json


GRAVITY = 9.81   # m/s²


# ---------------------------------------------------------------------------
# Ray ↔ plane intersection
# ---------------------------------------------------------------------------

def _ray_plane_intersection(K, u, v, plane_normal=(0, 0, 1), plane_d=0.0):
    """
    Intersect the back-projected ray from pixel (u,v) with a plane
    n·X = d  (default: Z=0 lane plane).

    Returns scale t such that  X_cam = t * K^{-1} [u v 1]^T,
    and the 3-D direction vector.
    """
    K_inv = np.linalg.inv(K)
    d_cam = K_inv @ np.array([u, v, 1.0])
    n = np.array(plane_normal, dtype=float)
    denom = n @ d_cam
    if abs(denom) < 1e-10:
        return None, d_cam   # ray parallel to plane
    t = (plane_d - n @ np.zeros(3)) / denom
    return t, d_cam


# ---------------------------------------------------------------------------
# Parabolic height model
# ---------------------------------------------------------------------------

def _parabola(t, Z0, vz):
    """Z(t) = Z0 + vz*t - 0.5*g*t²"""
    return Z0 + vz * t - 0.5 * GRAVITY * t**2


def fit_height_parabola(times, heights=None):
    """
    Fit parabolic height model to a sequence of times.
    If heights are unknown, initialise with heuristic (ball released at ~0.5 m).

    Returns (Z0, vz) coefficients.
    """
    if heights is not None and len(heights) >= 3:
        p0 = [0.3, 2.0]
        try:
            popt, _ = curve_fit(_parabola, times, heights, p0=p0, maxfev=5000)
            return float(popt[0]), float(popt[1])
        except RuntimeError:
            pass
    # Default heuristic: ball at 0.5 m at t=0, slightly downward
    return 0.5, -1.0


# ---------------------------------------------------------------------------
# Trajectory reconstruction
# ---------------------------------------------------------------------------

class TrajectoryReconstructor:
    """
    Reconstructs the 3-D bowling ball trajectory from per-frame detections.

    Usage
    -----
    rec = TrajectoryReconstructor(calibrator)
    for frame_idx, det in enumerate(detections):
        rec.add_detection(frame_idx, det, fps)
    traj = rec.get_trajectory()   # list of dicts with X,Y,Z,t,speed,…
    """

    def __init__(self, calibrator):
        """
        Parameters
        ----------
        calibrator : LaneCalibrator instance (must have K and H_img2lane set)
        """
        self.cal = calibrator
        self._raw = []   # list of (frame_idx, u, v, r)

    def add_detection(self, frame_idx: int, det: dict, fps: float):
        """Add a single-frame ball detection."""
        if det is None:
            return
        self._raw.append({
            'frame': frame_idx,
            't':     frame_idx / fps,
            'u':     float(det['x']),
            'v':     float(det['y']),
            'r':     float(det['r']),
        })

    def reconstruct(self, fps: float) -> List[dict]:
        """
        Reconstruct 3-D trajectory from all stored detections.

        Returns list of dicts:
            frame, t, u, v, r,
            X (lane right), Y (lane forward), Z (height),
            speed_2d, speed_3d, hook_angle_deg
        """
        if not self._raw:
            return []

        points = []
        for obs in self._raw:
            u, v = obs['u'], obs['v']
            lane_xy = self.cal.pixel_to_lane(u, v)
            if lane_xy is None:
                continue
            points.append({**obs, 'X': float(lane_xy[0]), 'Y': float(lane_xy[1])})

        if not points:
            return []

        # ---- Height: parabolic fit -----------------------------------------
        times = np.array([p['t'] for p in points])
        # We assume Z=0 at the start (release) and aim for Z≈0 at the end.
        # Without depth info, use a simple heuristic model.
        Z0, vz = fit_height_parabola(times)
        for p in points:
            p['Z'] = float(_parabola(p['t'] - times[0], Z0, vz))
            p['Z'] = max(0.0, p['Z'])   # ball can't go below the lane

        # ---- Smooth XY with Savitzky-Golay (lecture: noise reduction) ------
        if len(points) >= 7:
            wl = min(7, len(points) if len(points) % 2 == 1 else len(points) - 1)
            Xs = savgol_filter([p['X'] for p in points], window_length=wl, polyorder=2)
            Ys = savgol_filter([p['Y'] for p in points], window_length=wl, polyorder=2)
            for i, p in enumerate(points):
                p['X_smooth'] = float(Xs[i])
                p['Y_smooth'] = float(Ys[i])
        else:
            for p in points:
                p['X_smooth'] = p['X']
                p['Y_smooth'] = p['Y']

        # ---- Velocity / speed / hook angle ---------------------------------
        dt = np.diff(times)
        dX = np.diff([p['X_smooth'] for p in points])
        dY = np.diff([p['Y_smooth'] for p in points])
        dZ = np.diff([p['Z'] for p in points])

        speeds_2d = np.hypot(dX, dY) / dt
        speeds_3d = np.sqrt(dX**2 + dY**2 + dZ**2) / dt

        for i, p in enumerate(points):
            if i < len(speeds_2d):
                p['speed_2d'] = float(speeds_2d[i])
                p['speed_3d'] = float(speeds_3d[i])
            else:
                p['speed_2d'] = 0.0
                p['speed_3d'] = 0.0

        # Hook angle: angle of final lateral displacement relative to Y axis
        if len(points) >= 2:
            total_dX = points[-1]['X_smooth'] - points[0]['X_smooth']
            total_dY = points[-1]['Y_smooth'] - points[0]['Y_smooth']
            hook_angle = np.degrees(np.arctan2(total_dX, total_dY))
        else:
            hook_angle = 0.0

        for p in points:
            p['hook_angle_deg'] = float(hook_angle)

        return points

    def get_summary(self, points: List[dict]) -> dict:
        """Compute summary statistics for the trajectory."""
        if not points:
            return {}
        speeds = [p['speed_3d'] for p in points if p['speed_3d'] > 0]
        return {
            'n_frames':        len(points),
            'duration_s':      points[-1]['t'] - points[0]['t'],
            'distance_m':      float(np.hypot(
                                    points[-1]['X_smooth'] - points[0]['X_smooth'],
                                    points[-1]['Y_smooth'] - points[0]['Y_smooth'])),
            'avg_speed_ms':    float(np.mean(speeds)) if speeds else 0.0,
            'max_speed_ms':    float(np.max(speeds))  if speeds else 0.0,
            'hook_angle_deg':  points[0].get('hook_angle_deg', 0.0),
            'peak_height_m':   float(max(p['Z'] for p in points)),
        }

    def save_csv(self, points: List[dict], path: str):
        """Save trajectory to CSV."""
        import csv
        if not points:
            return
        keys = ['frame', 't', 'u', 'v', 'r',
                'X', 'Y', 'Z', 'X_smooth', 'Y_smooth',
                'speed_2d', 'speed_3d', 'hook_angle_deg']
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(points)

    def save_json(self, points: List[dict], path: str):
        with open(path, 'w') as f:
            json.dump(points, f, indent=2)
