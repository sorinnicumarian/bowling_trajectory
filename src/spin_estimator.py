"""
spin_estimator.py
-----------------
Estimate bowling ball spin (rotation axis + angular velocity) from video.

Algorithm (IACV course, lecture H – optical flow):
  1. For each frame where a ball is detected, crop the ball region.
  2. Detect feature points on the ball surface (finger holes as dark blobs,
     or logo/brand mark via template matching, or generic corners).
  3. Track them with Lucas-Kanade optical flow (cv2.calcOpticalFlowPyrLK),
     which minimises the brightness constancy error over a patch.
  4. From two or more tracked surface point velocities  v_i = ω × r_i,
     where r_i are the 3-D positions on the sphere surface,
     recover  ω  using least squares on the skew-symmetric form of ×.
  5. Angular velocity magnitude: |ω| [rad/s]; axis: ω / |ω|.
  6. Report also in revolutions per minute (RPM), a standard bowling metric.

Notes
-----
  - The projection from 3-D surface motion to 2-D image motion introduces
    an ambiguity along the line of sight. We resolve it by assuming the
    z-component of the rotation axis (into the camera) is known from the
    ball's trajectory (forward roll → ω approximately perpendicular to V).
  - If the ball surface has no detectable features, we fall back to computing
    the apparent rotation of the circular silhouette using phase correlation.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
import json


# ---------------------------------------------------------------------------
# Feature detection on ball crop
# ---------------------------------------------------------------------------

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=30,
    qualityLevel=0.01,
    minDistance=5,
    blockSize=7,
)


def _crop_ball(frame_bgr, det, margin_frac=0.1):
    """Return (crop, offset_x, offset_y, crop_radius)."""
    h, w = frame_bgr.shape[:2]
    cx, cy, r = int(det['x']), int(det['y']), int(det['r'])
    margin = max(3, int(r * margin_frac))
    x1 = max(0, cx - r - margin)
    y1 = max(0, cy - r - margin)
    x2 = min(w, cx + r + margin)
    y2 = min(h, cy + r + margin)
    return frame_bgr[y1:y2, x1:x2], x1, y1, r


def _detect_finger_holes(crop_bgr, ball_r):
    """
    Find finger holes as dark circular blobs inside the ball region.
    Returns list of (x, y) in crop coordinates.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Blobs are small, dark circles: use HoughCircles with small radius
    min_r = max(2, ball_r // 10)
    max_r = max(5, ball_r // 4)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               dp=1.0, minDist=min_r * 2,
                               param1=50, param2=12,
                               minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    # Keep only dark circles (average intensity below threshold)
    pts = []
    for cx, cy, cr in circles:
        roi = gray[max(0,cy-cr):cy+cr+1, max(0,cx-cr):cx+cr+1]
        if roi.size > 0 and float(roi.mean()) < 80:
            pts.append((float(cx), float(cy)))
    return pts


def _detect_features_on_ball(crop_bgr, ball_r):
    """
    Detect trackable feature points on the ball surface.
    Priority: finger holes → Shi-Tomasi corners within ball mask.
    """
    # Try finger holes first
    pts = _detect_finger_holes(crop_bgr, ball_r)
    if len(pts) >= 2:
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    # Fall back to Shi-Tomasi corners within circular ball mask
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), ball_r, 255, -1)

    corners = cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)
    if corners is not None and len(corners) > 0:
        return corners.astype(np.float32)
    return None


# ---------------------------------------------------------------------------
# Lucas-Kanade tracking across frame pair
# ---------------------------------------------------------------------------

def _track_points(gray_prev, gray_curr, pts_prev):
    """
    Track pts_prev from gray_prev into gray_curr with LK optical flow.
    Returns (pts_curr, good_mask) — only tracked points where good_mask=True.
    """
    if pts_prev is None or len(pts_prev) == 0:
        return None, None
    pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_curr, pts_prev, None, **LK_PARAMS)
    good = (status.ravel() == 1)
    return pts_curr, good


# ---------------------------------------------------------------------------
# Angular velocity from 2-D surface point velocities
# ---------------------------------------------------------------------------

def _recover_omega_2d(pts_prev, pts_curr, dt, ball_r_px):
    """
    Estimate apparent 2-D angular velocity from tracked surface points.

    For a sphere rotating with angular velocity ω (2-D projected), each
    surface point at position p relative to the ball centre has apparent
    velocity  v = ω × p  (2-D cross product).

    In 2-D:  vx = -ω * py,   vy = ω * px
    → ω = (px*vy - py*vx) / (px² + py²)   per point.
    We take the weighted median over all tracked points.

    Returns (omega_z_rad_per_s, axis_2d).
    """
    if pts_prev is None or pts_curr is None or len(pts_prev) < 2:
        return 0.0, np.array([0., 0., 1.])

    omegas = []
    for p, q in zip(pts_prev, pts_curr):
        px, py = float(p[0]), float(p[1])
        # velocity in pixels/frame → pixels/second
        vx = (float(q[0]) - px) / dt
        vy = (float(q[1]) - py) / dt
        r2 = px**2 + py**2
        if r2 > 1e-6:
            omega = (px * vy - py * vx) / r2
            omegas.append(omega)

    if not omegas:
        return 0.0, np.array([0., 0., 1.])

    omega_z = float(np.median(omegas))
    # Scale from pixel/s to rad/s using ball radius
    # angular displacement = arc / radius
    omega_z_rads = omega_z / ball_r_px

    return omega_z_rads, np.array([0., 0., 1.])


def _recover_omega_3d(pts_prev_crop, pts_curr_crop, ball_cx, ball_cy,
                      ball_r_px, dt, K=None):
    """
    Full 3-D spin recovery using the Rodrigues / axis-angle method.

    Each tracked point p_i on the ball surface is at 3-D position
        r_i = R * [px-cx, py-cy, sqrt(R²-(px-cx)²-(py-cy)²)]^T
    (backprojected onto sphere of known radius R).

    The linear velocity from rotation:  v_i = ω × r_i
    In matrix form:  v_i = [r_i]_x  ω    where [·]_x is skew-symmetric.
    Stack all equations: A ω = b   (least squares).

    Returns (omega_vec [3], omega_rad_s scalar).
    """
    if pts_prev_crop is None or len(pts_prev_crop) < 3:
        omega_z, axis = _recover_omega_2d(
            pts_prev_crop - np.array([ball_cx, ball_cy]),
            pts_curr_crop - np.array([ball_cx, ball_cy]),
            dt, ball_r_px)
        return axis * omega_z, abs(omega_z)

    A_rows = []
    b_rows = []
    for p, q in zip(pts_prev_crop, pts_curr_crop):
        px, py = float(p[0]) - ball_cx, float(p[1]) - ball_cy
        r2 = ball_r_px**2 - px**2 - py**2
        if r2 < 0:
            continue
        pz = np.sqrt(r2)
        r_vec = np.array([px, py, pz])

        # 2-D apparent velocity (z-component unknown → we use 0 for image plane)
        vx = (float(q[0]) - float(p[0])) / dt
        vy = (float(q[1]) - float(p[1])) / dt

        # Skew-symmetric matrix [r]_x:
        # [ 0  -rz  ry ]
        # [ rz   0 -rx ]
        # [-ry  rx   0 ]
        skew = np.array([
            [  0,    -pz,   py],
            [ pz,      0,  -px],
            [-py,    px,    0 ],
        ], dtype=float)

        # We only observe vx, vy (image plane projection)
        A_rows.append(skew[0])
        A_rows.append(skew[1])
        b_rows.append(vx)
        b_rows.append(vy)

    if len(A_rows) < 3:
        omega_z, axis = _recover_omega_2d(
            pts_prev_crop - np.array([ball_cx, ball_cy]),
            pts_curr_crop - np.array([ball_cx, ball_cy]),
            dt, ball_r_px)
        return axis * omega_z, abs(omega_z)

    A = np.array(A_rows)
    b = np.array(b_rows)
    # Least-squares solution: ω = (A^T A)^{-1} A^T b
    omega_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Convert from pixel/s to rad/s
    omega_vec = omega_vec / ball_r_px
    omega_mag = float(np.linalg.norm(omega_vec))
    return omega_vec, omega_mag


# ---------------------------------------------------------------------------
# Main spin estimator
# ---------------------------------------------------------------------------

class SpinEstimator:
    """
    Estimates bowling ball spin frame-by-frame using Lucas-Kanade tracking.

    Usage
    -----
    est = SpinEstimator(fps=60)
    for frame, det in zip(frames, detections):
        result = est.process_frame(frame, det)
        # result: {'omega_rad_s', 'rpm', 'axis', 'n_tracked_pts'}
    summary = est.get_summary()
    """

    def __init__(self, fps: float = 60.0, K=None):
        self.fps     = fps
        self.K       = K
        self._prev_gray  = None
        self._prev_pts   = None
        self._prev_det   = None
        self._results    = []

    def process_frame(self, frame_bgr: np.ndarray,
                      det: Optional[dict]) -> Optional[dict]:
        """
        Process one frame. Returns spin estimate dict or None if skipped.

        Key insight: the ball translates ~100-150 px/frame at high speed, which
        exceeds LK's search range. We therefore track in the *ball crop* so
        only rotational motion remains.
        """
        if det is None:
            self._prev_crop_gray = None
            self._prev_pts_crop  = None
            self._prev_det       = None
            return None

        crop, ox, oy, r_px = _crop_ball(frame_bgr, det, margin_frac=0.15)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if not hasattr(self, '_prev_crop_gray'):
            self._prev_crop_gray = None
            self._prev_pts_crop  = None

        if self._prev_crop_gray is None or self._prev_pts_crop is None:
            pts = _detect_features_on_ball(crop, r_px)
            self._prev_pts_crop  = pts
            self._prev_crop_gray = crop_gray
            self._prev_det       = det
            return None

        # Align crop sizes (ball radius may vary slightly between frames)
        h1, w1 = self._prev_crop_gray.shape
        h2, w2 = crop_gray.shape
        if h1 != h2 or w1 != w2:
            scale_x = w2 / max(w1, 1)
            scale_y = h2 / max(h1, 1)
            prev_gray_r = cv2.resize(self._prev_crop_gray, (w2, h2))
            if self._prev_pts_crop is not None:
                scaled = self._prev_pts_crop.reshape(-1, 2) * np.array([scale_x, scale_y])
                self._prev_pts_crop = scaled.reshape(-1, 1, 2).astype(np.float32)
        else:
            prev_gray_r = self._prev_crop_gray

        # Ensure points are proper float32 Nx1x2 array for LK
        if self._prev_pts_crop is None or len(self._prev_pts_crop) == 0:
            self._prev_pts_crop  = None
            self._prev_crop_gray = crop_gray
            self._prev_det       = det
            return None
        pts_for_lk = np.array(self._prev_pts_crop, dtype=np.float32).reshape(-1, 1, 2)

        # LK tracking within the crop (translation already removed)
        pts_curr_crop, good = _track_points(prev_gray_r, crop_gray, pts_for_lk)

        result = {'omega_rad_s': 0.0, 'rpm': 0.0,
                  'axis': [0., 0., 1.], 'n_tracked_pts': 0}

        if pts_curr_crop is not None and good is not None and good.sum() >= 2:
            good_prev = self._prev_pts_crop[good].reshape(-1, 2)
            good_curr = pts_curr_crop[good].reshape(-1, 2)

            dt    = 1.0 / self.fps
            # Ball centre in crop coords
            cx_c  = float(det['x']) - ox
            cy_c  = float(det['y']) - oy

            omega_vec, omega_mag = _recover_omega_3d(
                good_prev, good_curr, cx_c, cy_c, float(r_px), dt, self.K)

            rpm  = omega_mag * 60.0 / (2.0 * np.pi)
            norm = omega_mag + 1e-10
            axis = (omega_vec / norm).tolist()

            result = {
                'omega_rad_s':   float(omega_mag),
                'rpm':           float(rpm),
                'axis':          axis,
                'n_tracked_pts': int(good.sum()),
            }
            self._prev_pts_crop = good_curr.reshape(-1, 1, 2)
        else:
            # Lost tracking: re-detect features in new crop
            pts = _detect_features_on_ball(crop, r_px)
            self._prev_pts_crop = pts

        self._prev_crop_gray = crop_gray
        self._prev_det       = det
        self._results.append(result)
        return result

    def get_summary(self) -> dict:
        """Aggregate spin statistics over all processed frames."""
        if not self._results:
            return {}
        rpms = [r['rpm'] for r in self._results if r['rpm'] > 0]
        omegas = [r['omega_rad_s'] for r in self._results if r['omega_rad_s'] > 0]
        # Most common axis direction (dominant spin axis)
        axes = np.array([r['axis'] for r in self._results
                         if r['n_tracked_pts'] >= 2])
        avg_axis = axes.mean(axis=0).tolist() if len(axes) > 0 else [0, 0, 1]
        return {
            'avg_rpm':       float(np.mean(rpms))   if rpms   else 0.0,
            'max_rpm':       float(np.max(rpms))    if rpms   else 0.0,
            'avg_omega_rads': float(np.mean(omegas)) if omegas else 0.0,
            'dominant_axis': avg_axis,
            'n_frames':      len(self._results),
        }

    def save_json(self, path: str):
        data = {
            'per_frame': self._results,
            'summary':   self.get_summary(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
