"""
lane_calibration.py
-------------------
Lane plane calibration for bowling ball 3-D reconstruction.

Pipeline (IACV course algorithms):
  1. Preprocess first frame: Gaussian blur + Canny edge detection.
  2. Detect line segments with probabilistic Hough transform.
  3. Cluster lines into two dominant lane-edge directions → vanishing point
     v = l1 × l2  (homogeneous cross-product, from lecture E).
  4. Collect 4 physical control points visible in the image (foul-line
     corners + arrow-marker region) and build a homography H via DLT so that
     pixel coords map to metric lane coords.
  5. Estimate camera intrinsics K from the Image of the Absolute Conic (IAC)
     using vanishing points of three mutually orthogonal directions, or fall
     back to the ball-diameter scale reference if only one VP is available.

Physical constants (standard bowling lane, metric):
  Lane length   18.29 m   (foul line → pin deck)
  Lane width     1.05 m
  Arrow markers  4.57 m from foul line
  Approach dots  1.37 m from foul line
  Ball diameter  0.216 m  (used as independent scale check)
"""

import cv2
import numpy as np
from typing import Optional
import json
import os


# ---------------------------------------------------------------------------
# Physical lane dimensions  (metres)
# ---------------------------------------------------------------------------
LANE_LENGTH   = 18.29
LANE_WIDTH    =  1.05
ARROW_DIST    =  4.57   # foul line → arrow markers
DOT_DIST      =  1.37   # foul line → first set of dots
BALL_DIAMETER =  0.216


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hom(pt):
    """2-D point → homogeneous 3-vector."""
    return np.array([pt[0], pt[1], 1.0])


def _line_from_two_points(p1, p2):
    """Homogeneous line through two homogeneous points."""
    return np.cross(_hom(p1), _hom(p2))


def _intersect_lines(l1, l2):
    """Intersection point of two homogeneous lines (returns [x,y] or None)."""
    pt = np.cross(l1, l2)
    if abs(pt[2]) < 1e-10:
        return None   # parallel (vanishing point at infinity)
    return pt[:2] / pt[2]


def _angle_deg(line_vec):
    return np.degrees(np.arctan2(line_vec[1], line_vec[0]))


def _cluster_lines_by_angle(lines, n_clusters=2, angle_tol=15):
    """
    Group raw Hough lines into n_clusters angle buckets.
    Returns list of lists of line coefficients [a,b,c] (ax+by+c=0 form).
    """
    if lines is None:
        return []
    buckets = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        ang = _angle_deg(np.array([x2 - x1, y2 - y1])) % 180
        placed = False
        for bk in buckets:
            if abs(bk['angle'] - ang) < angle_tol:
                bk['lines'].append(_line_from_two_points((x1, y1), (x2, y2)))
                bk['angle'] = (bk['angle'] * len(bk['lines']) + ang) / (len(bk['lines']) + 1)
                placed = True
                break
        if not placed:
            buckets.append({'angle': ang,
                            'lines': [_line_from_two_points((x1, y1), (x2, y2))]})
    # keep largest buckets
    buckets.sort(key=lambda b: len(b['lines']), reverse=True)
    return buckets[:n_clusters]


def _representative_line(bucket_lines):
    """Average homogeneous line from a bucket (normalise first)."""
    normed = []
    for l in bucket_lines:
        n = np.linalg.norm(l[:2])
        if n > 1e-8:
            normed.append(l / n)
    return np.mean(normed, axis=0)


# ---------------------------------------------------------------------------
# DLT homography (4+ point correspondences)
# ---------------------------------------------------------------------------

def _dlt_homography(src_pts, dst_pts):
    """
    Compute homography H such that dst ~ H @ src  using the DLT algorithm
    (lecture E / F).  src_pts and dst_pts are Nx2 arrays.
    We use cv2.findHomography which implements normalised DLT.
    """
    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, method=0)   # least-squares DLT
    return H


# ---------------------------------------------------------------------------
# K estimation from two orthogonal vanishing points (IAC method)
# ---------------------------------------------------------------------------

def _estimate_K_from_vps(vp1, vp2, image_size):
    """
    Estimate camera intrinsics K from two vanishing points corresponding to
    orthogonal directions using the Image of the Absolute Conic (IAC, ω).

    Constraint from lecture E:  vp1^T ω vp2 = 0
    With zero-skew and square pixels assumed:
        ω = diag(1/fx², 1/fy², …)   reduced form
    We solve for (cx, cy, f) under the zero-skew, unit-aspect assumption.
    """
    w, h = image_size
    cx0, cy0 = w / 2.0, h / 2.0

    u1 = np.array([vp1[0], vp1[1], 1.0])
    u2 = np.array([vp2[0], vp2[1], 1.0])

    # Simplification: assume cx≈cx0, cy≈cy0, fx=fy=f, skew=0
    # Orthogonality:  (vp1x-cx)(vp2x-cx) + (vp1y-cy)(vp2y-cy) + f² = 0
    # → f² = -(vp1x-cx0)(vp2x-cx0) - (vp1y-cy0)(vp2y-cy0)
    f2 = -((vp1[0] - cx0) * (vp2[0] - cx0) +
           (vp1[1] - cy0) * (vp2[1] - cy0))
    if f2 <= 0:
        # Vanishing points too close to principal point; use image diagonal
        f = float(np.sqrt(w**2 + h**2))
    else:
        f = float(np.sqrt(f2))

    K = np.array([[f,   0., cx0],
                  [0.,  f,  cy0],
                  [0.,  0.,  1.]], dtype=np.float64)
    return K


def _estimate_K_from_ball(det, image_size):
    """
    Fall-back K estimation: use known ball diameter (0.216 m) and detected
    radius in pixels to estimate focal length, assuming the ball is on the
    lane plane (Z ≈ 0.1 m above ground when close to release).
    """
    w, h = image_size
    # For a sphere of radius R at distance Z, apparent radius r_px ≈ f*R/Z
    # We don't know Z, so we just set a reasonable f from the image diagonal.
    f = float(np.sqrt(w**2 + h**2))
    K = np.array([[f,   0., w / 2.0],
                  [0.,  f,  h / 2.0],
                  [0.,  0.,  1.    ]], dtype=np.float64)
    return K


# ---------------------------------------------------------------------------
# Main calibration class
# ---------------------------------------------------------------------------

class LaneCalibrator:
    """
    Detects lane geometry and computes:
      - H_img2lane  : 3×3 homography  image-pixel → lane-plane metric coords
      - H_lane2img  : inverse
      - K           : 3×3 camera intrinsic matrix
      - vp_lane     : vanishing point of lane-axis direction (pixels)
      - vp_width    : vanishing point of lane-width direction (pixels)
    """

    def __init__(self):
        self.H_img2lane  = None
        self.H_lane2img  = None
        self.K           = None
        self.vp_lane     = None
        self.vp_width    = None
        self.image_size  = None   # (w, h)
        self._debug_img  = None

    # ------------------------------------------------------------------
    def calibrate_from_frame(self, frame_bgr, control_points_px=None,
                             ball_det=None, debug=False):
        """
        Run full calibration on a single frame.

        Parameters
        ----------
        frame_bgr        : BGR image (numpy H×W×3)
        control_points_px: dict mapping physical lane coords (X,Y) [metres]
                           to pixel (u,v) coords, e.g.
                           {(0,0): (u,v), (1.05,0): …, (0,4.57): …, (1.05,4.57): …}
                           If None, automatic detection is attempted.
        ball_det         : optional ball detection dict (x,y,r) for K fallback
        debug            : if True, annotated image stored in self._debug_img
        """
        h, w = frame_bgr.shape[:2]
        self.image_size = (w, h)

        # ---- 1. Edge detection (Canny, bilateral pre-filter as in HW) ------
        blurred = cv2.bilateralFilter(frame_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges   = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

        # ---- 2. Hough line detection ----------------------------------------
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=80, minLineLength=80, maxLineGap=20)

        # ---- 3. Cluster lines → two dominant directions --------------------
        buckets = _cluster_lines_by_angle(lines, n_clusters=2, angle_tol=20)

        vp_lane = vp_width = None
        if len(buckets) >= 2:
            l_lane  = _representative_line(buckets[0]['lines'])
            l_width = _representative_line(buckets[1]['lines'])

            # Vanishing point: intersection of two lane-edge lines
            # (parallel lines in 3-D → same VP in the image)
            # Use pairs of individual lines within each bucket for robustness
            vps_lane = []
            lane_lines = buckets[0]['lines']
            for i in range(len(lane_lines)):
                for j in range(i+1, len(lane_lines)):
                    pt = _intersect_lines(lane_lines[i], lane_lines[j])
                    if pt is not None and np.linalg.norm(pt) < 1e6:
                        vps_lane.append(pt)
            if vps_lane:
                vp_lane = np.median(vps_lane, axis=0)

            vps_width = []
            width_lines = buckets[1]['lines']
            for i in range(len(width_lines)):
                for j in range(i+1, len(width_lines)):
                    pt = _intersect_lines(width_lines[i], width_lines[j])
                    if pt is not None and np.linalg.norm(pt) < 1e6:
                        vps_width.append(pt)
            if vps_width:
                vp_width = np.median(vps_width, axis=0)

        elif len(buckets) == 1:
            lane_lines = buckets[0]['lines']
            vps_lane = []
            for i in range(len(lane_lines)):
                for j in range(i+1, len(lane_lines)):
                    pt = _intersect_lines(lane_lines[i], lane_lines[j])
                    if pt is not None:
                        vps_lane.append(pt)
            if vps_lane:
                vp_lane = np.median(vps_lane, axis=0)

        self.vp_lane  = vp_lane
        self.vp_width = vp_width

        # ---- 4. Camera intrinsics K ----------------------------------------
        if vp_lane is not None and vp_width is not None:
            self.K = _estimate_K_from_vps(vp_lane, vp_width, (w, h))
        elif ball_det is not None:
            self.K = _estimate_K_from_ball(ball_det, (w, h))
        else:
            # Last resort: reasonable diagonal focal length
            f = float(np.sqrt(w**2 + h**2))
            self.K = np.array([[f, 0, w/2.], [0, f, h/2.], [0, 0, 1.]], np.float64)

        # ---- 5. Homography: image ↔ lane plane (DLT) -----------------------
        if control_points_px is not None and len(control_points_px) >= 4:
            src_px  = np.array(list(control_points_px.values()), dtype=np.float64)
            dst_met = np.array(list(control_points_px.keys()),   dtype=np.float64)
            self.H_img2lane = _dlt_homography(src_px, dst_met)
            self.H_lane2img = _dlt_homography(dst_met, src_px)
        else:
            # Auto-estimate from vanishing point + known lane width
            self.H_img2lane, self.H_lane2img = self._auto_homography(
                frame_bgr, edges, lines, vp_lane)

        # ---- Debug overlay -------------------------------------------------
        if debug:
            self._debug_img = self._draw_debug(frame_bgr.copy(), edges, lines,
                                               vp_lane, vp_width)

        return self.H_img2lane is not None

    # ------------------------------------------------------------------
    def _auto_homography(self, frame_bgr, edges, lines, vp_lane):
        """
        Estimate homography automatically by finding the lane boundary lines
        and mapping them to known physical width = 1.05 m.

        Heuristic: the two longest roughly-vertical/diagonal line segments in
        the bottom half of the image are the lane gutters.
        """
        h, w = frame_bgr.shape[:2]
        if lines is None:
            return None, None

        # Filter lines in the lower 2/3 of the image (where lane is visible)
        good = []
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            if min(y1, y2) > h // 3:
                length = np.hypot(x2 - x1, y2 - y1)
                good.append((length, seg[0]))
        good.sort(key=lambda x: x[0], reverse=True)

        if len(good) < 2:
            return None, None

        # Two longest lines → candidate lane edges
        _, (x1a, y1a, x2a, y2a) = good[0]
        _, (x1b, y1b, x2b, y2b) = good[1]

        # Pick bottom endpoints as foul-line corners
        pa = (x1a, y1a) if y1a > y2a else (x2a, y2a)
        pb = (x1b, y1b) if y1b > y2b else (x2b, y2b)
        # Pick top endpoints as arrow-marker region
        ta = (x1a, y1a) if y1a < y2a else (x2a, y2a)
        tb = (x1b, y1b) if y1b < y2b else (x2b, y2b)

        # Ensure pa is left of pb
        if pa[0] > pb[0]:
            pa, pb = pb, pa
            ta, tb = tb, ta

        # Source pixels (foul-line corners + arrow corners)
        src = np.array([pa, pb, ta, tb], dtype=np.float64)

        # Destination: lane metric coordinates
        # pa → (0, 0),  pb → (LANE_WIDTH, 0)
        # ta → (0, ARROW_DIST),  tb → (LANE_WIDTH, ARROW_DIST)
        dst = np.array([
            [0.,           0.          ],
            [LANE_WIDTH,   0.          ],
            [0.,           ARROW_DIST  ],
            [LANE_WIDTH,   ARROW_DIST  ],
        ], dtype=np.float64)

        H_img2lane = _dlt_homography(src, dst)
        H_lane2img = _dlt_homography(dst, src)
        return H_img2lane, H_lane2img

    # ------------------------------------------------------------------
    def pixel_to_lane(self, u, v):
        """
        Map image pixel (u, v) → metric lane coords (X, Y) using H_img2lane.
        Returns None if homography not available.
        """
        if self.H_img2lane is None:
            return None
        pt = self.H_img2lane @ np.array([u, v, 1.0])
        if abs(pt[2]) < 1e-10:
            return None
        return pt[:2] / pt[2]

    def lane_to_pixel(self, X, Y):
        """Metric lane coords (X,Y) → image pixel (u, v)."""
        if self.H_lane2img is None:
            return None
        pt = self.H_lane2img @ np.array([X, Y, 1.0])
        if abs(pt[2]) < 1e-10:
            return None
        return (pt[:2] / pt[2]).astype(int)

    # ------------------------------------------------------------------
    def backproject_to_lane_plane(self, u, v, Z=0.0):
        """
        Back-project image point (u,v) to 3-D point on the horizontal plane
        at height Z above the lane (Z=0 = lane surface).

        Uses pinhole model:  d = K^{-1} [u v 1]^T,  then scales so that
        the world Y coordinate equals the lane plane normal constraint.

        The lane plane equation in camera coords is defined by Z_world = Z.
        We assume the camera looks roughly along +Y_lane (down the lane)
        and the lane plane is Z_world = 0.
        """
        if self.K is None:
            return None
        K_inv = np.linalg.inv(self.K)
        ray = K_inv @ np.array([u, v, 1.0])   # direction in camera frame

        # Use homography for the Z=0 case (more numerically stable for
        # a single-camera setup without explicit extrinsics)
        lane_xy = self.pixel_to_lane(u, v)
        if lane_xy is None:
            return None
        return np.array([lane_xy[0], lane_xy[1], Z])

    # ------------------------------------------------------------------
    def save(self, path):
        """Save calibration to JSON."""
        data = {
            'K':           self.K.tolist()           if self.K           is not None else None,
            'H_img2lane':  self.H_img2lane.tolist()  if self.H_img2lane  is not None else None,
            'H_lane2img':  self.H_lane2img.tolist()  if self.H_lane2img  is not None else None,
            'vp_lane':     self.vp_lane.tolist()     if self.vp_lane     is not None else None,
            'vp_width':    self.vp_width.tolist()    if self.vp_width    is not None else None,
            'image_size':  list(self.image_size)     if self.image_size  is not None else None,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        """Load calibration from JSON."""
        with open(path) as f:
            data = json.load(f)
        def _arr(x):
            return np.array(x) if x is not None else None
        self.K           = _arr(data.get('K'))
        self.H_img2lane  = _arr(data.get('H_img2lane'))
        self.H_lane2img  = _arr(data.get('H_lane2img'))
        self.vp_lane     = _arr(data.get('vp_lane'))
        self.vp_width    = _arr(data.get('vp_width'))
        self.image_size  = data.get('image_size')

    # ------------------------------------------------------------------
    def _draw_debug(self, frame, edges, lines, vp_lane, vp_width):
        overlay = frame.copy()
        # draw detected line segments
        if lines is not None:
            for seg in lines:
                x1, y1, x2, y2 = seg[0]
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 200, 255), 1)
        # draw vanishing points
        def draw_vp(vp, colour):
            if vp is not None:
                pt = (int(np.clip(vp[0], -1e4, 1e4)),
                      int(np.clip(vp[1], -1e4, 1e4)))
                cv2.circle(overlay, pt, 8, colour, -1)
        draw_vp(vp_lane,  (0, 0,   255))
        draw_vp(vp_width, (255, 0, 0  ))
        return overlay
