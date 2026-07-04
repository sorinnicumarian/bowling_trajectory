"""
ball_detector.py
----------------
Bowling ball detection per video frame.

Two-stage approach (IACV course algorithms):
  1. Background subtraction (MOG2) to isolate moving foreground — removes
     static lane, pins, and advertisements in one step.
  2. Color segmentation in HSV to keep only ball-coloured pixels.
  3. Morphological open/close to clean the mask.
  4. Contour detection → minEnclosingCircle to find the ball blob.
  5. Fallback: cv2.HoughCircles on the colour-masked grayscale image when
     no good contour is found (e.g. first frames before background model
     is warmed up, or when ball is partially occluded).
  6. Temporal consistency: ROI around previous detection, reject large jumps.

The MOG2 model must be warmed up on ~3 s of footage before the throw.
Call `warm_up(frames)` on a BallDetector instance before detection.
"""

import cv2
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# HSV colour ranges for common bowling ball colours
# ---------------------------------------------------------------------------
BALL_HSV_RANGES = [
    # dark blue / indigo / purple  (most PBA balls)
    ((95,  30, 15),  (145, 255, 180)),
    # very dark (black ball, low saturation)
    ((0,   0,  0),   (180, 80,  70)),
    # red (two wrap-around ranges)
    ((0,   80, 80),  (12,  255, 255)),
    ((160, 80, 80),  (180, 255, 255)),
    # green
    ((38,  50, 50),  (82,  255, 255)),
    # orange / amber
    ((10,  100, 100), (25, 255, 255)),
]


def _colour_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in BALL_HSV_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


class BallDetector:
    """
    Stateful ball detector that maintains a MOG2 background model and
    temporal context across frames.
    """

    def __init__(self,
                 min_radius: int = 15,
                 max_radius: int = 100,
                 max_jump_px: int = 150,
                 mog2_history: int = 300,
                 mog2_threshold: float = 50.0):
        self.min_radius   = min_radius
        self.max_radius   = max_radius
        self.max_jump_px  = max_jump_px

        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history,
            varThreshold=mog2_threshold,
            detectShadows=False)
        self._prev: Optional[dict] = None
        self._warmed_up = False

    # ------------------------------------------------------------------
    def warm_up(self, frames):
        """Feed background frames to MOG2 without returning detections."""
        for f in frames:
            self._bg.apply(f)
        self._warmed_up = True

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """
        Detect ball in one frame. Returns dict(x, y, r, score) or None.
        Always feeds the frame into MOG2 (background model updates online).
        """
        h, w = frame_bgr.shape[:2]

        # --- Background subtraction ---
        fgmask = self._bg.apply(frame_bgr)
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  k_open)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, k_close)

        # --- Colour mask ---
        cmask = _colour_mask(frame_bgr)

        # --- Combined mask ---
        if self._warmed_up:
            combined = cv2.bitwise_and(fgmask, cmask)
        else:
            combined = cmask   # no reliable BG model yet

        combined = cv2.morphologyEx(combined,
                                    cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))

        # --- ROI around previous detection ---
        roi_mask = None
        if self._prev is not None:
            pad = self.max_jump_px
            roi_mask = np.zeros((h, w), np.uint8)
            x1 = max(0, self._prev['x'] - self._prev['r'] - pad)
            y1 = max(0, self._prev['y'] - self._prev['r'] - pad)
            x2 = min(w, self._prev['x'] + self._prev['r'] + pad)
            y2 = min(h, self._prev['y'] + self._prev['r'] + pad)
            roi_mask[y1:y2, x1:x2] = 255
            combined_roi = cv2.bitwise_and(combined, roi_mask)
        else:
            combined_roi = combined

        det = self._contour_detect(combined_roi, frame_bgr)

        # Fallback: full mask if ROI failed
        if det is None and roi_mask is not None:
            det = self._contour_detect(combined, frame_bgr)

        # Second fallback: Hough circles on colour mask only
        if det is None:
            det = self._hough_detect(frame_bgr)

        # Temporal consistency check
        if det is not None and self._prev is not None:
            dist = np.hypot(det['x'] - self._prev['x'],
                            det['y'] - self._prev['y'])
            if dist > self.max_jump_px:
                det = None   # implausible jump → reject

        if det is not None:
            self._prev = det
        return det

    # ------------------------------------------------------------------
    def _contour_detect(self, mask: np.ndarray,
                        frame_bgr: np.ndarray) -> Optional[dict]:
        """Find best circular contour in mask that matches ball constraints."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best = None
        h = frame_bgr.shape[0]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 400:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if not (self.min_radius <= r <= self.max_radius):
                continue
            # Reject top 15 % of frame (scoreboards/graphics)
            if cy < h * 0.15:
                continue
            circularity = area / (np.pi * r * r + 1e-6)
            if circularity < 0.40:
                continue
            score = circularity * area
            if best is None or score > best['score']:
                best = {'x': int(cx), 'y': int(cy), 'r': int(r),
                        'circularity': float(circularity),
                        'score': float(score), 'method': 'contour'}
        return best

    # ------------------------------------------------------------------
    def _hough_detect(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """Hough circle fallback on bilateral-filtered colour-masked image."""
        h = frame_bgr.shape[0]
        cmask = _colour_mask(frame_bgr)
        masked = cv2.bitwise_and(frame_bgr, frame_bgr, mask=cmask)
        blurred = cv2.bilateralFilter(masked, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=self.max_radius,
            param1=50, param2=20,
            minRadius=self.min_radius,
            maxRadius=self.max_radius)
        if circles is None:
            return None
        circles = np.round(circles[0]).astype(int)
        # Reject top 15% and sort by radius
        valid = [c for c in circles if c[1] > h * 0.15]
        if not valid:
            valid = list(circles)
        valid.sort(key=lambda c: -c[2])
        cx, cy, cr = valid[0]
        return {'x': int(cx), 'y': int(cy), 'r': int(cr),
                'circularity': 0.0, 'score': float(cr), 'method': 'hough'}

    # ------------------------------------------------------------------
    def reset(self):
        self._prev = None


# ---------------------------------------------------------------------------
# Convenience functions (backward-compatible with old API)
# ---------------------------------------------------------------------------

_default_detector: Optional[BallDetector] = None


def detect_ball_robust(frame_bgr: np.ndarray,
                       prev: Optional[dict] = None,
                       min_radius: int = 15,
                       max_radius: int = 100,
                       max_jump_px: int = 150,
                       **kwargs) -> Optional[dict]:
    """
    Stateless-ish wrapper around BallDetector for backward compatibility.
    Creates a module-level detector on first call.
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = BallDetector(
            min_radius=min_radius,
            max_radius=max_radius,
            max_jump_px=max_jump_px)
    _default_detector.min_radius  = min_radius
    _default_detector.max_radius  = max_radius
    _default_detector.max_jump_px = max_jump_px
    if prev is not None:
        _default_detector._prev = prev
    return _default_detector.detect(frame_bgr)


def reset_detector():
    global _default_detector
    _default_detector = None


def draw_detection(frame_bgr: np.ndarray, det: dict) -> np.ndarray:
    """Draw detection overlay on a copy of the frame."""
    out = frame_bgr.copy()
    if det is None:
        return out
    method_color = (0, 255, 0) if det.get('method') == 'contour' else (0, 200, 255)
    cv2.circle(out, (det['x'], det['y']), det['r'], method_color, 3)
    cv2.circle(out, (det['x'], det['y']), 4, (0, 0, 255), -1)
    label = f"r={det['r']}px  [{det.get('method','?')}]"
    cv2.putText(out, label,
                (det['x'] + det['r'] + 6, det['y']),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, method_color, 2)
    return out
