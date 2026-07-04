"""
make_multi_throw_grid.py
------------------------
Generate a 2x2 grid screenshot showing ball trajectory overlaid on a
mid-throw frame for 4 different throws. Saved as results/screenshots/shot5_multi_throw.jpg.

Usage:
    python src/make_multi_throw_grid.py
"""

import cv2
import numpy as np
import os, sys, json

BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
VIDEO_PATH = '/tmp/pba_slowmo.mp4'
OUT_PATH   = os.path.join(BASE_DIR, 'results', 'screenshots', 'shot5_multi_throw.jpg')

# Six throws to show: (throw_id, start_frame, end_frame, label)
THROWS = [
    ( 8,  7545,  8025, 'Throw 8  (t=126s)'),
    (14, 13680, 14085, 'Throw 14 (t=228s)'),
    (22, 24525, 24885, 'Throw 22 (t=409s)'),
    (32, 36795, 37575, 'Throw 32 (t=614s)'),
    (38, 47910, 48735, 'Throw 38 (t=799s)'),
    (56, 67470, 67875, 'Throw 56 (t=1126s)'),
]
GRID_COLS = 3
GRID_ROWS = 2


def load_detections_from_video(cap, start, end):
    """Run a lightweight ball detector on a throw segment, return list of (frame, x, y, r)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ball_detector import BallDetector, reset_detector

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    reset_detector()
    det = BallDetector(min_radius=18, max_radius=110, max_jump_px=160)

    # warmup
    warmup_start = max(0, start - int(3 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    warmup = []
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < start:
        ret, f = cap.read()
        if not ret: break
        warmup.append(f)
    det.warm_up(warmup)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    points = []
    frames = {}
    for fi in range(start, end + 1):
        ret, frame = cap.read()
        if not ret: break
        d = det.detect(frame)
        if d is not None:
            points.append({'fi': fi, 'x': int(d['x']), 'y': int(d['y']), 'r': int(d['r'])})
        # Keep every 60th frame as candidate background
        if (fi - start) % 60 == 30:
            frames[fi] = frame.copy()
    return points, frames


def pick_mid_frame(points, frames):
    """Return the frame closest to the midpoint of the trajectory."""
    if not points:
        return None, None
    mid_idx = len(points) // 2
    mid_fi  = points[mid_idx]['fi']
    # find closest stored frame
    best = min(frames.keys(), key=lambda f: abs(f - mid_fi))
    return frames[best], mid_fi


def draw_panel(frame, points, label, mid_fi):
    """Draw trajectory trail on frame, return annotated BGR image."""
    out = frame.copy()
    H, W = out.shape[:2]

    # gradient path blue→yellow→red
    n = len(points)
    for i in range(1, n):
        t  = i / max(n - 1, 1)
        if t < 0.5:
            r, g, b = int(t * 2 * 255), int(t * 2 * 255), int(255 * (1 - t * 2))
        else:
            r, g, b = 255, int((1 - (t - 0.5) * 2) * 255), 0
        cv2.line(out,
                 (points[i-1]['x'], points[i-1]['y']),
                 (points[i  ]['x'], points[i  ]['y']),
                 (b, g, r), 4)

    # dots
    for p in points:
        cv2.circle(out, (p['x'], p['y']), 4, (255, 255, 255), -1)

    # start / end markers
    if points:
        cv2.circle(out, (points[0]['x'],  points[0]['y']),  8, (255, 80, 0),   2)
        cv2.circle(out, (points[-1]['x'], points[-1]['y']), 8, (0,  80, 255),  2)

    # label banner
    cv2.rectangle(out, (0, 0), (W, 36), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f'{len(points)} detections', (W - 200, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2, cv2.LINE_AA)
    return out


def main():
    if not os.path.isfile(VIDEO_PATH):
        # try copying from Google Drive
        import glob
        candidates = glob.glob(os.path.join(BASE_DIR, 'data', 'video', '*.mp4'))
        if candidates:
            import shutil
            print(f"Copying {candidates[0]} → {VIDEO_PATH} …")
            shutil.copy2(candidates[0], VIDEO_PATH)
        else:
            print("[ERROR] Video not found at /tmp/pba_slowmo.mp4")
            sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panels = []
    for tid, start, end, label in THROWS:
        print(f"\nProcessing {label} …")
        points, frames = load_detections_from_video(cap, start, end)
        print(f"  {len(points)} detections, {len(frames)} candidate frames")

        if not frames:
            # fallback: grab the first frame of the segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + (end - start) // 2)
            ret, frame = cap.read()
            frames = {start: frame} if ret else {}

        frame, mid_fi = pick_mid_frame(points, frames)
        if frame is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        panel = draw_panel(frame, points, label, mid_fi)
        panels.append(panel)

    cap.release()

    # Build 2x3 grid — scale each panel to same size first
    ph, pw = H // GRID_ROWS, W // GRID_COLS
    grid_rows = []
    for r in range(GRID_ROWS):
        row = []
        for c in range(GRID_COLS):
            idx = r * GRID_COLS + c
            p = cv2.resize(panels[idx], (pw, ph))
            row.append(p)
        grid_rows.append(np.hstack(row))
    grid = np.vstack(grid_rows)

    # Title bar
    title_h = 50
    title = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title, 'Ball Detection & Trajectory — 4 different throws from the same broadcast clip',
                (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    final = np.vstack([title, grid])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, final, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nSaved → {OUT_PATH}")


if __name__ == '__main__':
    main()
