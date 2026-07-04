"""
make_multi_throw_grid.py  v3
----------------------------
Grid of 6 throws. Key improvements:
 - RANSAC polynomial filter: keeps only inliers within 25px of best-fit line
 - Lane-only ROI (right 55%, y 15-90%)
 - Display frame chosen as last frame where ball is clearly on-lane
 - Auto-selects best 6 throws by inlier count
"""

import cv2, numpy as np, os, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
VIDEO_PATH = '/tmp/pba_slowmo.mp4'
OUT_PATH   = os.path.join(BASE_DIR, 'results', 'screenshots', 'shot5_multi_throw.jpg')

CANDIDATES = [
    ( 8,  7545,  8025, 'Throw 8   t=126s'),
    ( 9,  9150,  9885, 'Throw 9   t=153s'),
    (14, 13680, 14085, 'Throw 14  t=228s'),
    (15, 14265, 14910, 'Throw 15  t=238s'),
    (25, 32490, 33150, 'Throw 25  t=542s'),
    (32, 36795, 37575, 'Throw 32  t=614s'),
    (35, 43140, 43620, 'Throw 35  t=720s'),
    (36, 46650, 47130, 'Throw 36  t=778s'),
    (51, 63165, 63525, 'Throw 51  t=1054s'),
    (56, 67470, 67875, 'Throw 56  t=1126s'),
]
GRID_COLS, GRID_ROWS = 3, 2


def detect_raw(cap, start, end):
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_x, roi_y0, roi_y1 = int(W * 0.45), int(H * 0.15), int(H * 0.90)

    mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
    warmup_start = max(0, start - int(3 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < start:
        ret, f = cap.read()
        if not ret: break
        mog2.apply(f[roi_y0:roi_y1, roi_x:])

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    raw, all_frames = [], {}

    for fi in range(start, end + 1):
        ret, frame = cap.read()
        if not ret: break

        roi  = frame[roi_y0:roi_y1, roi_x:]
        fg   = mog2.apply(roi)
        blur = cv2.bilateralFilter(roi, 9, 75, 75)
        hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        masks = [
            cv2.inRange(hsv, (0,   0,   0),   (180, 55, 75)),   # near-black
            cv2.inRange(hsv, (90, 40,  40),   (145, 255, 255)),  # blue
            cv2.inRange(hsv, (0,  70,  70),   (12,  255, 255)),  # red
            cv2.inRange(hsv, (168,70,  70),   (180, 255, 255)),  # red wrap
            cv2.inRange(hsv, (15, 70,  70),   (35,  255, 255)),  # orange/green
        ]
        cmask = masks[0]
        for m in masks[1:]: cmask = cv2.bitwise_or(cmask, m)
        kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, kern)
        mask  = cv2.bitwise_and(fg, cmask)

        conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in conts:
            area = cv2.contourArea(c)
            if area < 500: continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if not (20 <= r <= 85): continue
            circ = area / (np.pi * r * r)
            if circ < 0.38: continue
            score = circ * area
            fx, fy = cx + roi_x, cy + roi_y0
            if best is None or score > best['s']:
                best = {'fi': fi, 'x': fx, 'y': fy, 'r': r, 's': score}

        if best: raw.append(best)

        # Store every frame so we can pick the best display frame
        if (fi - start) % 10 == 5:
            all_frames[fi] = frame.copy()

    return raw, all_frames, W, H


def ransac_filter(raw, deg=2, thresh=30, iters=50):
    """Keep inliers of best polynomial fit via RANSAC."""
    if len(raw) < 6:
        return raw
    pts = np.array([[d['fi'], d['x'], d['y']] for d in raw], dtype=float)
    t   = pts[:, 0]
    t0, ts = t.mean(), t.std() + 1e-6
    tn  = (t - t0) / ts

    best_mask = np.zeros(len(raw), bool)
    rng = np.random.default_rng(42)
    for _ in range(iters):
        idx = rng.choice(len(raw), deg + 2, replace=False)
        try:
            pu = np.polyfit(tn[idx], pts[idx, 1], deg)
            pv = np.polyfit(tn[idx], pts[idx, 2], deg)
        except Exception:
            continue
        eu  = pts[:, 1] - np.polyval(pu, tn)
        ev  = pts[:, 2] - np.polyval(pv, tn)
        err = np.sqrt(eu**2 + ev**2)
        mask = err < thresh
        if mask.sum() > best_mask.sum():
            best_mask = mask

    # Refit on inliers and filter again
    if best_mask.sum() >= deg + 1:
        pu = np.polyfit(tn[best_mask], pts[best_mask, 1], deg)
        pv = np.polyfit(tn[best_mask], pts[best_mask, 2], deg)
        eu  = pts[:, 1] - np.polyval(pu, tn)
        ev  = pts[:, 2] - np.polyval(pv, tn)
        err = np.sqrt(eu**2 + ev**2)
        best_mask = err < thresh

    return [raw[i] for i in range(len(raw)) if best_mask[i]]


def pick_display_frame(inliers, all_frames):
    """
    Pick the frame with the HIGHEST-SCORE detection (most confident ball detection).
    Show that frame with the ball clearly circled = visual proof detection works.
    """
    if not inliers or not all_frames:
        return None, None
    # Best detection = highest score
    best_det = max(inliers, key=lambda p: p.get('s', p['r']))
    best_fi  = min(all_frames.keys(), key=lambda f: abs(f - best_det['fi']))
    return all_frames[best_fi], best_det


def draw_panel(frame, inliers, best_det, label, W, H):
    out = frame.copy()
    n   = len(inliers)

    # Draw short tail (last 20 detections before best) as context
    if n >= 2:
        tail = inliers[-min(20, n):]
        for i in range(1, len(tail)):
            t = i / max(len(tail) - 1, 1)
            b2 = int(255 * max(0, 1 - t * 2))
            g2 = int(200 * t)
            r2 = int(255 * min(1, t * 2))
            p1 = (int(tail[i-1]['x']), int(tail[i-1]['y']))
            p2 = (int(tail[i  ]['x']), int(tail[i  ]['y']))
            cv2.line(out, p1, p2, (b2, g2, r2), 3, cv2.LINE_AA)

    # Draw the best detection with a clear circle
    if best_det:
        bx, by, br = int(best_det['x']), int(best_det['y']), int(best_det['r'])
        cv2.circle(out, (bx, by), br + 4, (0, 0, 0),   4)   # black shadow
        cv2.circle(out, (bx, by), br + 4, (0, 255, 80), 3)   # green circle
        cv2.circle(out, (bx, by), 4,      (0, 255, 80), -1)  # centre dot

    cv2.rectangle(out, (0, 0), (W, 40), (15, 15, 15), -1)
    cv2.putText(out, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f'{n} det', (W - 115, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (120, 255, 120), 2, cv2.LINE_AA)
    return out


def main():
    if not os.path.isfile(VIDEO_PATH):
        cands = glob.glob(os.path.join(BASE_DIR, 'data', 'video', '*.mp4'))
        if cands:
            import shutil
            print(f"Copying {cands[0]} → {VIDEO_PATH} …")
            shutil.copy2(cands[0], VIDEO_PATH)
        else:
            print("[ERROR] Video not found"); return

    cap = cv2.VideoCapture(VIDEO_PATH)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scored = []
    for tid, start, end, label in CANDIDATES:
        print(f"  {label} … ", end='', flush=True)
        raw, frames, _, _ = detect_raw(cap, start, end)
        inliers = ransac_filter(raw)
        print(f"{len(raw)} raw → {len(inliers)} inliers")
        scored.append((len(inliers), inliers, frames, label, tid))

    cap.release()

    scored.sort(key=lambda x: -x[0])
    chosen = scored[:GRID_COLS * GRID_ROWS]
    chosen.sort(key=lambda x: x[4])  # chronological

    panels = []
    for n_in, inliers, frames, label, tid in chosen:
        frame, best_det = pick_display_frame(inliers, frames)
        if frame is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        panels.append(draw_panel(frame, inliers, best_det, label, W, H))

    ph, pw = H // GRID_ROWS, W // GRID_COLS
    rows = []
    for r in range(GRID_ROWS):
        rows.append(np.hstack([cv2.resize(panels[r*GRID_COLS+c], (pw, ph))
                                for c in range(GRID_COLS)]))
    grid = np.vstack(rows)

    title = np.zeros((54, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title,
        'Ball Detection  --  6 different throws from the same broadcast clip',
        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (210, 210, 210), 2, cv2.LINE_AA)
    final = np.vstack([title, grid])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, final, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"\nSaved → {OUT_PATH}")
    print("Chosen:", [(x[4], x[0]) for x in chosen])


if __name__ == '__main__':
    main()
