"""
make_multi_throw_traj.py
------------------------
3D trajectory reconstruction for 6 throws, shown on a top-down lane view.

Depth estimation via ball apparent size:
    depth = f * R_ball / r_pixels        (pinhole model, known sphere)
Lateral position:
    x_cam = (u - cx) * depth / f
Lane Y = depth (camera ~at foul line, looking down the lane).
Lane X = x_cam + offset so first detection starts at plausible position.

This gives physically meaningful hook shapes even without full extrinsic
calibration.  Absolute XY positions are approximate (±20 cm); shape is correct.
"""

import cv2, numpy as np, os, sys, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
VIDEO_PATH = '/tmp/pba_slowmo.mp4'
OUT_PATH   = os.path.join(BASE_DIR, 'results', 'screenshots', 'shot6_multi_traj.png')

# Camera intrinsics (from IAC estimation on first frame)
FOCAL   = 2202.9    # px  (fx = fy)
CX      = 960.0     # px
CY      = 540.0     # px
R_BALL  = 0.108     # m  (half of 21.6 cm diameter)

# Standard lane dimensions
LANE_W  = 1.05      # m
LANE_L  = 18.29     # m
ARROW_Y = 4.57      # m from foul line

# ---- Same 6 throws as the detection grid ---------------------------------
THROWS = [
    ( 8,  7545,  8025, 'Throw 8'),
    ( 9,  9150,  9885, 'Throw 9'),
    (14, 13680, 14085, 'Throw 14'),
    (25, 32490, 33150, 'Throw 25'),
    (41, 52110, 52515, 'Throw 41'),
    (56, 67470, 67875, 'Throw 56'),
]

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# -------------------------------------------------------------------------
def detect_raw(cap, start, end):
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_x, roi_y0, roi_y1 = int(W * 0.45), int(H * 0.15), int(H * 0.90)

    mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50,
                                               detectShadows=False)
    warmup = max(0, start - int(3 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup)
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < start:
        ret, f = cap.read()
        if not ret: break
        mog2.apply(f[roi_y0:roi_y1, roi_x:])

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    raw = []
    for fi in range(start, end + 1):
        ret, frame = cap.read()
        if not ret: break
        roi  = frame[roi_y0:roi_y1, roi_x:]
        fg   = mog2.apply(roi)
        blur = cv2.bilateralFilter(roi, 9, 75, 75)
        hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        masks = [
            cv2.inRange(hsv, (0,   0,   0),   (180, 55, 75)),
            cv2.inRange(hsv, (90, 40,  40),   (145, 255, 255)),
            cv2.inRange(hsv, (0,  70,  70),   (12,  255, 255)),
            cv2.inRange(hsv, (168,70,  70),   (180, 255, 255)),
            cv2.inRange(hsv, (15, 70,  70),   (35,  255, 255)),
        ]
        cmask = masks[0]
        for m in masks[1:]: cmask = cv2.bitwise_or(cmask, m)
        kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, kern)
        mask  = cv2.bitwise_and(fg, cmask)
        conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in conts:
            area = cv2.contourArea(c)
            if area < 500: continue
            (cx2, cy2), r = cv2.minEnclosingCircle(c)
            if not (20 <= r <= 85): continue
            circ = area / (np.pi * r * r)
            if circ < 0.38: continue
            score = circ * area
            fx2 = cx2 + roi_x
            fy2 = cy2 + roi_y0
            if best is None or score > best['s']:
                best = {'fi': fi, 'u': fx2, 'v': fy2, 'r': r, 's': score}
        if best:
            raw.append(best)
    return raw, fps


def ransac_filter(raw, deg=2, thresh=30, iters=50):
    if len(raw) < 6:
        return raw
    pts = np.array([[d['fi'], d['u'], d['v']] for d in raw], dtype=float)
    t = pts[:, 0]; t0, ts = t.mean(), t.std() + 1e-6; tn = (t - t0) / ts
    best_mask = np.zeros(len(raw), bool)
    rng = np.random.default_rng(42)
    for _ in range(iters):
        idx = rng.choice(len(raw), deg + 2, replace=False)
        try:
            pu = np.polyfit(tn[idx], pts[idx, 1], deg)
            pv = np.polyfit(tn[idx], pts[idx, 2], deg)
        except Exception:
            continue
        err = np.sqrt((pts[:,1]-np.polyval(pu,tn))**2 +
                      (pts[:,2]-np.polyval(pv,tn))**2)
        mask = err < thresh
        if mask.sum() > best_mask.sum():
            best_mask = mask
    if best_mask.sum() >= deg + 1:
        pu = np.polyfit(tn[best_mask], pts[best_mask, 1], deg)
        pv = np.polyfit(tn[best_mask], pts[best_mask, 2], deg)
        err = np.sqrt((pts[:,1]-np.polyval(pu,tn))**2 +
                      (pts[:,2]-np.polyval(pv,tn))**2)
        best_mask = err < thresh
    return [raw[i] for i in range(len(raw)) if best_mask[i]]


def to_lane_coords(dets):
    """
    Convert pixel detections to lane metric coordinates.

    Strategy:
    - Sort detections by frame, smooth u/v/r with wide SG filter
    - Y_lane: use smoothed 1/r (depth proxy), linearly scaled to [0, visible_dist]
              where visible_dist = f*R*(1/r_min - 1/r_max)
    - X_lane: linear map u → [0, 1.05] using per-throw pixel extent of lane

    This avoids relying on noisy per-frame r for absolute depth;
    instead uses the monotone change in r across the throw.
    """
    from scipy.signal import savgol_filter

    if len(dets) < 4:
        return []

    dets = sorted(dets, key=lambda d: d['fi'])
    us = np.array([d['u'] for d in dets], dtype=float)
    vs = np.array([d['v'] for d in dets], dtype=float)
    rs = np.array([max(d['r'], 5) for d in dets], dtype=float)

    # Wide smoothing window
    def sg(arr, wl=15):
        wl = min(wl, len(arr))
        if wl % 2 == 0: wl -= 1
        if wl < 3: return arr
        return savgol_filter(arr, window_length=wl, polyorder=2)

    us = sg(us, 15)
    vs = sg(vs, 15)
    rs = np.clip(sg(rs, 15), 5, 200)

    # Y_lane: depth from smoothed radius (pinhole model: depth = f*R/r)
    # r decreases as ball moves away → 1/r increases → use as depth proxy
    ys_lane = FOCAL * R_BALL / rs   # metres from camera (≈ foul line)
    ys_lane = np.clip(ys_lane, 0.5, LANE_L)

    # X_lane: perspective-correct lateral position.
    # At depth d, lane width in pixels = LANE_W * f / d
    # x_cam = (u - cx) * d / f  [metres, camera frame]
    # x_lane = x_cam + offset   [offset estimated so median start = 0.5m]
    depths = FOCAL * R_BALL / rs
    x_cam_arr = (us - CX) * depths / FOCAL   # metres lateral in camera frame

    # Estimate camera lateral offset using median of first 10% of trajectory
    # Professional bowlers start near center-right of lane (X ≈ 0.4-0.7m)
    n_head = max(1, len(x_cam_arr) // 10)
    x_head = np.median(x_cam_arr[:n_head])
    # Assume ball starts at X=0.5m (center, conservative estimate)
    offset = 0.5 - x_head
    xs_lane = np.clip(x_cam_arr + offset, 0.0, LANE_W)  # physically constrained

    # Force Y to be monotonically increasing (ball moves forward only)
    result = []
    y_max = -np.inf
    for i in range(len(dets)):
        y = float(ys_lane[i])
        x = float(xs_lane[i])
        if y > y_max:
            y_max = y
            result.append({'x': x, 'y': y, 't': dets[i]['fi']})

    return result


def smooth(arr, window=7):
    if len(arr) < window:
        return arr
    from scipy.signal import savgol_filter
    wl = window if window % 2 == 1 else window - 1
    wl = min(wl, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    if wl < 3:
        return arr
    return savgol_filter(arr, window_length=wl, polyorder=2)


def draw_lane(ax):
    """Draw top-down lane outline, dots, arrows, pins."""
    # Lane outline
    ax.add_patch(mpatches.Rectangle((0, 0), LANE_W, LANE_L,
                                    fill=False, edgecolor='#8B6914',
                                    linewidth=2, zorder=1))
    ax.set_facecolor('#F5E6C8')

    # Approach dots (Y = 1.37 m)
    for x in np.linspace(0.1, LANE_W - 0.1, 7):
        ax.plot(x, 1.37, 'o', color='#8B6914', ms=4, zorder=2)

    # Arrow markers (Y = 4.57 m)
    for x in np.linspace(0.1, LANE_W - 0.1, 7):
        ax.plot(x, ARROW_Y, 'v', color='#8B6914', ms=5, zorder=2)

    # Pin positions (standard triangle)
    pin_x = [0.525,
             0.525 - 0.305/2, 0.525 + 0.305/2,
             0.525 - 0.305,   0.525,            0.525 + 0.305,
             0.525 - 0.305*3/2, 0.525 - 0.305/2, 0.525 + 0.305/2, 0.525 + 0.305*3/2]
    pin_y = [18.29 - 0.0,
             18.29 - 0.305, 18.29 - 0.305,
             18.29 - 0.610, 18.29 - 0.610, 18.29 - 0.610,
             18.29 - 0.915, 18.29 - 0.915, 18.29 - 0.915, 18.29 - 0.915]
    ax.scatter(pin_x, pin_y, s=60, color='white', edgecolors='#555',
               linewidths=1.5, zorder=3)

    # Foul line
    ax.axhline(0, color='red', linewidth=1.5, linestyle='--', alpha=0.6, zorder=2)
    ax.text(-0.02, 0, 'Foul\nline', ha='right', va='center',
            fontsize=7, color='red')


def main():
    if not os.path.isfile(VIDEO_PATH):
        cands = glob.glob(os.path.join(BASE_DIR, 'data', 'video', '*.mp4'))
        if cands:
            import shutil; shutil.copy2(cands[0], VIDEO_PATH)
        else:
            print("[ERROR] Video not found"); return

    cap = cv2.VideoCapture(VIDEO_PATH)
    all_trajs = []

    for tid, start, end, label in THROWS:
        print(f"  {label} … ", end='', flush=True)
        raw, fps = detect_raw(cap, start, end)
        inliers  = ransac_filter(raw)
        lane_pts = to_lane_coords(inliers)
        print(f"{len(raw)} raw → {len(inliers)} inliers → {len(lane_pts)} 3D pts")
        all_trajs.append((label, lane_pts))

    cap.release()

    # ---- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 14))
    draw_lane(ax)

    for (label, pts), col in zip(all_trajs, COLORS):
        if len(pts) < 3:
            continue
        xs = smooth(np.array([p['x'] for p in pts]))
        ys = smooth(np.array([p['y'] for p in pts]))

        # Clip to visible lane range
        mask = (ys >= 0) & (ys <= LANE_L) & (xs >= -0.2) & (xs <= LANE_W + 0.2)
        xs, ys = xs[mask], ys[mask]
        if len(xs) < 2:
            continue

        # Colour gradient along trajectory
        segs = [[(xs[i], ys[i]), (xs[i+1], ys[i+1])] for i in range(len(xs)-1)]
        alphas = np.linspace(0.3, 1.0, len(segs))
        lc = LineCollection(segs, colors=[col]*len(segs),
                            linewidths=2.5, alpha=0.85, zorder=5)
        ax.add_collection(lc)

        # Start dot + end arrow
        ax.plot(xs[0],  ys[0],  'o', color=col, ms=7, zorder=6)
        ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-3], ys[-3]),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2), zorder=6)

    # Legend
    handles = [mpatches.Patch(color=c, label=THROWS[i][3])
               for i, c in enumerate(COLORS)]
    ax.legend(handles=handles, loc='upper right', fontsize=8,
              framealpha=0.9, title='Throw', title_fontsize=8)

    ax.set_xlim(-0.15, LANE_W + 0.15)
    ax.set_ylim(-0.5, LANE_L + 0.5)
    ax.set_xlabel('Lane width (m)', fontsize=10)
    ax.set_ylabel('Distance from foul line (m)', fontsize=10)
    ax.set_title('Ball Trajectories — Top-Down Lane View\n(6 throws, depth from apparent ball size)',
                 fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, linewidth=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {OUT_PATH}")


if __name__ == '__main__':
    main()
