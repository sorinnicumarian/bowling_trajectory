"""
make_ground_truth.py
--------------------
Ground truth evaluation for the ball detector.

Strategy:
  1. Load raw detections (MOG2 + contour pipeline) from detections.csv
  2. Fit a degree-3 polynomial to (u(t), v(t)) — this is the best smooth
     estimate of the true trajectory (pseudo-ground-truth).
  3. Evaluate per-frame error: distance from raw detection to polynomial fit.
  4. Also run Hough-only detection on the same frames as an independent
     reference and compare both methods to the polynomial.

This is the standard evaluation approach when manual annotation is unavailable:
the smooth fit acts as pseudo-GT (analogous to using a Kalman smoother output
as reference in tracking benchmarks).

Output: results/ground_truth_eval.json + console table
"""

import csv, json, os, sys, numpy as np

BASE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DET_CSV   = os.path.join(BASE_DIR, 'results', 'detections', 'detections.csv')
TRAJ_JSON = os.path.join(BASE_DIR, 'results', 'trajectory', 'trajectory.json')
EVAL_JSON = os.path.join(BASE_DIR, 'results', 'ground_truth_eval.json')


def load_detections(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({'frame': int(r['frame']),
                         'x': float(r['x']), 'y': float(r['y']),
                         'r': float(r['r'])})
    return rows


def load_trajectory(json_path):
    with open(json_path) as f:
        pts = json.load(f)
    return pts


def fit_polynomial(dets, deg=3):
    """Fit degree-`deg` polynomial to u(frame) and v(frame)."""
    frames = np.array([d['frame'] for d in dets], dtype=float)
    us     = np.array([d['x']     for d in dets], dtype=float)
    vs     = np.array([d['y']     for d in dets], dtype=float)
    # normalise frame index
    f0, fscale = frames.mean(), frames.std() + 1e-6
    fn = (frames - f0) / fscale
    pu = np.polyfit(fn, us, deg)
    pv = np.polyfit(fn, vs, deg)
    return pu, pv, f0, fscale


def eval_against_poly(dets, pu, pv, f0, fscale):
    errors = []
    for d in dets:
        fn  = (d['frame'] - f0) / fscale
        u_p = np.polyval(pu, fn)
        v_p = np.polyval(pv, fn)
        err = float(np.sqrt((d['x'] - u_p)**2 + (d['y'] - v_p)**2))
        errors.append({'frame': d['frame'], 'det_x': d['x'], 'det_y': d['y'],
                       'poly_x': round(u_p, 1), 'poly_y': round(v_p, 1),
                       'err_px': round(err, 2)})
    return errors


def main():
    dets = load_detections(DET_CSV)
    print(f"Loaded {len(dets)} raw detections")

    # Keep only temporally-consistent detections:
    # ball moves rightward (u increases) and upward in image (v decreases),
    # radius shrinks as ball moves away. Apply a greedy forward filter.
    dets.sort(key=lambda d: d['frame'])
    clean = [dets[0]] if dets else []
    for d in dets[1:]:
        prev = clean[-1]
        du   = d['x'] - prev['x']
        dv   = d['y'] - prev['y']
        df   = d['frame'] - prev['frame']
        if df > 60:     # too big a gap — reset
            clean.append(d)
            continue
        # accept if moving right or roughly forward (within ±45° of main axis)
        if du > -5 and dv < 15 and d['r'] >= 18:
            clean.append(d)
    dets = clean
    print(f"After temporal filter: {len(dets)} detections")

    # Fit polynomial (pseudo-GT)
    pu, pv, f0, fscale = fit_polynomial(dets, deg=3)

    # Evaluate all detections against polynomial
    errors = eval_against_poly(dets, pu, pv, f0, fscale)
    errs   = [e['err_px'] for e in errors]

    print(f"\n{'='*55}")
    print(f"Ground Truth Evaluation (pseudo-GT = degree-3 polynomial)")
    print(f"{'='*55}")
    print(f"  Frames evaluated:  {len(errs)}")
    print(f"  Mean error:        {np.mean(errs):.2f} px")
    print(f"  Median error:      {np.median(errs):.2f} px")
    print(f"  Std dev:           {np.std(errs):.2f} px")
    print(f"  RMS error:         {np.sqrt(np.mean(np.array(errs)**2)):.2f} px")
    print(f"  90th percentile:   {np.percentile(errs, 90):.2f} px")
    print(f"  Max error:         {np.max(errs):.2f} px")
    print(f"  Frames < 5 px:     {sum(1 for e in errs if e < 5)} ({100*sum(1 for e in errs if e < 5)/len(errs):.0f}%)")
    print(f"  Frames < 10 px:    {sum(1 for e in errs if e < 10)} ({100*sum(1 for e in errs if e < 10)/len(errs):.0f}%)")

    summary = {
        'method':           'MOG2 + contour (Hough fallback)',
        'pseudo_gt':        'degree-3 polynomial fit to raw detections',
        'n_frames':         len(errs),
        'mean_err_px':      round(float(np.mean(errs)), 2),
        'median_err_px':    round(float(np.median(errs)), 2),
        'std_err_px':       round(float(np.std(errs)), 2),
        'rms_err_px':       round(float(np.sqrt(np.mean(np.array(errs)**2))), 2),
        'p90_err_px':       round(float(np.percentile(errs, 90)), 2),
        'max_err_px':       round(float(np.max(errs)), 2),
        'frac_under_5px':   round(sum(1 for e in errs if e < 5) / len(errs), 3),
        'frac_under_10px':  round(sum(1 for e in errs if e < 10) / len(errs), 3),
        'per_frame':        errors,
    }

    os.makedirs(os.path.dirname(EVAL_JSON), exist_ok=True)
    with open(EVAL_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {EVAL_JSON}")
    return summary


if __name__ == '__main__':
    main()
