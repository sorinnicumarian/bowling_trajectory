"""
scan_throws.py
--------------
Scan the full video and auto-detect throw segments by motion analysis.

Strategy:
  1. Sample every N frames (default 15) to build a motion signal.
  2. Use inter-frame absolute difference in the lane ROI (bottom 70% of frame,
     right 60% to avoid the bowler's approach area on the left).
  3. Threshold the motion signal → binary "ball in motion" mask.
  4. Group consecutive active frames into candidate throws.
  5. Filter by minimum duration (>= 2 s of visible ball).

Output: JSON list of {throw_id, start_frame, end_frame, duration_s}
        printed to stdout and saved to data/throws.json.

Usage:
    python src/scan_throws.py [--video data/video/pba_slowmo.mp4]
                              [--sample-every 15]
                              [--motion-thresh 8]
                              [--min-throw-s 2.0]
                              [--gap-fill-s 1.0]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video',         default=os.path.join(BASE_DIR, 'data', 'video', 'pba_slowmo.mp4'))
    p.add_argument('--sample-every',  type=int,   default=15,
                   help='Sample one frame every N frames (speed vs accuracy tradeoff)')
    p.add_argument('--motion-thresh', type=float, default=8.0,
                   help='Mean absolute pixel difference to count as "motion"')
    p.add_argument('--min-throw-s',   type=float, default=2.0,
                   help='Minimum throw duration in seconds')
    p.add_argument('--gap-fill-s',    type=float, default=1.0,
                   help='Merge segments separated by less than this many seconds')
    p.add_argument('--out',           default=os.path.join(BASE_DIR, 'data', 'throws.json'))
    return p.parse_args()


def main():
    args = parse_args()

    video_path = args.video
    # Use /tmp copy if on network drive
    if 'CloudStorage' in video_path or 'Google' in video_path:
        tmp_path = '/tmp/pba_slowmo.mp4'
        if not os.path.isfile(tmp_path):
            print("Copying video to /tmp …")
            import shutil
            shutil.copy2(video_path, tmp_path)
        video_path = tmp_path

    if not os.path.isfile(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {W}x{H} @ {fps:.1f} fps  total={total} ({total/fps:.0f}s)")

    # Lane ROI: bottom 70%, right 55% — avoids scoreboard and bowler approach
    roi_x0 = int(W * 0.45)
    roi_y0 = int(H * 0.30)

    N          = args.sample_every
    gap_fill_f = int(args.gap_fill_s * fps / N)   # in sample units
    min_dur_f  = int(args.min_throw_s * fps / N)

    active_samples = []  # list of (sample_idx, frame_idx, is_active)

    prev_gray = None
    sample_idx = 0
    frame_idx  = 0

    print(f"Scanning every {N} frames …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % N == 0:
            roi   = frame[roi_y0:, roi_x0:]
            gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray  = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev_gray is not None:
                diff   = cv2.absdiff(gray, prev_gray).astype(np.float32)
                motion = float(np.mean(diff))
                active = motion > args.motion_thresh
                active_samples.append((sample_idx, frame_idx, active, motion))
            prev_gray = gray
            sample_idx += 1
            if sample_idx % 200 == 0:
                t = frame_idx / fps
                print(f"  {t:.0f}s / {total/fps:.0f}s …")
        frame_idx += 1

    cap.release()
    print(f"Scanned {sample_idx} samples from {frame_idx} frames")

    # -------------------------------------------------------------------------
    # Group active samples into throw segments
    # -------------------------------------------------------------------------
    # Binary activity array
    active_arr = np.array([s[2] for s in active_samples], dtype=bool)

    # Gap filling: merge short inactive gaps
    i = 0
    while i < len(active_arr):
        if active_arr[i]:
            j = i + 1
            while j < len(active_arr) and not active_arr[j]:
                j += 1
            gap = j - i - 1
            if gap <= gap_fill_f and j < len(active_arr):
                active_arr[i:j] = True
            i = j
        else:
            i += 1

    # Extract contiguous active runs
    throws_raw = []
    in_throw   = False
    seg_start  = 0
    for i, a in enumerate(active_arr):
        if a and not in_throw:
            seg_start = i
            in_throw  = True
        elif not a and in_throw:
            throws_raw.append((seg_start, i - 1))
            in_throw = False
    if in_throw:
        throws_raw.append((seg_start, len(active_arr) - 1))

    # Convert sample indices → frame indices
    def sample_to_frame(si):
        return active_samples[si][1]

    throws = []
    for si_start, si_end in throws_raw:
        f_start = sample_to_frame(si_start)
        f_end   = sample_to_frame(si_end)
        dur_s   = (f_end - f_start) / fps
        if (si_end - si_start) >= min_dur_f:
            throws.append({
                'throw_id':    len(throws) + 1,
                'start_frame': int(f_start),
                'end_frame':   int(f_end),
                'duration_s':  round(dur_s, 2),
                'start_t_s':   round(f_start / fps, 1),
            })

    print(f"\nFound {len(throws)} throw segments:")
    for t in throws:
        print(f"  Throw {t['throw_id']:2d}: "
              f"frame {t['start_frame']:6d}–{t['end_frame']:6d}  "
              f"t={t['start_t_s']:.1f}s  dur={t['duration_s']:.1f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(throws, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
