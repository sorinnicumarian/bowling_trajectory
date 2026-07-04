"""
main.py
-------
Main pipeline for bowling ball analysis (IACV Project F11).

Usage:
    python src/main.py [--video data/video/pba_slowmo.mp4]
                       [--fps-override 60]
                       [--start-frame 0] [--max-frames 300]
                       [--calib-pts data/camera_params/control_points.json]
                       [--debug]
                       [--no-video]

All processing is single-pass to keep memory usage low.
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ball_detector    import BallDetector, draw_detection, reset_detector
from lane_calibration import LaneCalibrator
from trajectory_3d    import TrajectoryReconstructor
from spin_estimator   import SpinEstimator
from visualization    import (plot_lane_topdown, plot_trajectory_3d,
                               plot_spin_timeseries, plot_summary)

# ---------------------------------------------------------------------------
BASE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# Use /tmp for all output to avoid Google Drive I/O timeouts.
# Copy results back to project folder after pipeline finishes.
RESULTS_DIR = '/tmp/bowling_results'
DET_DIR     = os.path.join(RESULTS_DIR, 'detections')
TRAJ_DIR    = os.path.join(RESULTS_DIR, 'trajectory')
SPIN_DIR    = os.path.join(RESULTS_DIR, 'spin')
CAM_DIR     = os.path.join(RESULTS_DIR, 'camera_params')

for d in [DET_DIR, TRAJ_DIR, SPIN_DIR, CAM_DIR]:
    os.makedirs(d, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video',        default=os.path.join(BASE_DIR, 'data', 'video', 'pba_slowmo.mp4'))
    p.add_argument('--fps-override', type=float, default=None)
    p.add_argument('--start-frame',  type=int,   default=0)
    p.add_argument('--max-frames',   type=int,   default=500,
                   help='Number of frames to process (default 500 ≈ 8 s @ 60fps)')
    p.add_argument('--calib-pts',    default=None)
    p.add_argument('--debug',        action='store_true')
    p.add_argument('--no-video',     action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    # Use /tmp copy if video is on a network drive (avoids Google Drive timeout)
    video_path = args.video
    if 'CloudStorage' in video_path or 'Google' in video_path:
        tmp_path = '/tmp/pba_slowmo.mp4'
        if not os.path.isfile(tmp_path):
            print(f"Copying video to /tmp for faster access …")
            import shutil
            shutil.copy2(video_path, tmp_path)
        video_path = tmp_path

    if not os.path.isfile(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    fps = args.fps_override or cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {W}x{H} @ {fps:.1f} fps  ({total} frames = {total/fps:.1f}s)")
    print(f"Processing frames {args.start_frame} … {args.start_frame + args.max_frames}")

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    # -----------------------------------------------------------------------
    # Objects
    # -----------------------------------------------------------------------
    reset_detector()
    detector      = BallDetector(min_radius=18, max_radius=110, max_jump_px=160)
    calibrator    = LaneCalibrator()
    reconstructor = TrajectoryReconstructor(calibrator)
    spin_est      = SpinEstimator(fps=fps)

    # Warm up MOG2 background model on 3 s of frames before start_frame
    if args.start_frame > 0:
        print("Warming up background model …")
        warmup_start = max(0, args.start_frame - int(3 * fps))
        cap_wu = cv2.VideoCapture(video_path)
        cap_wu.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
        warmup_frames = []
        while cap_wu.get(cv2.CAP_PROP_POS_FRAMES) < args.start_frame:
            ret, f = cap_wu.read()
            if not ret: break
            warmup_frames.append(f)
        cap_wu.release()
        detector.warm_up(warmup_frames)
        print(f"  Warmed up on {len(warmup_frames)} frames")

    calibrated       = False
    calib_done_frame = None

    # Manual control points
    control_points_px = None
    if args.calib_pts and os.path.isfile(args.calib_pts):
        with open(args.calib_pts) as f:
            raw = json.load(f)
        control_points_px = {tuple(e['lane']): tuple(e['pixel']) for e in raw}
        print(f"Loaded {len(control_points_px)} manual control points")

    # Output CSV writers
    det_csv_file = open(os.path.join(DET_DIR, 'detections.csv'), 'w', newline='')
    det_writer   = csv.writer(det_csv_file)
    det_writer.writerow(['frame', 'x', 'y', 'r', 'score'])

    # Optional annotated video writer
    vid_writer = None
    if not args.no_video:
        out_path = os.path.join(RESULTS_DIR, 'annotated.mp4')
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        vid_writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # -----------------------------------------------------------------------
    # Single-pass processing
    # -----------------------------------------------------------------------
    detections   = []
    spin_results = []
    prev_det     = None
    t0           = time.time()

    for local_idx in range(args.max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        global_fi = args.start_frame + local_idx

        # --- Ball detection -------------------------------------------------
        det = detector.detect(frame)
        detections.append(det)
        if det is not None:
            det_writer.writerow([global_fi, det['x'], det['y'],
                                  det['r'], det.get('score', '')])

        # --- Lane calibration (once, on first good detection) ---------------
        if not calibrated and det is not None:
            ok = calibrator.calibrate_from_frame(
                frame,
                control_points_px=control_points_px,
                ball_det=det,
                debug=args.debug)
            calibrated = ok
            calib_done_frame = global_fi
            if ok:
                calib_path = os.path.join(CAM_DIR, 'camera_params.json')
                calibrator.save(calib_path)
                K = calibrator.K
                print(f"\nCalibrated at frame {global_fi}  "
                      f"fx={K[0,0]:.0f} fy={K[1,1]:.0f}  "
                      f"cx={K[0,2]:.0f} cy={K[1,2]:.0f}")
                spin_est.K = K
                if args.debug and calibrator._debug_img is not None:
                    cv2.imwrite(os.path.join(DET_DIR, 'calib_debug.jpg'),
                                calibrator._debug_img)

        # --- Trajectory point -----------------------------------------------
        reconstructor.add_detection(global_fi, det, fps)

        # --- Spin estimation ------------------------------------------------
        spin_r = spin_est.process_frame(frame, det)
        spin_results.append(spin_r)

        # --- Annotated frame ------------------------------------------------
        if vid_writer is not None:
            ann = draw_detection(frame, det) if det else frame.copy()
            # Small info overlay
            n_det = sum(1 for d in detections if d is not None)
            cv2.putText(ann, f"Frame {global_fi}  det={n_det}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if spin_r:
                cv2.putText(ann, f"RPM: {spin_r['rpm']:.0f}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            vid_writer.write(ann)

        if local_idx % 60 == 0:
            n_det = sum(1 for d in detections if d is not None)
            elapsed = time.time() - t0
            fps_proc = (local_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{local_idx:4d}/{args.max_frames}]  "
                  f"detected={n_det}  {fps_proc:.1f} frames/s")

    cap.release()
    det_csv_file.close()
    if vid_writer:
        vid_writer.release()

    n_det = sum(1 for d in detections if d is not None)
    print(f"\nDetection: {n_det}/{len(detections)} frames with ball")

    # -----------------------------------------------------------------------
    # 3D Trajectory
    # -----------------------------------------------------------------------
    print("\n--- 3D trajectory ---")
    traj_points  = reconstructor.reconstruct(fps)
    traj_summary = reconstructor.get_summary(traj_points)
    print(f"  {len(traj_points)} points reconstructed")
    if traj_summary:
        avg_kmh = traj_summary['avg_speed_ms'] * 3.6
        print(f"  Duration:   {traj_summary['duration_s']:.2f} s")
        print(f"  Avg speed:  {traj_summary['avg_speed_ms']:.2f} m/s  ({avg_kmh:.1f} km/h)")
        print(f"  Hook:       {traj_summary['hook_angle_deg']:.1f}°")
        print(f"  Peak height:{traj_summary['peak_height_m']:.3f} m")

    reconstructor.save_csv(traj_points, os.path.join(TRAJ_DIR, 'trajectory.csv'))
    reconstructor.save_json(traj_points, os.path.join(TRAJ_DIR, 'trajectory.json'))

    # -----------------------------------------------------------------------
    # Spin summary
    # -----------------------------------------------------------------------
    spin_summary = spin_est.get_summary()
    spin_est.save_json(os.path.join(SPIN_DIR, 'spin.json'))
    print(f"\n--- Spin ---")
    print(f"  Avg: {spin_summary.get('avg_rpm', 0):.0f} RPM")
    print(f"  Max: {spin_summary.get('max_rpm', 0):.0f} RPM")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("\n--- Plots ---")
    if traj_points:
        plot_lane_topdown(traj_points,
                          output_path=os.path.join(TRAJ_DIR, 'topdown.png'))
        plot_trajectory_3d(traj_points,
                           output_path=os.path.join(TRAJ_DIR, 'trajectory_3d.png'))

    valid_spin = [r for r in spin_results if r is not None and r.get('rpm', 0) > 0]
    if valid_spin:
        plot_spin_timeseries(spin_results, fps,
                             output_path=os.path.join(SPIN_DIR, 'spin_timeseries.png'))

    plot_summary(traj_points, spin_summary, traj_summary,
                 output_path=os.path.join(RESULTS_DIR, 'summary.png'))

    print(f"\n=== Done in {time.time()-t0:.1f}s ===")
    print(f"Results (local) → {RESULTS_DIR}/")

    # Copy results back to project folder (Google Drive)
    import shutil
    project_results = os.path.join(BASE_DIR, 'results')
    try:
        if os.path.isdir(project_results):
            shutil.rmtree(project_results)
        shutil.copytree(RESULTS_DIR, project_results)
        print(f"Results copied → {project_results}/")
    except Exception as e:
        print(f"[WARN] Could not copy to project folder: {e}")
        print(f"       Results are still in {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
