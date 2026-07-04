"""
run_all_throws.py
-----------------
Run the full bowling pipeline on every throw detected by scan_throws.py.

Workflow:
  1. Read data/throws.json  (produced by scan_throws.py)
  2. For each throw, call main.py with the correct --start-frame / --max-frames
  3. Save per-throw results under results/throw_<N>/
  4. After all throws, print a comparison table

Usage:
    python src/run_all_throws.py [--video data/video/pba_slowmo.mp4]
                                 [--throws data/throws.json]
                                 [--no-video]
                                 [--throws-ids 1,3,5]   # optional subset
"""

import argparse
import json
import os
import subprocess
import sys
import shutil

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video',      default=os.path.join(BASE_DIR, 'data', 'video', 'pba_slowmo.mp4'))
    p.add_argument('--throws',     default=os.path.join(BASE_DIR, 'data', 'throws.json'))
    p.add_argument('--no-video',   action='store_true', help='Skip annotated video output (faster)')
    p.add_argument('--throws-ids', default=None,
                   help='Comma-separated list of throw IDs to process (default: all)')
    p.add_argument('--fps-override', type=float, default=None)
    return p.parse_args()


def load_throws(path, ids_filter=None):
    with open(path) as f:
        throws = json.load(f)
    if ids_filter:
        keep = set(ids_filter)
        throws = [t for t in throws if t['throw_id'] in keep]
    return throws


def run_throw(throw, video_path, no_video, fps_override, results_root):
    tid        = throw['throw_id']
    start      = throw['start_frame']
    end        = throw['end_frame']
    max_frames = end - start + 1
    out_dir    = os.path.join(results_root, f'throw_{tid:02d}')

    print(f"\n{'='*60}")
    print(f"  Throw {tid}  frames {start}–{end}  ({throw['duration_s']:.1f}s)")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py'),
        '--video',       video_path,
        '--start-frame', str(start),
        '--max-frames',  str(max_frames),
    ]
    if no_video:
        cmd.append('--no-video')
    if fps_override:
        cmd += ['--fps-override', str(fps_override)]

    env = os.environ.copy()
    env['BOWLING_RESULTS_DIR'] = out_dir   # main.py picks this up if set

    # Patch main.py output dir via env var — use /tmp/<tid> then move
    tmp_out = f'/tmp/bowling_throw_{tid}'
    env['BOWLING_RESULTS_DIR'] = tmp_out

    ret = subprocess.run(cmd, env=env)

    # Move results from /tmp/bowling_results (main.py default) to per-throw dir
    tmp_default = '/tmp/bowling_results'
    if os.path.isdir(tmp_default):
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        shutil.copytree(tmp_default, out_dir)
        shutil.rmtree(tmp_default)
        print(f"  Results → {out_dir}")

    # Read summary
    traj_json = os.path.join(out_dir, 'trajectory', 'trajectory.json')
    spin_json = os.path.join(out_dir, 'spin', 'spin.json')
    summary   = {'throw_id': tid, 'start_t_s': throw['start_t_s'],
                 'duration_s': throw['duration_s']}

    if os.path.isfile(traj_json):
        with open(traj_json) as f:
            traj = json.load(f)
        # trajectory.json is a list of point dicts
        if isinstance(traj, list) and traj:
            summary['n_detections'] = len(traj)
            # speed from first/last point
            pts = [p for p in traj if p.get('X') is not None]
            if len(pts) >= 2:
                t_span = pts[-1]['t'] - pts[0]['t']
                dx = pts[-1].get('X', 0) - pts[0].get('X', 0)
                dy = pts[-1].get('Y', 0) - pts[0].get('Y', 0)
                dist = (dx**2 + dy**2) ** 0.5
                summary['avg_speed_ms'] = round(dist / t_span, 1) if t_span > 0 else 0
                # hook = lateral displacement
                lat = pts[-1].get('X', 0) - pts[0].get('X', 0)
                summary['hook_angle_deg'] = round(float(lat), 2)
            zvals = [p.get('Z', 0) for p in pts if p.get('Z') is not None]
            summary['peak_height_m'] = round(max(zvals), 3) if zvals else 0

    if os.path.isfile(spin_json):
        with open(spin_json) as f:
            spin = json.load(f)
        summary['avg_rpm'] = round(spin.get('avg_rpm', 0), 0)
        summary['max_rpm'] = round(spin.get('max_rpm', 0), 0)

    return summary


def print_table(summaries):
    print("\n" + "="*80)
    print("MULTI-THROW SUMMARY")
    print("="*80)
    hdr = f"{'ID':>3}  {'t_start':>8}  {'dur':>5}  {'dets':>5}  {'spd m/s':>8}  {'hook°':>6}  {'Z_peak':>7}  {'RPM':>5}"
    print(hdr)
    print("-"*80)
    for s in summaries:
        print(f"{s['throw_id']:>3}  "
              f"{s.get('start_t_s', '—'):>8}  "
              f"{s.get('duration_s', '—'):>5}  "
              f"{s.get('n_detections', '—'):>5}  "
              f"{s.get('avg_speed_ms', '—'):>8}  "
              f"{s.get('hook_angle_deg', '—'):>6}  "
              f"{s.get('peak_height_m', '—'):>7}  "
              f"{s.get('avg_rpm', '—'):>5}")
    print("="*80)

    # Save as JSON
    out = os.path.join(BASE_DIR, 'results', 'multi_throw_summary.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary saved → {out}")


def main():
    args = parse_args()

    if not os.path.isfile(args.throws):
        print(f"[ERROR] throws.json not found: {args.throws}")
        print("Run scan_throws.py first.")
        sys.exit(1)

    ids_filter = None
    if args.throws_ids:
        ids_filter = [int(x) for x in args.throws_ids.split(',')]

    throws = load_throws(args.throws, ids_filter)
    print(f"Processing {len(throws)} throws …")

    video_path = args.video
    if 'CloudStorage' in video_path or 'Google' in video_path:
        tmp_path = '/tmp/pba_slowmo.mp4'
        if not os.path.isfile(tmp_path):
            print("Copying video to /tmp …")
            shutil.copy2(video_path, tmp_path)
        video_path = tmp_path

    results_root = os.path.join(BASE_DIR, 'results')
    summaries    = []

    for throw in throws:
        s = run_throw(throw, video_path, args.no_video,
                      args.fps_override, results_root)
        summaries.append(s)

    print_table(summaries)


if __name__ == '__main__':
    main()
