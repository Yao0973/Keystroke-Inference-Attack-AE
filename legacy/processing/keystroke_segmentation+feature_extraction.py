#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract keystroke features from PIN_*.csv (positive pulses only).

Output format:
  Column 1: PIN (e.g., "163589")
  Columns 2-7: Keystroke timestamps (seconds)
  Columns 8-37: 30 features = 6 keys × [E_pa, E_te, E_fwhm, E_rt, E_tc]

Assumptions:
  - Only positive current pulses are used (negative ignored)
  - Sampling rate is 10,000 Hz (adjust FS if needed)
"""

import numpy as np
from scipy.signal import find_peaks
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_ROOT = REPO_ROOT / "data" / "raw_signals"
DERIVED_DATA_ROOT = REPO_ROOT / "data" / "derived"

# ----------------------------
# Load raw signal (skip headers)
# ----------------------------
def load_raw_signal(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    segments_text = []
    current_seg = []
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            if current_seg:
                segments_text.append("\n".join(current_seg))
                current_seg = []
        else:
            current_seg.append(stripped)
    if current_seg:
        segments_text.append("\n".join(current_seg))

    signals = []
    for seg_text in segments_text:
        nums = []
        for line in seg_text.splitlines():
            parts = line.replace(',', ' ').split()
            if not parts:
                continue
            try:
                row_vals = [float(x) for x in parts]
                nums.extend(row_vals)
            except ValueError:
                continue  # Skip header/comment lines
        
        if len(nums) % 2 != 0:
            print(f"[WARN] Odd number of values in segment. Skipping.")
            continue
        if len(nums) == 0:
            continue

        times = np.array(nums[::2])
        currents = np.array(nums[1::2])
        signals.append((times, currents))
    
    return signals

# ----------------------------
# Envelope: simple smoothing or direct pass-through
# ----------------------------
def extract_envelope(times, currents, window_size=3):
    if len(currents) < window_size:
        return currents.copy()
    env = np.convolve(currents, np.ones(window_size)/window_size, mode='same')
    return env

# ----------------------------
# Segment keystrokes (positive peaks only)
# ----------------------------
def segment_keystrokes(times, envelope, fs=10000):
    min_dwell = int(0.02 * fs)  # 20 ms minimum spacing
    height_thresh = np.max(envelope) * 0.1 if np.max(envelope) > 0 else 0.0

    peaks, _ = find_peaks(
        envelope,
        height=height_thresh,
        distance=min_dwell
    )

    segments = []
    half_win_left = int(0.015 * fs)   # 15 ms before peak
    half_win_right = int(0.025 * fs)  # 25 ms after peak

    for pk in peaks[:6]:  # Take at most 6
        lk = max(0, pk - half_win_left)
        rk = min(len(envelope) - 1, pk + half_win_right)
        segments.append((lk, pk, rk))
    
    return segments

# ----------------------------
# Extract 5 features per key + timestamp
# ----------------------------
def extract_5_features_and_time(times, envelope, lk, pk, rk, fs=10000):
    dt = 1.0 / fs
    t_seg = times[lk:rk+1]
    e_seg = envelope[lk:rk+1]

    if len(e_seg) == 0:
        return 0.0, np.zeros(5)

    # Temporal centroid as timestamp
    if np.sum(e_seg) == 0:
        timestamp = t_seg[len(t_seg)//2]
    else:
        timestamp = np.sum(t_seg * e_seg) / np.sum(e_seg)

    E_pa = np.max(e_seg)
    E_te = np.sum(e_seg ** 2)

    # FWHM
    half_max = 0.5 * E_pa
    above_half = np.where(e_seg >= half_max)[0]
    if len(above_half) >= 2:
        E_fwhm = (above_half[-1] - above_half[0]) * dt
    elif len(above_half) == 1:
        E_fwhm = dt
    else:
        E_fwhm = 0.0

    # Rise time (10% to 90% on rising edge)
    rise_part = e_seg[:pk - lk + 1]
    if len(rise_part) < 2:
        E_rt = 0.0
    else:
        e10 = 0.1 * E_pa
        e90 = 0.9 * E_pa
        try:
            idx10 = np.where(rise_part >= e10)[0][0]
            idx90 = np.where(rise_part >= e90)[0][0]
            E_rt = (idx90 - idx10) * dt
        except IndexError:
            E_rt = 0.0

    E_tc = timestamp

    features = np.array([E_pa, E_te, E_fwhm, E_rt, E_tc], dtype=np.float64)
    return timestamp, features

# ----------------------------
# Main function
# ----------------------------
def main():
    RAW_FILE = RAW_DATA_ROOT / "PIN_163589.csv"
    OUTPUT_FILE = DERIVED_DATA_ROOT / "extracted_features_163589.csv"
    REAL_PIN = "163589"      # Bundled example PIN label used by the packaged trace.
    FS = 10000               # Sampling rate in Hz (confirmed from plot)
    TARGET_DIGITS = 6
    DERIVED_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # Load and filter signal (keep only positive current)
    signals = load_raw_signal(RAW_FILE)
    filtered_signals = []
    for t, c in signals:
        mask = c > 0
        if np.any(mask):
            filtered_signals.append((t[mask], c[mask]))
    signals = filtered_signals

    if not signals:
        print("[ERROR] No positive current segments found!")
        sys.exit(1)

    all_timestamps = []
    all_features = []

    for times, currents in signals:
        env = extract_envelope(times, currents, window_size=3)
        segments = segment_keystrokes(times, env, fs=FS)
        for (lk, pk, rk) in segments:
            ts, feat = extract_5_features_and_time(times, env, lk, pk, rk, fs=FS)
            all_timestamps.append(ts)
            all_features.append(feat)

    # Ensure exactly 6 digits
    if len(all_features) < TARGET_DIGITS:
        print(f"[WARN] Only {len(all_features)} digits detected. Padding to 6.")
        while len(all_features) < TARGET_DIGITS:
            all_timestamps.append(0.0)
            all_features.append(np.zeros(5))
    elif len(all_features) > TARGET_DIGITS:
        print(f"[WARN] {len(all_features)} digits detected. Truncating to 6.")
        all_timestamps = all_timestamps[:TARGET_DIGITS]
        all_features = all_features[:TARGET_DIGITS]

    # Flatten
    timestamps_flat = [float(t) for t in all_timestamps]  # 6 values
    features_flat = np.concatenate(all_features).tolist() # 30 values

    # Build output row
    row = [REAL_PIN] + timestamps_flat + features_flat
    assert len(row) == 37, f"Expected 37 columns, got {len(row)}"

    # Save
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"[SUCCESS] Saved to {OUTPUT_FILE.relative_to(REPO_ROOT)}")
    print(f"Columns: 1 (PIN) + 6 (timestamps) + 30 (features) = 37")

if __name__ == "__main__":
    main()
