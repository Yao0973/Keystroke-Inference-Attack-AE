# plot_synthetic_signal.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import raw_signal_path

plt.rcParams["font.family"] = "Times New Roman"

def plot_keystroke_signal(csv_file, pin_sequence=None):
        # --- 1. Load data ---
    df = pd.read_csv(csv_file)
    t = df['Time(s)'].values
    s = df['CH1(V)'].values

    print(f"✅ Signal loaded: {csv_file}")
    print(f"   - Duration: {t[-1]:.2f} s")
    print(f"   - Sample rate: {len(t)/t[-1]:.0f} Hz")

    # --- 2. Calculate envelope (for annotation and visualization) ---
    analytic_signal = hilbert(s)
    amplitude_envelope = np.abs(analytic_signal)
    envelope_smooth = gaussian_filter1d(amplitude_envelope, sigma=100)

    # --- 3. Plotting ---
    plt.figure(figsize=(5.7, 3.3))


    # Original signal (simulating oscilloscope)
    plt.plot(t, s, color='#005cab', linewidth=0.7, alpha=1, label='Leakage Current')

    # Envelope (verify waveform shape)
    plt.plot(t, envelope_smooth, color='#e31b23', linewidth=2.7, alpha=1, linestyle='-', label='Signal Envelope')

    # --- 4. Auto-annotate keystroke positions ---
    if pin_sequence and len(pin_sequence) == 6:
        # Find regions where envelope exceeds threshold
        threshold = 0.2
        above_thresh = envelope_smooth > threshold

        # Extract starting points of continuous segments as keystroke centers
        changes = np.diff(above_thresh.astype(int))
        start_indices = np.where(changes == 1)[0] + 1

        # Limit to maximum 6 annotations
        key_centers = []
        last_t = -1.0
        for idx in start_indices:
            if t[idx] - last_t > 0.4 and len(key_centers) < len(pin_sequence):
                key_centers.append(t[idx])
                last_t = t[idx]

# Annotation loop
        for i, center_t in enumerate(key_centers):
            # === [Modification point] ===
            # Default position: 0.15s to the left of detected rising edge
            text_x = center_t - 0.15

            # Special handling: if it's '0', force shift to the right (move back)
            # Reason: '0' has low amplitude and may be affected by noise, causing detected center_t to be biased left
            if pin_sequence[i] == '6':
                text_x = center_t + 0.28  # Added 0.3s relative displacement here (-0.15 -> +0.15)
            
            plt.text(text_x, 0.3, f"'{pin_sequence[i]}'", 
                     ha='center', va='bottom', 
                     color='#c0392b', fontsize=15, fontweight='bold')

    # --- 5. Chart beautification ---
    # plt.title(f'Figure: Realistic Keystroke Trace from {csv_file}', fontsize=12, fontname='Times New Roman')
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Norma. Ampl.', fontsize=18, labelpad=-0.9)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, t[-10000])
    plt.ylim(-1.1, 1.0)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(fontsize=13.7, loc='lower right', ncol=2)

    plt.tight_layout()
    
    plt.show()

# ==============================
# Main program
# ==============================
if __name__ == "__main__":
    
    csv_filename = raw_signal_path("SittingPosture_PIN_140730.csv")
    pin = "140730"  

    try:
        plot_keystroke_signal(csv_filename, pin_sequence=pin)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_filename}")
    except Exception as e:
        print(f"❌ Plotting error: {e}")
