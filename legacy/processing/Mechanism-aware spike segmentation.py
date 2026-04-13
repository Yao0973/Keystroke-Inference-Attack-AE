import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
import matplotlib.font_manager as font_manager

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import raw_signal_path


def process_and_plot_AMPD(input_csv, period=4600, percentile=60):
    """
    Read CSV file, detect significant protruding peaks, and perform spline interpolation
    """
    print("Reading data...")
    df = pd.read_csv(input_csv)
    data = df['CH1(V)'].values
    time = df['Time(s)'].values
    N = len(data)

    # Calculate threshold using percentile
    threshold = np.percentile(data, percentile)

    peaks_indices = []
    print("Starting to detect significant peaks...")
    for i in tqdm(range(0, N - period, period), desc="Processing period"):
        period_data = data[i:i + period]
        max_index = np.argmax(period_data)
        max_value = period_data[max_index]
        if max_value > threshold:
            peaks_indices.append(i + max_index)

    peaks_indices = np.array(peaks_indices)
    peaks_times = time[peaks_indices]

    # Spline interpolation smoothing
    if len(peaks_times) > 3:
        unique_times, unique_indices = np.unique(peaks_times, return_index=True)
        if len(unique_times) > 3:
            X_smooth = np.linspace(unique_times.min(), unique_times.max(), 300)
            spl = make_interp_spline(unique_times, data[peaks_indices[unique_indices]], k=3)
            y_smooth = spl(X_smooth)
            return X_smooth, y_smooth
        else:
            return peaks_times, data[peaks_indices]

    return peaks_indices, peaks_times

def detect_valleys_and_split(data, time, theta=0.35, alpha=0.1):
    """
    Detect valleys and perform cutting based on set thresholds.
    Ensure each peak corresponds to two valleys
    """
    valley_indices, _ = find_peaks(-data)  # Get valley indices
    peak_indices, _ = find_peaks(data, distance=10)  # Get peak indices
    print("valley indices:", valley_indices)
    print("peak indices:", peak_indices)

    valid_peaks = []
    valid_valleys = []

    # Find two valleys before and after each peak
    for peak_idx in peak_indices:
        # Find the nearest valleys before and after the peak
        prev_valley = None
        next_valley = None

        # Check previous valley
        for vi in reversed(valley_indices[valley_indices < peak_idx]):
            if data[vi] < theta:  # Ensure valley meets condition
                prev_valley = vi
                break

        # Check next valley
        for vi in valley_indices[valley_indices > peak_idx]:
            if data[vi] < theta:  # Ensure valley meets condition
                next_valley = vi
                break

        # If both previous and next valleys are found, and their difference with the peak is greater than alpha, consider valid
        if prev_valley is not None and next_valley is not None:
            prev_peak_diff = data[peak_idx] - data[prev_valley]
            next_peak_diff = data[peak_idx] - data[next_valley]

            if prev_peak_diff > alpha and next_peak_diff > alpha:
                valid_peaks.append(peak_idx)
                valid_valleys.append(prev_valley)
                valid_valleys.append(next_valley)

    # Return valid cut point times
    cut_points = time[valid_valleys]
    cut_points = sorted(cut_points)
    segments = []
    for i in range(0, len(cut_points) - 1, 2):
        segments.append((cut_points[i], cut_points[i + 1]))

    return valid_peaks, valid_valleys, segments

if __name__ == '__main__':
    file_name = raw_signal_path("keystroke_sequence_170246_output.csv")
    x_data, y_data = process_and_plot_AMPD(file_name)

    # Read the original signal data
    df = pd.read_csv(file_name)
    original_time = df['Time(s)'].values
    original_data = df['CH1(V)'].values

    # Normalize the X-axis to [0, 10]
    x_min, x_max = np.min(x_data), np.max(x_data)
    x_data_scaled = (x_data - x_min) / (x_max - x_min) * 10

    original_min, original_max = np.min(original_time), np.max(original_time)
    original_time_scaled = (original_time - original_min) / (original_max - original_min) * 10

    # Filter the original signal data where y values are between 0 and 1
    valid_original_indices = np.where((original_data >= 0) & (original_data <= 1))
    original_time_scaled = original_time_scaled[valid_original_indices]
    original_data = original_data[valid_original_indices]

    # Filter the fitted signal data where y values are between 0 and 1
    valid_fitted_indices = np.where((y_data >= 0) & (y_data <= 1))
    x_data_scaled = x_data_scaled[valid_fitted_indices]
    y_data = y_data[valid_fitted_indices]

    # Get peak and valley indices and segments
    peak_indices, valley_indices, segments = detect_valleys_and_split(y_data, x_data_scaled, theta=0.35, alpha=0.1)

    plt.figure(figsize=(2.0, 1.5), dpi=800)
    plt.grid(False)

    # Plot the original signal
    plt.plot(original_time_scaled, original_data, color='#005cab', linewidth=0.3, label='Original signal')

    # Plot the fitted signal
    plt.plot(x_data_scaled, y_data, color='#EDB120', linewidth=0.6, label='Fitted envelope')##e31b23-red

    # Plot red dashed lines according to the cutting intervals
    first_segment = True
    for (start_time, end_time) in segments:
        indices = np.where((x_data_scaled >= start_time) & (x_data_scaled <= end_time))[0]
        if len(indices) > 1:
            if first_segment:
                plt.plot(x_data_scaled[indices], y_data[indices], color='#e31b23', linewidth=0.3, label='Interaction segment')
                first_segment = False
            else:
                plt.plot(x_data_scaled[indices], y_data[indices], color='#e31b23', linewidth=0.3)


    # Add a legend
    font = font_manager.FontProperties(family='Times New Roman', style='normal', size=3.7)

    plt.legend(loc='best', ncol=1, prop=font).get_frame().set_linewidth(0.37)

    # Coordinate settings
    plt.xlim(0, 8)
    plt.ylim(0, 1.01)    
    plt.xticks([0, 2, 4, 6, 8], ['0', '2', '4', '6', '8'],
               fontproperties='Times New Roman', size=5)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
               fontproperties='Times New Roman', size=5)
    plt.xlabel('Time (s)', fontproperties='Times New Roman', size=6, labelpad=0.7)
    plt.ylabel('Norm. Current Ampl.', fontproperties='Times New Roman', size=6, labelpad=0.7)

    # Coordinate axis line width
    width = 0.4
    plt.tick_params(width=width, length=1, axis='both', which='major', pad=1)
    ax = plt.gca()
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(width)

    # Save and display
    plt.tight_layout()
    plt.show()
