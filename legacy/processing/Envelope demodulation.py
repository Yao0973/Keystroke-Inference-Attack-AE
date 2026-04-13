import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# from anaconda_project.internal.cli.environment_commands import main_add
from tqdm import tqdm
from scipy.interpolate import make_interp_spline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import raw_signal_path
#Keystroke parameters
# def process_and_plot_AMPD(input_csv, period=4600, percentile=70):
#Swipe parameters
# def process_and_plot_AMPD(input_csv, period=4000, percentile=65):
#Handwriting parameters
def process_and_plot_AMPD(input_csv, period=4600, percentile=65):

    print("Reading data...")
    df = pd.read_csv(input_csv)
    data = df['CH1(V)'].values

    time = df['Time(s)'].values
    N = len(data)

    # Calculate threshold using percentile
    threshold = np.percentile(data, percentile)

    peaks_indices = []
    print("Starting to detect significant peaks...")

    # Process data by period
    for i in tqdm(range(0, N - period, period), desc="Processing period"):
        # Find maximum value exceeding threshold in each period
        period_data = data[i:i + period]
        max_index = np.argmax(period_data)
        max_value = period_data[max_index]

        # Only record significant peaks
        if max_value > threshold:
            peaks_indices.append(i + max_index)

    peaks_indices = np.array(peaks_indices)
    peaks_times = time[peaks_indices]



    # Create smooth curve using spline interpolation, add deduplication logic
    if len(peaks_times) > 3:
        # Remove duplicate time points
        unique_times, unique_indices = np.unique(peaks_times, return_index=True)

        # Only perform interpolation when there are enough unique points
        if len(unique_times) > 3:
            X_smooth = np.linspace(unique_times.min(), unique_times.max(), 300)
            spl = make_interp_spline(unique_times, data[peaks_indices[unique_indices]], k=3)
            y_smooth = spl(X_smooth)


            return X_smooth, y_smooth
        else:

            return peaks_times, data[peaks_indices]

    return peaks_indices, peaks_times


if __name__ == '__main__':
    file_name = raw_signal_path("keystroke.csv")
    x_data,y_data=process_and_plot_AMPD(file_name)
    # Apply linear transformation
    x_min, x_max = np.min(x_data), np.max(x_data)
    x_data_scaled = (x_data - x_min) / (x_max - x_min) * 10

    plt.figure(figsize=(2.3, 1.6), dpi=500)
    plt.grid(False)

    # Plot noise signal, set line color to purple and thicken
    plt.plot( x_data_scaled, y_data, linewidth=0.7)

    #Set horizontal axis range to 0 to 10 seconds
    # plt.xlim(0, 10)
    #
    # # Set vertical axis range to -1 to 1
    # plt.ylim(0, 1)

    #Set X and Y axis ticks
    a = [0, 2, 4, 6, 8, 10]  # set to 0-10 second ticks
    plt.xticks(a, ('0', '2', '4', '6', '8', '10'), fontproperties='Times New Roman', size=5)

    b = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(b, ('0', '0.2', '0.4', '0.6', '0.8','1.0'),
               fontproperties='Times New Roman', size=5)

    # Set tick parameters
    plt.tick_params(axis='both', which='major', pad=0.7)

    # Add labels
    plt.xlabel('Time (s1)', fontproperties='Times New Roman', size=6, labelpad=0.7)
    plt.ylabel('Amplitude', fontproperties='Times New Roman', size=6, labelpad=0.7)

    # Set axis width
    width = 0.4
    plt.tick_params(width=width, length=1, axis='both', which='major', pad=1)

    # Set axis border width
    ax = plt.gca()
    ax.spines['top'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)

    # Display image
    plt.show()
