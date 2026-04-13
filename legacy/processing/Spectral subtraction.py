import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import stft
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import derived_path, raw_signal_path
def process_audio_with_spectral_subtraction(input_csv, output_csv, noise_csv_files, alpha=2.0, beta=0.002):
    """
    Use multiple noise sample files for spectral subtraction denoising

    Parameters:
        input_csv: Input CSV file path
        output_csv: Output CSV file path
        noise_csv_files: List of noise CSV file paths
        alpha: Over-subtraction factor (1.0~3.0)
        beta: Spectral floor noise factor
    """
    # Read and preprocess data
    df = pd.read_csv(input_csv)
    audio_data = df['CH1(V)'].values.astype(np.float32)
    time_data = df['Time(s)'].values
    sample_rate = 10000  # Assume sample rate is 10000
    original_audio = audio_data.copy()

    # STFT parameter settings
    frame_length = 512
    hop_length = 128

    # Calculate STFT
    _, _, stft = signal.stft(audio_data, fs=sample_rate, window='hann',
                             nperseg=frame_length, noverlap=frame_length - hop_length)

    # Replace the original noise estimation part
    noise_power_list = []
    for noise_file in noise_csv_files:
        # Read noise file
        noise_df = pd.read_csv(noise_file)
        noise_data = noise_df['CH1(V)'].values.astype(np.float32)

        # Calculate noise STFT
        _, _, noise_stft = signal.stft(noise_data, fs=sample_rate, window='hann',
                                       nperseg=frame_length, noverlap=frame_length - hop_length)

        # Calculate noise power spectrum
        current_noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1)
        noise_power_list.append(current_noise_power)

    # Use the average of multiple noise samples as the final noise estimate
    noise_power = np.mean(noise_power_list, axis=0)

    # Spectral subtraction processing
    stft_clean = np.zeros_like(stft)
    for i in range(stft.shape[1]):
        power_spectrum = np.abs(stft[:, i]) ** 2
        snr = 10 * np.log10(power_spectrum / (noise_power + 1e-10))
        alpha_adjusted = alpha - np.clip((snr - 20) / 20, -2, 2)
        power_clean = np.maximum(power_spectrum - alpha_adjusted * noise_power,
                                 beta * power_spectrum)
        gain = np.sqrt(power_clean / (power_spectrum + 1e-10))
        stft_clean[:, i] = stft[:, i] * gain

    # Signal reconstruction
    _, cleaned_audio = signal.istft(stft_clean, fs=sample_rate, window='hann',
                                    nperseg=frame_length, noverlap=frame_length - hop_length)

    # Ensure length matching
    min_length = min(len(original_audio), len(cleaned_audio))
    original_audio = original_audio[:min_length]
    cleaned_audio = cleaned_audio[:min_length]

    # Normalization processing
    if len(cleaned_audio) > 0:
        cleaned_audio = cleaned_audio / np.max(np.abs(cleaned_audio))
    if len(original_audio) > 0:
        original_audio = original_audio / np.max(np.abs(original_audio))

    # Replace first 100 outliers
    if len(cleaned_audio) >= 200:
        cleaned_audio[:100] = cleaned_audio[100:200]
        # Replace last 100 outliers
        cleaned_audio[-100:] = cleaned_audio[-200:-100]


    # Save processed data
    output_df = pd.DataFrame({
        'Time(s)': time_data,
        'CH1(V)': cleaned_audio
    })
    output_df.to_csv(output_csv, index=False)

    return original_audio, cleaned_audio


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def STFT(signal, fs=10000, output_pdf="clean.pdf"):
    # Calculate STFT
    f, t_stft, Zxx = stft(signal, fs=fs, nperseg=1000)  # Window length 1000 suitable for low frequency
    print(Zxx.shape)
    # Calculate magnitude spectrum
    Zxx_magnitude = np.abs(Zxx)

    # Get original time length
    T = t_stft[-1]  # Maximum time value calculated by STFT (usually 10s)

    # Linearly scale time axis: map 0-T to 0-8
    t_stft_scaled = (t_stft / T) *8.9  # Perform time axis scaling

    # Plot spectrogram
    plt.figure(figsize=(2.0, 1.3), dpi=800)
    plt.grid(False)
    cax = plt.pcolormesh(t_stft_scaled, f, Zxx_magnitude, shading='auto', cmap='plasma')  # Use pcolormesh to plot spectrogram

    # Set time axis ticks (still using 0-8)
    plt.xlim(0, 8)
    plt.xticks(np.linspace(0, 8, 5), ['0', '2', '4', '6', '8'], fontproperties='Times New Roman', size=5)
    plt.ylim(0, 50)

    b = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    plt.yticks( b,fontproperties='Times New Roman', size=5)
    plt.xlabel('Time (s)', fontproperties='Times New Roman', size=6, labelpad=0.7)
    plt.ylabel('Frequency (Hz)', fontproperties='Times New Roman', size=6, labelpad=0.7)


    # Axis style adjustment
    width = 0.4
    plt.tick_params(width=width, length=1, axis='both', which='major', pad=1)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)

    plt.tight_layout()
    plt.show()

def draw(signal_data, file_name):
    fs = 10000  # Sample rate 10000 Hz

    # Normalize to [-1, 1]
    signal_data = signal_data / np.max(np.abs(signal_data))

    # Calculate time axis corresponding to x-axis and scale
    time = np.arange(0, len(signal_data)) / fs
    time = time * 0.8  # Map to 0-8 seconds

    # Plot signal
    plt.figure(figsize=(2.0, 1.3), dpi=800)
    plt.grid(False)
    plt.plot(time, signal_data, color='#005cab', linewidth=0.25)

    # Set horizontal axis range
    plt.xlim(0, 8)

    # Set x and y axis ticks
    plt.xticks([0, 2, 4, 6, 8], ['0', '2', '4', '6', '8'], fontproperties='Times New Roman', size=5)
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], ['-1.0', '-0.5', '0.0', '0.5', '1.0'],
               fontproperties='Times New Roman', size=5)

    # Set tick parameters
    plt.tick_params(axis='both', which='major', pad=0.7)

    # Add labels
    plt.xlabel('Time (s)', fontproperties='Times New Roman', size=6, labelpad=0.7)
    plt.ylabel('Norm. Current Ampl.', fontproperties='Times New Roman', size=6, labelpad=0.7)

    # Set axis width
    width = 0.4
    plt.tick_params(width=width, length=1, axis='both', which='major', pad=1)

    # Set axis border width
    ax = plt.gca()
    ax.spines['top'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)

    # Save image
    plt.savefig(file_name, format='pdf', dpi=500)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':

    # Example call
    input_csv = raw_signal_path('1234567890_original.CSV')
    output_csv = derived_path('output_cleaned_signal.csv')
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    noise_csv_files = [raw_signal_path('noise.CSV')]

    original_audio, cleaned_audio = process_audio_with_spectral_subtraction(input_csv, output_csv, noise_csv_files)
    # Plot original signal and denoised signal

    STFT(original_audio)

    STFT(cleaned_audio)
