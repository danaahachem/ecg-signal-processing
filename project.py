import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy
from scipy import constants
from scipy import signal


def main():
    """
    Main function that runs the ECG signal processing pipeline:
    1. Loads ECG data from CSV
    2. Plots raw ECG signal
    3. Applies bandpass filter
    4. Detects R-peaks
    5. Plots filtered signal with R-peaks
    6. Calculates and prints heart rate
    """
    print("Starting ECG project...")

    # Initialize an empty ECG array (in case loading fails)
    ECG_values = np.array([])

    try:
        # Load ECG data from file
        ECG_values = np.loadtxt("ecg_data.csv")
        print("Loaded ECG values:")
        print(ECG_values)

        # Step 1: Plot the raw (unfiltered) ECG signal
        plot_ECG(ECG_values, "Raw ECG Signal")

        # Define filter parameters
        fs = 250            # Sampling rate in Hz
        lowcut = 0.5        # Low cutoff frequency (Hz)
        highcut = 40        # High cutoff frequency (Hz)

        # Step 2: Apply bandpass filter to remove noise/artifacts
        filtered_signal = bandpass_filter(ECG_values, lowcut, highcut, fs)

        # Step 3: Detect R-peaks in the filtered ECG signal
        r_peaks = detect_r_peaks(filtered_signal, fs)

        # Step 4: Plot the filtered ECG signal with red dots at R-peaks
        plot_ECG_w_peaks(filtered_signal, "Filtered ECG Signal", r_peaks)

        # Step 5: Compute heart rate (BPM) from R-peaks
        if len(r_peaks) > 1:
            bpm = compute_heart_rate(r_peaks, fs)
            print(f"Estimated Heart Rate: {bpm:.2f} BPM")
        else:
            print("Not enough peaks detected to compute heart rate.")

    except FileNotFoundError:
        sys.exit("Error: File not found!")


def plot_ECG(signal, title):
    """
    Plots a basic ECG signal.
    Args:
        signal: 1D numpy array of ECG values
        title: string title of the plot
    """
    plt.plot(signal, color="pink")
    plt.title(title)
    plt.xlabel("Sample number")
    plt.ylabel("Voltage (mV)")
    plt.grid(True)
    plt.show()


def plot_ECG_w_peaks(signal, title, peaks):
    """
    Plots ECG signal with red circles at detected R-peaks.
    Args:
        signal: filtered ECG signal
        title: title of the plot
        peaks: indices of detected R-peaks
    """
    plt.plot(signal, color="pink")
    plt.plot(peaks, signal[peaks], "ro")  # Red circles on R-peaks
    plt.title(title)
    plt.xlabel("Sample number")
    plt.ylabel("Voltage (mV)")
    plt.grid(True)
    plt.show()


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Applies a Butterworth bandpass filter to the ECG signal.
    Args:
        signal: raw ECG signal
        lowcut: low cutoff frequency (Hz)
        highcut: high cutoff frequency (Hz)
        fs: sampling frequency (Hz)
        order: order of the filter (default = 4)
    Returns:
        filtered_signal: bandpass-filtered ECG signal
    """
    b, a = scipy.signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="ba")
    filtered_signal = scipy.signal.lfilter(b, a, signal)
    return filtered_signal


def detect_r_peaks(filtered_signal, fs):
    """
    Detects R-peaks from a filtered ECG signal using a peak detection algorithm.
    Args:
        filtered_signal: ECG signal after filtering
        fs: sampling frequency (used for setting peak distance)
    Returns:
        peaks: indices where R-peaks occur
    """
    peaks, _ = scipy.signal.find_peaks(filtered_signal, height=0.5, distance=150)
    return peaks


def compute_heart_rate(peaks, fs):
    """
    Computes average heart rate in BPM based on R-peak positions.
    Args:
        peaks: list of indices of detected R-peaks
        fs: sampling frequency (Hz)
    Returns:
        Mean BPM (heart rate)
    """
    diff_sample = []
    for i in range(len(peaks) - 1):
        diff_sample.append(peaks[i + 1] - peaks[i])  # Time between peaks in samples
    RR_interval = np.array(diff_sample) / fs        # Convert to seconds
    BPM = 60 / RR_interval                           # Beats per minute
    return np.mean(BPM)


if __name__ == "__main__":
    main()
