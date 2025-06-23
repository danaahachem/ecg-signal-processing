import numpy as np
from project import bandpass_filter, detect_r_peaks, compute_heart_rate

def test_bandpass_filter():
    raw = np.ones(1000)  # A flat signal (should stay flat)
    filtered = bandpass_filter(raw, 0.5, 40, 250)
    assert len(filtered) == len(raw)
    assert np.allclose(filtered, filtered[0], atol=1e-2)  # Still flat

def test_detect_r_peaks():
    # Simple fake ECG-like signal
    test_signal = np.array([0, 0, 1, 0, 0, 1, 0, 0])
    fs = 250
    peaks = detect_r_peaks(test_signal, fs)
    assert list(peaks) == [2, 5]

def test_compute_heart_rate():
    # Assume peaks 1 second apart at 250 Hz
    fs = 250
    peaks = [0, 250, 500, 750]  # 4 beats = 3 intervals of 1s
    bpm = compute_heart_rate(peaks, fs)
    assert int(bpm) == 60  # 60 BPM
