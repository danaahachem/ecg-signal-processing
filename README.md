# ECG Signal Processing in Python

#### Description:
This project implements a simple ECG (electrocardiogram) signal processing pipeline in Python. It loads real or simulated ECG data from a `.csv` file, applies a bandpass filter to remove noise, detects R-peaks (heartbeats), and calculates the average heart rate (BPM).

The main features:
- Load raw ECG data using NumPy
- Apply a Butterworth bandpass filter using SciPy
- Detect R-peaks using `scipy.signal.find_peaks`
- Plot raw and filtered signals using Matplotlib
- Calculate average heart rate from R-R intervals

### File Structure:
- `project.py`: The main program. Includes:
  - `main()` function
  - `bandpass_filter()`
  - `detect_r_peaks()`
  - `compute_heart_rate()`
  - `plot_ECG()` and `plot_ECG_w_peaks()`
- `test_project.py`: Contains unit tests for the key functions using `pytest`
- `requirements.txt`: Lists the required Python libraries
- `ecg_data.csv`: The ECG signal data (simulated for demo)

### Technologies Used:
- Python 3
- NumPy
- SciPy
- Matplotlib
- Pytest

### Why I Chose This Project:
As someone interested in biomedical signal processing and real-world data, I wanted to build a complete pipeline to explore how ECG signals can be cleaned and interpreted using Python, without relying on pre-built medical software.

### Design Decisions:
I used a bandpass filter between 0.5 Hz and 40 Hz to match the typical frequency range of ECG signals. I chose 250 Hz as the sampling rate and added flexibility in adjusting thresholds. I also wanted clean plots and testable code, so I broke functionality into modular functions.