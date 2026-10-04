import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ECG DSP Analysis System",
    page_icon="❤️",
    layout="wide"
)

st.title("ECG DSP Analysis System")

st.caption(
    "Software-based ECG signal processing for heart-rate monitoring "
    "and basic signal analysis"
)
st.info(
    "ECG Input → Filtering → R-Peak Detection → "
    "RR Interval → Heart Rate → HRV → FFT Analysis"
)
st.write(
    "This software analyzes ECG signals using digital signal processing "
    "techniques to estimate heart rate, detect R-peaks, analyze RR "
    "intervals, calculate HRV features, and study the frequency spectrum."
)

# --------------------------------------------------
# CSV Upload
# --------------------------------------------------

st.sidebar.title("ECG Controls")
st.sidebar.header("Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload ECG CSV file",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.header("Synthetic ECG Generator")

generate_ecg = st.sidebar.button("Generate ECG")

synthetic_hr = st.sidebar.slider(
    "Heart Rate (BPM)",
    min_value=50,
    max_value=120,
    value=90,
    step=1
)
# Use uploaded file if available
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

elif generate_ecg:
    fs = 360
    duration = 10
    time = np.arange(0, duration, 1 / fs)

    rr = 60 / synthetic_hr

    ecg = 0.03 * np.sin(2 * np.pi * 0.33 * time)

    beat_times = np.arange(0.6, duration, rr)

    for bt in beat_times:
        ecg += 0.12 * np.exp(-((time - (bt - 0.20)) / 0.045) ** 2)
        ecg -= 0.10 * np.exp(-((time - (bt - 0.045)) / 0.018) ** 2)
        ecg += 0.95 * np.exp(-((time - bt) / 0.012) ** 2)
        ecg -= 0.25 * np.exp(-((time - (bt + 0.035)) / 0.020) ** 2)
        ecg += 0.28 * np.exp(-((time - (bt + 0.24)) / 0.080) ** 2)

    rng = np.random.default_rng(42)
    ecg += 0.02 * rng.normal(size=len(time))

    df = pd.DataFrame({
        "time": time,
        "ecg": ecg
    })

    st.sidebar.success(
        f"Synthetic ECG generated at {synthetic_hr} BPM"
    )

else:
    df = pd.read_csv("data/ecg.csv")
    st.sidebar.info("Using default ECG dataset.")
st.sidebar.markdown("---")
st.sidebar.header("Export")
st.sidebar.download_button(
    label="Download Current ECG Data",
    data=df.to_csv(index=False),
    file_name="ECG_Data.csv",
    mime="text/csv"
)
# --------------------------------------------------
# Check CSV columns
# --------------------------------------------------

if "time" not in df.columns or "ecg" not in df.columns:

    st.error(
        "Invalid CSV format. The file must contain "
        "'time' and 'ecg' columns."
    )

    st.stop()

# --------------------------------------------------
# ECG data
# --------------------------------------------------

time = df["time"].values
ecg = df["ecg"].values

# Sampling frequency
fs = 360

# --------------------------------------------------
# ECG Filtering
# --------------------------------------------------

b, a = butter(
    4,
    [0.5 / (fs / 2), 40 / (fs / 2)],
    btype="band"
)

filtered_ecg = filtfilt(b, a, ecg)

# --------------------------------------------------
# R-Peak Detection
# --------------------------------------------------

derivative = np.diff(
    filtered_ecg,
    prepend=filtered_ecg[0]
)

squared = derivative ** 2

window_size = int(0.15 * fs)

integrated = np.convolve(
    squared,
    np.ones(window_size) / window_size,
    mode="same"
)

threshold = 0.35 * np.max(integrated)

peaks, _ = find_peaks(
    integrated,
    height=threshold,
    distance=int(0.25 * fs)
)

# --------------------------------------------------
# Heart Rate
# --------------------------------------------------

r_peak_times = time[peaks]

if len(r_peak_times) >= 2:

    rr_intervals = np.diff(r_peak_times)

    average_rr = np.mean(rr_intervals)

    heart_rate = 60 / average_rr

else:

    rr_intervals = np.array([])

    average_rr = 0

    heart_rate = 0

# --------------------------------------------------
# FFT
# --------------------------------------------------

N = len(filtered_ecg)

fft_result = np.fft.fft(filtered_ecg)

frequencies = np.fft.fftfreq(
    N,
    1 / fs
)

magnitude = np.abs(fft_result) / N

positive = frequencies >= 0

frequencies = frequencies[positive]
magnitude = magnitude[positive]

if len(magnitude) > 1:

    dominant_index = np.argmax(
        magnitude[1:]
    ) + 1

    dominant_frequency = frequencies[
        dominant_index
    ]

else:

    dominant_frequency = 0
if heart_rate > 0:
    heart_rate_frequency = heart_rate / 60
else:
    heart_rate_frequency = 0

# --------------------------------------------------
# Basic Assessment
# --------------------------------------------------

if heart_rate == 0:

    assessment = "Unable to calculate heart rate"

elif heart_rate < 60:

    assessment = (
        "Below selected resting reference range"
    )

elif heart_rate <= 100:

    assessment = (
        "Within selected resting reference range"
    )

else:

    assessment = (
        "Above selected resting reference range"
    )

# --------------------------------------------------
# Results
# --------------------------------------------------

st.header("ECG Analysis Results")
st.caption("Key parameters calculated from the processed ECG signal")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Heart Rate",
        f"{heart_rate:.2f} BPM"
    )

with col2:

    st.metric(
        "R-Peaks",
        len(peaks)
    )

with col3:

    st.metric(
        "Average RR Interval",
        f"{average_rr:.3f} s"
    )

with col4:
    st.metric(
        "Heart Rate Frequency",
        f"{heart_rate_frequency:.2f} Hz"
    )

st.info(
    f"Basic Assessment: {assessment}"
)
st.subheader("Heart Rate Level")

hr_progress = min(max(heart_rate / 150, 0.0), 1.0)

st.progress(hr_progress)

st.caption(
    f"Current Heart Rate: {heart_rate:.2f} BPM | "
    "Reference scale: 0–150 BPM"
)
if heart_rate == 0:
    st.error("Heart Rate Status: Unable to calculate")
elif heart_rate < 60:
    st.warning("Heart Rate Status: Below selected resting range")
elif heart_rate <= 100:
    st.success("Heart Rate Status: Within selected resting range")
else:
    st.warning("Heart Rate Status: Above selected resting range")

# --------------------------------------------------
# Original ECG
# --------------------------------------------------

st.header("ECG Signal")
st.caption("Original ECG signal received as input")
st.subheader("Signal Analysis Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric("Sampling Frequency", f"{fs} Hz")

with summary_col2:
    st.metric("Signal Duration", f"{time[-1]:.2f} s")

with summary_col3:
    st.metric("ECG Samples", len(ecg))

fig1, ax1 = plt.subplots(
    figsize=(12, 4)
)

ax1.plot(
    time,
    ecg
)

ax1.set_xlabel(
    "Time (seconds)"
)

ax1.set_ylabel(
    "ECG Amplitude"
)

ax1.set_title(
    "Original ECG Signal"
)

ax1.grid()

st.pyplot(fig1)

# --------------------------------------------------
# Filtered ECG + R-Peaks
# --------------------------------------------------

st.header("Filtered ECG & R-Peak Detection")
st.caption("Noise-reduced ECG signal with detected R-peaks")
fig2, ax2 = plt.subplots(
    figsize=(12, 4)
)

ax2.plot(
    time,
    filtered_ecg,
    label="Filtered ECG"
)

ax2.scatter(
    r_peak_times,
    filtered_ecg[peaks],
    marker="o",
    label="R-peaks"
)

ax2.set_xlabel(
    "Time (seconds)"
)

ax2.set_ylabel(
    "ECG Amplitude"
)

ax2.set_title(
    "Filtered ECG and Detected R-Peaks"
)

ax2.legend()
ax2.grid()

st.pyplot(fig2)

# --------------------------------------------------
# FFT
# --------------------------------------------------

st.header("Frequency Spectrum")
st.caption("FFT-based frequency analysis of the ECG signal")

fig3, ax3 = plt.subplots(
    figsize=(12, 4)
)

ax3.plot(
    frequencies,
    magnitude
)

ax3.set_xlim(
    0,
    50
)

ax3.set_xlabel(
    "Frequency (Hz)"
)

ax3.set_ylabel(
    "Magnitude"
)

ax3.set_title(
    "FFT Frequency Spectrum"
)

ax3.grid()

st.pyplot(fig3)

# --------------------------------------------------
# RR Interval
# --------------------------------------------------

st.header("RR Interval Analysis")
st.caption("Variation of time between consecutive R-peaks")
st.subheader("HRV Analysis")

if len(rr_intervals) >= 2:
    sdnn = np.std(rr_intervals, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))

    hrv_col1, hrv_col2 = st.columns(2)

    with hrv_col1:
        st.metric("SDNN", f"{sdnn:.3f} s")

    with hrv_col2:
        st.metric("RMSSD", f"{rmssd:.3f} s")
else:
    st.warning("Not enough RR intervals for HRV analysis.")

if len(rr_intervals) > 0:

    fig4, ax4 = plt.subplots(
        figsize=(12, 4)
    )

    ax4.plot(
        range(
            1,
            len(rr_intervals) + 1
        ),
        rr_intervals,
        marker="o"
    )

    ax4.set_xlabel(
        "Beat Number"
    )

    ax4.set_ylabel(
        "RR Interval (seconds)"
    )

    ax4.set_title(
        "RR Interval Variation"
    )

    ax4.grid()

    st.pyplot(fig4)

else:

    st.warning(
        "Not enough R-peaks for RR interval analysis."
    )

# --------------------------------------------------
# System Information
# --------------------------------------------------

st.header("System Information")
st.caption("DSP processing parameters used for the current ECG analysis")

st.write(
    f"**Sampling Frequency:** {fs} Hz"
)

st.write(
    f"**Number of ECG Samples:** {len(ecg)}"
)

st.write(
    f"**Recording Duration:** {time[-1]:.2f} seconds"
)

st.write(
    "**Filtering:** 4th-order Butterworth "
    "Band-Pass Filter (0.5–40 Hz)"
)

st.write(
    "**Peak Detection:** Derivative + Squaring "
    "+ Moving Window Integration"
)
st.success(
    "ECG signal processing completed successfully."
)

# --------------------------------------------------
# Download Analysis Report
# --------------------------------------------------

report = f"""
ECG DSP ANALYSIS REPORT
=======================

Sampling Frequency: {fs} Hz
Number of ECG Samples: {len(ecg)}
Recording Duration: {time[-1]:.2f} seconds

R-Peaks Detected: {len(peaks)}
Average RR Interval: {average_rr:.3f} seconds
Average Heart Rate: {heart_rate:.2f} BPM
Heart Rate Frequency: {heart_rate_frequency:.2f} Hz

Basic Assessment:
{assessment}

DSP Processing:
1. Band-pass filtering
2. Differentiation
3. Squaring
4. Moving-window integration
5. R-peak detection
6. RR interval calculation
7. Heart-rate calculation
8. FFT frequency analysis

Note:
This software provides educational ECG signal
processing analysis and is not intended for
medical diagnosis.
"""

st.download_button(
    label="Download Analysis Report",
    data=report,
    file_name="ECG_DSP_Report.txt",
    mime="text/plain"
)

st.warning(
    "Educational software only. This system is not intended "
    "for medical diagnosis or clinical decision-making."
)