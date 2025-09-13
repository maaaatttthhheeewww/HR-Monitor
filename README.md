# Remote Photoplethysmography (rPPG) & Contact Photoplethysmography (cPPG) Monitoring

This repository contains the implementation of my Final Year Project (FYP), which explores heart rate monitoring using both **remote photoplethysmography (rPPG)** (face-based) and **contact photoplethysmography (cPPG)** (finger-based).

The project demonstrates how computer vision and signal processing can be used to estimate **heart rate (BPM)** and **heart rate variability (HRV)** from camera input in real-time.

---

## 📂 Project Structure

### `facial_rppg.py`  
Implements **facial rPPG monitoring**:

- Detects the user’s face using OpenCV Haar cascades.  
- Defines regions of interest (forehead, cheeks).  
- Extracts color signals and applies the **CHROM method** for rPPG.  
- Processes signals with detrending, bandpass filtering, and smoothing.  
- Estimates heart rate (BPM) from the strongest frequency component.  
- Displays real-time camera feed, ROIs, signal plots, and BPM overlay.  
- Logs BPM values to CSV for further analysis.  

### `finger_cppg.py`  
Implements **finger cPPG monitoring**:

- Detects whether a finger covers the camera (based on mean red channel intensity).  
- Extracts the red intensity signal.  
- Estimates BPM using **Welch’s periodogram**.  
- Calculates HRV metrics:
  - **SDNN** (Standard Deviation of NN intervals)  
  - **RMSSD** (Root Mean Square of Successive Differences)  
- Displays real-time feed with signal plots and metrics overlay.  
- Logs BPM, SDNN, and RMSSD values to CSV.

---

## 🧑‍💻 Requirements

Install the required dependencies:

```bash
pip install opencv-python numpy scipy matplotlib
```
## ▶️ Usage

### 1. Facial rPPG (Remote)

```bash
python facial_rppg.py --camera 0
```
#### Arguments
--buffer_duration : Signal buffer duration (default: 30s)
--hr_min : Minimum frequency (Hz) for HR (~42 BPM default)
--hr_max : Maximum frequency (Hz) for HR (~180 BPM default)
--bpm_update_interval : How often to update BPM (seconds, default: 20)
--downsample : Process every nth frame (default: 1)
--camera : Camera index (default: 0)

### 2. Fingertip cPPG (Contact)

```bash
python finger_cppg.py --camera 0
```
#### Arguments
--buffer_duration : Signal buffer duration (default: 20s)
--hrv_buffer_duration : HRV buffer duration (default: 60s)
--bpm_update_interval : BPM update interval (default: 20s)
--hrv_update_interval : HRV update interval (default: 60s)
--hr_min, --hr_max : HR frequency range in Hz (default: 0.7–3.0)
--downsample : Process every nth frame (default: 1)
--camera : Camera index (default: 0)

## 📊 Output
Live camera feed with overlays.
Signal plots (raw, filtered, power spectral density).
BPM and HRV metrics displayed on screen.