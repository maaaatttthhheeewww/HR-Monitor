import cv2
import numpy as np
import argparse
import time
import math
from collections import deque
from scipy.signal import butter, filtfilt, detrend, welch, find_peaks, peak_prominences
import matplotlib.pyplot as plt
import io
import csv
import os
from datetime import datetime


def get_fps(time_buffer):
    if len(time_buffer) < 2:
        return None
    durations = np.diff(time_buffer)
    return 1.0 / np.mean(durations)


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)


def adaptive_smoothing(data, alpha=0.1):
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def normalize_signal(data):
    """Normalize signal to zero mean and unit variance."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data
    return (data - mean) / std


def create_plot_image(raw_signal, filtered, freqs, psd):
    """Create a plot image for visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    
    ax_raw = axes[0]
    ax_raw.plot(raw_signal, label="Raw Signal", color="blue", alpha=0.6)
    ax_raw.set_title("Raw Signal")
    ax_raw.set_xlabel("Sample Index")
    ax_raw.set_ylabel("Intensity")
    ax_raw.legend()
    
    ax_filt = axes[1]
    ax_filt.plot(filtered, label="Filtered Signal", color="red")
    ax_filt.set_title("Filtered Signal")
    ax_filt.set_xlabel("Sample Index")
    ax_filt.set_ylabel("Intensity")
    ax_filt.legend()
    
    ax_psd = axes[2]
    ax_psd.plot(freqs, psd, label="PSD", color="green")
    ax_psd.set_title("Power Spectral Density")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power")
    ax_psd.legend()
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plot_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return plot_img


def create_unified_display(frame, plot_img=None, finger_detected=False, display_text="Place finger over camera"):
    """Create a unified display with camera feed and plot if available."""
    h, w = frame.shape[:2]
    
    if plot_img is not None:
        plot_h, plot_w = plot_img.shape[:2]
        aspect_ratio = plot_w / plot_h
        new_plot_h = h
        new_plot_w = int(new_plot_h * aspect_ratio)
        plot_img_resized = cv2.resize(plot_img, (new_plot_w, new_plot_h))
        
        display = np.zeros((h, w + new_plot_w, 3), dtype=np.uint8)
        display[:, :w] = frame
        display[:, w:w+new_plot_w] = plot_img_resized
    else:
        display = np.zeros((h, w * 2, 3), dtype=np.uint8)
        display[:, :w] = frame
        
        cv2.putText(display, "Waiting for data...", (w + 50, h // 2),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    overlay = display.copy()
    panel_height = 125
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
    
    cv2.putText(display, "Finger cPPG Monitor", (20, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    text_color = (0, 255, 0) if finger_detected else (0, 160, 255)
    cv2.putText(display, display_text, (20, 70),
              cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    
    status_color = (0, 255, 0) if finger_detected else (0, 0, 255)
    cv2.circle(display, (w - 40, 30), 10, status_color, -1)
  
    return display


class FingerCPPG:
    def __init__(self, args):
        self.buffer_duration = args.buffer_duration
        self.hrv_buffer_duration = args.hrv_buffer_duration
        self.min_buffer_length = self.buffer_duration * args.fps
        self.filter_order = args.filter_order
        self.hr_min = args.hr_min
        self.hr_max = args.hr_max
        self.bpm_update_interval = args.bpm_update_interval
        self.hrv_update_interval = args.hrv_update_interval
        self.fps = args.fps
        self.signal_buffer = deque(maxlen=int(self.buffer_duration * self.fps))
        self.hrv_buffer = deque(maxlen=int(self.hrv_buffer_duration * self.fps))
        self.time_buffer = deque(maxlen=int(self.hrv_buffer_duration * self.fps))
        self.last_bpm_update = time.time()
        self.last_hrv_update = time.time()
        self.current_bpm = None
        self.current_hrv = None
        self.plot_img = None
        self.stabilization_time = 4.0 
        self.start_time = None 
        
        os.makedirs("./resources/output", exist_ok=True)
        self.log_filename = f"./resources/output/cppg_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'bpm', 'sdnn', 'rmssd'])


    def log_data(self):
        timestamp = datetime.now().isoformat()
        bpm = round(self.current_bpm, 2) if self.current_bpm is not None else ""
        sdnn = round(self.current_hrv["SDNN"], 4) if self.current_hrv and self.current_hrv["SDNN"] is not None else ""
        rmssd = round(self.current_hrv["RMSSD"], 4) if self.current_hrv and self.current_hrv["RMSSD"] is not None else ""
        with open(self.log_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, bpm, sdnn, rmssd])

    def estimate_bpm(self):
        """Estimate BPM if enough data is available and update current_bpm."""
        if (time.time() - self.last_bpm_update >= self.bpm_update_interval and
            len(self.signal_buffer) >= self.min_buffer_length):
            fps_est = get_fps(self.time_buffer)
            if fps_est is not None:
                signal_arr = np.array(self.signal_buffer)
                signal_norm = normalize_signal(signal_arr)
                signal_detrended = detrend(signal_norm)
                filtered = bandpass_filter(signal_detrended,
                                           self.hr_min,
                                           self.hr_max,
                                           fps_est,
                                           order=self.filter_order)
                filtered_smoothed = adaptive_smoothing(filtered, alpha=0.1)
                freqs, psd = welch(filtered_smoothed, fs=fps_est, nperseg=len(filtered_smoothed)//4, scaling="spectrum")
                freq_mask = (freqs >= self.hr_min) & (freqs <= self.hr_max)
                if np.sum(freq_mask) > 0:
                    freqs_range = freqs[freq_mask]
                    psd_range = psd[freq_mask]
                    peak_freq = freqs_range[np.argmax(psd_range)]
                    heart_rate = peak_freq * 60 
                    if self.current_bpm is None:
                        self.current_bpm = heart_rate
                        print(f"Initial BPM set to {math.ceil(self.current_bpm)}")
                    else:
                        alpha = 0.2
                        self.current_bpm = alpha * heart_rate + (1 - alpha) * self.current_bpm
                    
                    self.plot_img = create_plot_image(signal_arr, filtered_smoothed, freqs_range, psd_range)
                    self.last_bpm_update = time.time()
                    self.log_data()

        return self.current_bpm
    
    def estimate_hrv(self):
        if (time.time() - self.last_hrv_update >= self.hrv_update_interval and len(self.hrv_buffer) >= (self.min_buffer_length * 3)):
            fps_est = get_fps(self.time_buffer)
            if fps_est is not None:
                raw_data = np.array(self.hrv_buffer)

                data = np.max(raw_data) - raw_data
    
                peaks, _ = find_peaks(data)
                peaks = np.array(peaks, dtype=int)

                prominences = peak_prominences(data, peaks)[0]
    
                optimal_prominence = np.mean(prominences) * 2

                min_ibi = 0.3   

                distance_frames = int(min_ibi * fps_est)
                
                filtered_peaks, _ = find_peaks(data, prominence=optimal_prominence, distance=distance_frames)
    
                filtered_peaks = np.array(filtered_peaks, dtype=int)

                if len(filtered_peaks) < 2:
                    print("Less than 2 peaks detected for HRV calculation.")
                    return {"SDNN": None, "RMSSD": None} 
    
                rr_intervals = np.diff(filtered_peaks) / fps_est 
    
                rr_intervals = rr_intervals[(rr_intervals > 0.75) & (rr_intervals < 1.07)] 
    
                if len(rr_intervals) < 2:
                    print("Not enough RR intervals for HRV calculation.")
                    return {"SDNN": None, "RMSSD": None}
    
                sdnn = np.std(rr_intervals, ddof=1) 
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
                self.last_hrv_update = time.time()
                print(f"SDNN: {sdnn:.2f}, RMSSD: {rmssd:.2f}")
                self.current_hrv = {"SDNN": sdnn, "RMSSD": rmssd}
                self.log_data()

        return self.current_hrv


    def update(self, frame, timestamp, finger_threshold):
        mean_red = np.mean(frame[:, :, 2])
        finger_detected = mean_red > finger_threshold
        
        if finger_detected:
            if self.start_time is None:
                self.start_time = timestamp
                print(f"Finger detected. Starting {self.stabilization_time}s stabilization period...")
            
            in_stabilization = timestamp - self.start_time < self.stabilization_time

            if not in_stabilization:

                self.signal_buffer.append(mean_red)
                self.hrv_buffer.append(mean_red)
                self.time_buffer.append(timestamp)
                bpm = self.estimate_bpm()
                hrv = self.estimate_hrv()
              
                if bpm is not None:
                    display_text = f"BPM: {math.ceil(bpm)}"
                else:
                    display_text = "Calculating BPM..."
                if hrv is not None and hrv["SDNN"] is not None:
                    display_text += f"  |  SDNN: {hrv['SDNN']:.2f}  RMSSD: {hrv['RMSSD']:.2f}"
                else:
                    display_text += "  |  Calculating HRV..."
            else:
                display_text = "Stabilizing..."
        else:
            self.signal_buffer.clear()
            self.hrv_buffer.clear()
            self.time_buffer.clear()
            self.current_bpm = None
            self.current_hrv = {"SDNN": None, "RMSSD": None}
            self.last_bpm_update = timestamp
            display_text = "Place finger over camera"
                    
        unified_display = create_unified_display(
            frame,
            plot_img=self.plot_img,
            finger_detected=finger_detected,
            display_text=display_text
        )
        
        cv2.imshow("Finger cPPG Monitor", unified_display)
        
        return display_text


def main():
    parser = argparse.ArgumentParser(
        description="Finger cPPG using Mean Red Pixel Intensity"
    )
    parser.add_argument("--fps", type=int, help="Camera frames per second")
    parser.add_argument("--buffer_duration", type=float, default=20,
                        help="Signal buffer duration (seconds)")
    parser.add_argument("--hrv_buffer_duration", type=float, default=60,
                        help="HRV buffer duration (seconds)")
    parser.add_argument("--min_buffer_length", type=int,
                        help="Minimum samples for heart rate estimation")
    parser.add_argument("--filter_order", type=int, default=3,
                        help="Order of the Butterworth filter")
    parser.add_argument("--hr_min", type=float, default=0.7,
                        help="Min frequency (Hz) for heart rate (approx. 42 BPM)")
    parser.add_argument("--hr_max", type=float, default=3.0,
                        help="Max frequency (Hz) for heart rate (approx. 180 BPM)")
    parser.add_argument("--bpm_update_interval", type=float, default=20.0,
                        help="Interval (seconds) for BPM update")
    parser.add_argument("--hrv_update_interval", type=float, default=60.0,
                        help="Interval (seconds) for HRV update")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Process every nth frame")
    parser.add_argument("--camera", type=int, default=1, 
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
        
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Actual FPS of the input video:", actual_fps)
    args.fps = actual_fps
    args.min_buffer_length = int(args.buffer_duration * args.fps)
    print("Minimum buffer length for heart rate estimation:", args.min_buffer_length)
    window_name = "Finger cPPG Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    cppg = FingerCPPG(args)
    finger_threshold = 200

    print("Waiting for finger to cover the camera...")
    
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break
            
        timestamp = time.time()
        
        if frame_counter % args.downsample == 0:
            cppg.update(frame, timestamp, finger_threshold)
        
        frame_counter += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("User requested exit.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()