#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
import time
from kan import *
import time
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, mkPen
import pyqtgraph as pg
import numpy as np

def run_live_plot(duration=10):
    """Stream EEG data and plot in real-time for a specified duration."""
    # Initialize BrainFlow parameters
    SampleRate = 256
    params = BrainFlowInputParams()
    params.serial_port = "COM7"  # Replace with your actual port
    board_id = BoardIds.MUSE_S_BLED_BOARD.value
    board = BoardShim(board_id, params)

    # Prepare the EEG session
    board.prepare_session()
    print("Starting Stream")
    board.start_stream()

    # Fetch EEG channels
    eeg_channels = board.get_eeg_channels(board_id)
    channel_count = len(eeg_channels)
    sample_window = SampleRate  # Plot the last second of data
    data_buffer = np.zeros((channel_count, sample_window))

    # Initialize PyQt application
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(title="Real-Time EEG Data")
    win.resize(800, 600)
    win.show()

    # Add plot and curves for each channel
    plots = []
    curves = []
    for i in range(channel_count):
        plot = win.addPlot(row=i, col=0)
        plot.setYRange(-100, 100)  # Adjust the range as per your data
        plot.setLabel("left", f"Channel {i + 1}", units="µV")
        plot.showGrid(x=True, y=True)
        curve = plot.plot(pen=pg.mkPen(color=pg.intColor(i, hues=channel_count), width=2))
        plots.append(plot)
        curves.append(curve)

    def update():
        """Fetch and plot live EEG data."""
        nonlocal data_buffer

        # Fetch the latest data
        current_data = board.get_current_board_data(25)
        for idx, channel_idx in enumerate(eeg_channels):
            channel_data = current_data[channel_idx]

            # Apply low-pass filter
            DataFilter.perform_lowpass(
                channel_data,
                SampleRate,
                35,  # Cutoff frequency
                3,   # Filter order
                FilterTypes.BUTTERWORTH,
                0
            )

            # Update the buffer
            data_buffer[idx, :-25] = data_buffer[idx, 25:]  # Shift data left
            data_buffer[idx, -25:] = channel_data  # Add new data

            # Update the curve
            curves[idx].setData(data_buffer[idx])

    def stop_after_duration():
        """Stop the EEG session and close the application after the specified duration."""
        print("10 seconds elapsed. Stopping stream.")
        board.stop_stream()
        board.release_session()
        app.quit()

    # Set up a timer to update the plot periodically
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(update)
    update_timer.start(40)  # ~25 Hz update rate

    # Set up a timer to stop the session after 30 seconds
    stop_timer = QtCore.QTimer()
    stop_timer.singleShot(duration * 1000, stop_after_duration)

    app.exec_()


if __name__ == "__main__":
    run_live_plot(duration=10)  # Run for 30 seconds

# Main Functions
def read_live():
    SampleRate = 256
    params = BrainFlowInputParams()
    params.serial_port = "COM7" # Depending on your device and OS
    board_id = BoardIds.MUSE_S_BLED_BOARD.value # For Muse S Device
    board = BoardShim(board_id, params)
    board.prepare_session()
    print("Analyzing Waves")
    board.start_stream()
    for i in range(5):
        time.sleep(1)
        # Gets the last 25 samples from the board without removing them from the buffer
        current_data = board.get_current_board_data(25)
    time.sleep(1)
    live_data = board.get_board_data()

    # Apply low-pass filter to each channel
    for channel_idx in range(1, live_data.shape[0]):  # Assuming first index is timestamps
        DataFilter.perform_lowpass(
            live_data[channel_idx],  # Channel data
            SampleRate,              # Sampling rate
            35,                    # Cutoff frequency (e.g., 30 Hz)
            3,                       # Filter order
            FilterTypes.BUTTERWORTH, # Filter type
            0                        # Ripple (used only for Chebyshev filters)
        )

    print("Ending Stream")
    board.stop_stream()
    board.release_session()
    return live_data

# Function to get power bins for each wave (alpha, beta, delta, gamma, theta)
def BrainwaveBins(data):  
    SampleRate = 256
    
    channel=data[0] 
    fftData = np.fft.fft(channel)
    freq = np.fft.fftfreq(len(channel)) * SampleRate

    outFftData = fftData[1:int(len(fftData)/2)]
    outMag = outFftData.real**2 + outFftData.imag**2
    outFreq = freq[1:int(len(freq)/2)]

    binsTotal = [0, 0, 0, 0, 0]
    binsCount = [0, 0, 0, 0, 0]

    for point in range(len(outFreq)):
        frequency = outFreq[point]

        if frequency < 4:  # Delta (0 - 4Hz)
            binsTotal[0] += outMag[point]
            binsCount[0] += 1
        elif frequency < 7.5:  # Theta (4 - 7.5Hz)
            binsTotal[1] += outMag[point]
            binsCount[1] += 1
        elif frequency < 12.5:  # Alpha (7.5 - 12.5Hz)
            binsTotal[2] += outMag[point]
            binsCount[2] += 1
        elif frequency < 30:  # Beta (12.5 - 30Hz)
            binsTotal[3] += outMag[point]
            binsCount[3] += 1
        elif frequency < 120:  # Gamma (30 - 120Hz)
            binsTotal[4] += outMag[point]
            binsCount[4] += 1

    binsAverage = list(np.array(binsTotal) / np.array(binsCount))
    return binsAverage

def wave_to_df(bin):
  wave_data = {
    "Delta" : bin[0],
    "Theta" : bin[1],
    "Alpha" : bin[2],
    "Beta" :  bin[3],
    "Gamma" : bin[4]
  }
  df = pd.DataFrame.from_dict([wave_data])
  return df

def main():
    data = read_live()
    bins = BrainwaveBins(data)
    df = wave_to_df(bins)
    return df

# Preprocess Based on Master Data - Make sure to have master data in directory
master_data = pd.read_csv(r"C:\Users\Ezra\Desktop\master_df1.csv") # Change this for your directory

y = master_data['Emotion']
X = master_data.drop('Emotion', axis=1)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
scaler = StandardScaler()
scaler.fit(X)
le = LabelEncoder()
y = le.fit_transform(y)

# Make Prediction
pred = main()
pred_scaled = scaler.transform(pred) 

import warnings
warnings.filterwarnings("ignore")
model = KAN.loadckpt(r"\Users\Ezra\Desktop\model\0.2") # Change this for your directory (Remove the ending after "0.2")

# Convert to tensor and add a batch dimension
new_instance_tensor = torch.tensor(pred_scaled, dtype=torch.float32)

# Pass the scaled instance through the model
logits = model(new_instance_tensor)

# Get the predicted class index
predicted_class_idx = torch.argmax(logits, dim=1).item()

# Decode the predicted index back to the emotion label
predicted_emotion = le.inverse_transform([predicted_class_idx])[0]

# Angry Bin Correlates To Stress
if predicted_emotion == "Angry":
    predicted_emotion = "Stressed"
    
# Ensure that predicted_emotion is a string representing the class label
# Mapping index to emotion directly, since inverse_transform() will return string labels
print()
print()
print(predicted_emotion)






