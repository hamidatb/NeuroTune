#!/usr/bin/env python
# coding: utf-8

# In[13]:


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


# In[23]:


# Main Functions
def read_live():
    SampleRate = 256
    params = BrainFlowInputParams()
    params.serial_port = "COM7" # Depending on your device and OS
    board_id = BoardIds.MUSE_S_BLED_BOARD.value # For Muse S Device
    board = BoardShim(board_id, params)
    board.prepare_session()
    print("Starting Stream")
    board.start_stream()
    for i in range(5):
        time.sleep(1)
        # Gets the last 25 samples from the board without removing them from the buffer
        current_data = board.get_current_board_data(25)
        print(current_data.shape)
    time.sleep(1)
    live_data = board.get_board_data()
    print("Ending Stream")
    board.stop_stream()
    board.release_session()
    return live_data

# Function to get power bins for each wave (alpha, beta, delta, gamma, theta)
def BrainwaveBins(data):  
    SampleRate = 256
    
    channel=data[1] 
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


# In[24]:


# Preprocess Based on Master Data - Make sure to have master data in directory
master_data = pd.read_csv(r"C:\Users\Ezra\Desktop\master_df.csv")
y = master_data['Mood']
X = master_data.drop('Mood', axis=1)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
scaler = StandardScaler()
scaler.fit(X)
le = LabelEncoder()
y = le.fit_transform(y)


# In[25]:


# Make Prediction
pred = main()
pred_scaled = scaler.transform(pred) 

import warnings
warnings.filterwarnings("ignore")
model = KAN.loadckpt(r"\Users\Ezra\Desktop\model\0.5")

# Convert to tensor and add a batch dimension
new_instance_tensor = torch.tensor(pred_scaled, dtype=torch.float32)

# Pass the scaled instance through the model
logits = model(new_instance_tensor)

# Get the predicted class index
predicted_class_idx = torch.argmax(logits, dim=1).item()

# Decode the predicted index back to the emotion label
predicted_emotion = le.inverse_transform([predicted_class_idx])[0]

# Ensure that predicted_emotion is a string representing the class label
# Mapping index to emotion directly, since inverse_transform() will return string labels
print()
print()
print(predicted_emotion)


# In[ ]:




