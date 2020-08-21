#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-08-21 11:57
import pyaudio
import numpy as np
from mfcc import mfcc_data
from inference import Inference

chunk = 320  # 20ms
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 16000  # Record at 44100 samples per second

p = pyaudio.PyAudio()  # Create an interface to PortAudio

inference = Inference()

print("Recording")

stream = p.open(
    format=sample_format,
    channels=channels,
    rate=fs,
    frames_per_buffer=chunk,
    input=True,
)

buffer = np.array([], dtype=np.float32)

while True:
    data = stream.read(chunk)
    buffer = buffer[chunk - fs :]
    tmp = np.frombuffer(data, dtype=np.int16)
    buffer = np.append(buffer, tmp / 32767)
    if len(buffer) != fs:
        continue
    x = mfcc_data(np.expand_dims(buffer, 1))
    print(inference.run(x))
