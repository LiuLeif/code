#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-08-22 20:14

INFERENCE_INTERVAL = 50  # ms
SMOOTH_INTERVAL = 500  # ms
MIN_PROBABILITY = 0.5
CLIP_DURATION = 1000  # ms
RESPONSE_INTERVAL = 1000  # ms

# MFCC
WINDOW_SIZE_MS = 40  # ms
WINDOW_STRIDE_MS = 20  # ms
DCT_COEFFICIENT_COUNT = 10

SAMPLE_RATE = 16000
SAMPLES = SAMPLE_RATE * CLIP_DURATION // 1000
FRAMES = (CLIP_DURATION - WINDOW_SIZE_MS) // WINDOW_STRIDE_MS + 1

# train
WORDS = ["silent", "unknown", "right"]
WORDS_TO_CHECK = set(["silent", "unknown", "right"])
