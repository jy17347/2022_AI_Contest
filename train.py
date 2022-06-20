import time, os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from get_dataset import Audio_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
import librosa
import librosa.display


from transform import LoadAudio
from transform import FixAudioLength
from transform import ToMelSpectrogram




path_dataset = "./sample"

dataset = Audio_dataset(path_dataset)
pre1 = LoadAudio()
pre2 = FixAudioLength()
pre3 = ToMelSpectrogram()

dataset = pre3(pre2(pre1(dataset[1])))
plt.figure()
librosa.display.specshow(dataset['mel_spectrogram'])