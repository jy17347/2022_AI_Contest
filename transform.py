
import librosa
import numpy as np

class LoadAudio(object):

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, data):
        path = data['path_wave']
        if path:
            samples, sample_rate = librosa.load(path, self.sample_rate)
        else:
            sample_rate = self.sample_rate
            samples = np.zeros(sample_rate, dtype=np.float32)
        data['samples'] = samples
        data['sample_rate'] = sample_rate
        return data

class FixAudioLength(object):
    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)

        if length < len(samples):
            data['samples'] = samples[:length]
        elif length > len(samples):
            data['samples'] = np.pad(samples, (0, length - len(samples)), 'constant')

        return data

class ToMelSpectrogram(object):

    def __init__(self, n_mels=32):
        self.n_mels = n_mels

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        s = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=self.n_mels)
        data['mel_spectrogram'] = librosa.power_to_db(s, ref=np.max)
        return data

