from torchvision import transforms
from get_dataset import Audio_dataset
from transform import LoadAudio
from transform import FixAudioLength
from transform import ToMelSpectrogram
from transform import ToTensor

n_mels=40
batch_size=4
use_gpu=False
num_dataloader_workers_=1

path_dataset = "./sample"

dataset = Audio_dataset(path_dataset)

_loadaudio = LoadAudio()
_fixaudiolength = FixAudioLength()
_melSpec = ToMelSpectrogram()

composed = transforms.Compose([LoadAudio(),
                                FixAudioLength(),
                                ToMelSpectrogram(n_mels=n_mels), 
                                ToTensor('mel_spectrogram', 'input')])


sample = dataset[0]

transformed_sample = _loadaudio(sample)
print(transformed_sample['samples'])

transformed_sample = _fixaudiolength(sample)
print(transformed_sample['samples'])

transformed_sample = _melSpec(sample)
print(transformed_sample['mel_spectrogram'])

transformed_sample = composed(sample)
print(transformed_sample['path_wave'], transformed_sample['input'].size(), transformed_sample['target'])