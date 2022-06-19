import torch
from torchvision import transforms
from transform import LoadAudio
from transform import FixAudioLength
from transform import ToMelSpectrogram
from transform import ToTensor
import numpy as np
from torch.utils.data import Dataset, DataLoader
from get_dataset import Audio_dataset
from tqdm import tqdm

path_dataset = "./sample"
n_mels=40
batch_size=4
use_gpu=False

num_dataloader_workers=1


dataset = Audio_dataset(path_dataset,transforms.Compose([LoadAudio(), FixAudioLength(),
                                         		ToMelSpectrogram(n_mels=n_mels), 
                                         		ToTensor('mel_spectrogram', 'input')]))


dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        pin_memory=use_gpu,
                                        num_workers=0)


for batch in tqdm(dataloader, unit='audios', unit_scale=dataloader.batch_size):
    inputs = batch['input']
    targets = batch['target']
    print(len(inputs), len(targets))