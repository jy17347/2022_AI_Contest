import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader
from get_dataset import Audio_dataset

path_dataset = "./sample"


dataset = Audio_dataset(path_dataset)

for i in range(len(dataset)):
    sample = dataset[i]

    print(i, sample['path_wave'], sample['target'])
