import torch
import os
import pandas as pd
import natsort


class Audio_dataset():

    def __init__(self, folder):
        
        wav_dir = (folder + '/audio')
        wav_list = natsort.natsorted(os.listdir(wav_dir))
        label_dir = (folder + '/label')
        label_csv = pd.read_csv(label_dir + "/sample_labels.csv")
        
        data = []

        for wav in wav_list:
            label = label_csv[label_csv['file_name']==wav]['text'].tolist()[0]
            data.append((wav_dir+'/'+wav, label))

        self.data = data

    
    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):
        path,target = self.data[index]
        data = {'path_wave' : path, 'target' : target}

        return data
