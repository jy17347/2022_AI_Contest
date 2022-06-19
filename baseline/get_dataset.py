import torch
import os
import pandas as pd
import natsort


class Audio_dataset(torch.utils.data.Dataset):

    def __init__(self, folder, transform=None, silence_percentage=0.1):
        
        wav_dir = os.path.join(folder, 'audio')
        wav_dir = natsort.natsorted(os.listdir(wav_dir))
        label_dir = os.path.join(folder, 'label')
        label = pd.read_csv(label_dir + "/sample_labels.csv")
        
        data = []
        for wav in wav_dir:
            data.append((wav, label[label['file_name']==wav]['text'].tolist()[0]))

            

        # data = []
        # for c in all_classes:
        #     d = os.path.join(folder,c)
        #     target = class_to_idx[c]
        #     for f in os.listdir(d):
        #         path = os.path.join(d, f)
        #         data.append((path, target))

        # target = class_to_idx['silence']
        # data += [('', target)] * int(len(data) * silence_percentage)

        # self.classes = classes
        self.data = data
        self.transform = transform
    
    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):
        path,target = self.data[index]
        data = {'path_wave' : path, 'target' : target}

        if self.transform is not None:
            data = self.transform(data)

        return data
