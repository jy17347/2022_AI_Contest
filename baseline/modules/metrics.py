# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import Levenshtein as Lev

def get_metric(metric_name, vocab):
        
    if metric_name == 'CER':
        return CharacterErrorRate(vocab)

    else:
        raise ValueError('Unsupported metric: {0}'.format(metric_name))

class CharacterErrorRate(nn.Module):
    """
    Computes the Character Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to characters.
    """
    def __init__(self, vocab):
        super(CharacterErrorRate, self).__init__()
        self.vocab = vocab

    def metric(self, s1: str, s2: str):
        """
        Computes the Character Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to characters.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1 = s1.replace(' ', '')
        s2 = s2.replace(' ', '')

        # if '_' in sentence, means subword-unit, delete '_'
        if '_' in s1:
            s1 = s1.replace('_', '')

        if '_' in s2:
            s2 = s2.replace('_', '')

        dist = Lev.distance(s2, s1)
        length = len(s1.replace(' ', ''))

        return dist, length

    def forward(self, targets, y_hats):
        total_dist = 0
        total_length = 0

        for i in range(len(targets)):
            batch_targets = targets[i]
            batch_y_hats = y_hats[i]
            for (target, y_hat) in zip(batch_targets, batch_y_hats):
                s1 = self.vocab.label_to_string(target)
                s2 = self.vocab.label_to_string(y_hat)

                dist, length = self.metric(s1, s2)

                total_dist += dist
                total_length += length

        cer = total_dist / total_length
        return cer