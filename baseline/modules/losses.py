"""loss 정의
"""

import torch.nn as nn


def get_loss(vocab, config):

    if config.train.loss_name == 'CTC':
        return nn.CTCLoss(blank=vocab.blank_id, reduction=config.train.reduction, zero_infinity=True)

    else:
        raise ValueError('Unsupported loss: {0}'.format(config.train.loss_name))
