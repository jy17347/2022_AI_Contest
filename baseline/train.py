"""학습 스크립트
"""

from modules.utils import get_logger
from modules.earlystoppers import EarlyStopper
from modules.recorders import Recorder
from modules.trainer import Trainer

from modules.optimizers import get_optimizer
from modules.metrics import get_metric
from modules.losses import get_loss

from datetime import datetime, timezone, timedelta
import numpy as np
import random, copy, os, wandb, queue, hydra, warnings, torch, math

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
import torch.nn as nn
from kospeech.vocabs import KsponSpeechVocabulary
from kospeech.utils import (
    check_envirionment,
    get_optimizer,
    get_lr_scheduler
)
from kospeech.data.audio import (
    FilterBankConfig,
    MelSpectrogramConfig,
    MfccConfig,
    SpectrogramConfig,
)
from kospeech.data.label_loader import load_dataset
from kospeech.data.data_loader import SpectrogramDataset

from kospeech.trainer import (
    DeepSpeech2TrainConfig
)
from kospeech.optim import Optimizer
from kospeech.data import (
    MultiDataLoader,
    AudioDataLoader
)
from kospeech.data.data_loader import split_dataset
from kospeech.model_builder import build_model


def train(config: DictConfig) -> nn.DataParallel:

    PROJECT_DIR = os.path.dirname(__file__)
    # Train Serial
    kst = timezone(timedelta(hours=9))
    train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

    # Recorder directory
    RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)
    os.makedirs(RECORDER_DIR, exist_ok=True)

    # Seed
    torch.manual_seed(config.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)
    device = check_envirionment(config.train.use_cuda, config.train.gpus)

    """
    00. Set Logger
    """
    logger = get_logger(name='train', dir_=RECORDER_DIR, stream=False)
    logger.info(f"Set Logger {RECORDER_DIR}")
    logger.info(OmegaConf.to_yaml(config))

    """
    01. Set Vocabulary
    """
    vocab = KsponSpeechVocabulary(
        os.path.join(PROJECT_DIR, 'data', 'vocab','labels.csv'),
        output_unit=config.train.output_unit,
    )

    """
    02. Load data
    """
    train_time_step, valid_time_step, trainset_list, validset = split_dataset(config, os.path.join(PROJECT_DIR, config.train.transcripts_path), vocab, 0.2, config.train.seed)

    for trainset in trainset_list:
        trainset.shuffle()


    """
    03. Set model
    """
    model = build_model(config, vocab, device)

    """
    04. Set trainer
    """
    # Optimizer
    optimizer = get_optimizer(model, config)
    lr_scheduler = get_lr_scheduler(config, optimizer, train_time_step)
    optimizer = Optimizer(optimizer, lr_scheduler, config.train.total_steps, config.train.max_grad_norm)

    # Loss
    loss = get_loss(vocab, config)

    # Metric
    metric = get_metric(metric_name='CER', vocab=vocab)

    # Early stoppper
    early_stopper = EarlyStopper(patience=config.train.early_stopping_patience,
                                 mode=config.train.early_stopping_mode,
                                 logger=logger)

    # Recorder
    recorder = Recorder(record_dir=RECORDER_DIR,
                        model=model,
                        optimizer=optimizer,
                        scheduler=None,
                        logger=logger)

    # !Wandb
    if config.train.wandb:
        wandb_project_serial = 'kospeech'
        wandb_username = '#' # 수정하세요.
        wandb.init(project=wandb_project_serial, dir=RECORDER_DIR, entity=wandb_username)
        wandb.run.name = train_serial
        wandb.config.update(config)
        wandb.watch(model)

    # Trainer
    n_epochs = config.train.num_epochs
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss=loss,
                      metric=metric,
                      device=device,
                      logger=logger,
                      n_epochs=n_epochs,
                      num_workers=config.train.num_workers,
                      interval=config.train.print_every)

    """
    05. TRAIN
    """
    # Train
    for epoch_index in range(n_epochs):

        train_queue = queue.Queue(config.train.num_workers << 1)
        train_loader = MultiDataLoader(
            trainset_list, train_queue, config.train.batch_size, config.train.num_workers, vocab.pad_id
        )
        train_loader.start()

        # Set Recorder row
        row_dict = dict()
        row_dict['epoch_index'] = epoch_index
        row_dict['train_serial'] = train_serial

        """
        Train
        """
        trainer.train(queue=train_queue, epoch_time_step=train_time_step, mode='train', epoch_index=epoch_index)

        row_dict['train_loss'] = trainer.loss_mean
        row_dict['train_elapsed_time'] = trainer.elapsed_time

        train_loader.join()

        for metric_str, score in trainer.score_dict.items():
            row_dict[f"train_{metric_str}"] = score
        trainer.clear_history()

        """
        Validation
        """
        valid_queue = queue.Queue(config.train.num_workers << 1)
        # valid_loader = AudioDataLoader(validset, valid_queue, config.train.batch_size, 0, vocab.pad_id)
        valid_loader = MultiDataLoader(
            validset, valid_queue, config.train.batch_size, config.train.num_workers, vocab.pad_id
        )
        valid_loader.start()

        trainer.train(queue=valid_queue, epoch_time_step=valid_time_step, mode='val', epoch_index=epoch_index)

        row_dict['val_loss'] = trainer.loss_mean
        row_dict['val_elapsed_time'] = trainer.elapsed_time

        for metric_str, score in trainer.score_dict.items():
            row_dict[f"val_{metric_str}"] = score
        trainer.clear_history()

        """
        Record
        """
        recorder.add_row(row_dict)
        recorder.save_plot(['loss', 'CER'])

        # !WANDB
        if config.train.wandb:
            wandb.log(row_dict)

        """
        Early Stop Check
        """
        early_stopping_target = config.train.early_stopping_target
        early_stopper.check_early_stopping(loss=row_dict[early_stopping_target])

        if early_stopper.patience_counter == 0:
            recorder.save_weight(epoch_index)
            best_row_dict = copy.deepcopy(row_dict)

        if early_stopper.stop == True:
            logger.info(
                f"Eearly stopped, coutner {early_stopper.patience_counter}/{config.train.early_stopping_patience}")

            if config.train.wandb:
                wandb.log(best_row_dict)
            break
        torch.cuda.empty_cache()

cs = ConfigStore.instance()
cs.store(group="audio", name="fbank", node=FilterBankConfig, package="audio")
cs.store(group="audio", name="melspectrogram", node=MelSpectrogramConfig, package="audio")
cs.store(group="audio", name="mfcc", node=MfccConfig, package="audio")
cs.store(group="audio", name="spectrogram", node=SpectrogramConfig, package="audio")
cs.store(group="train", name="ds2_train", node=DeepSpeech2TrainConfig, package="train")

@hydra.main(config_path=os.path.join('.', "config"), config_name="train")
def main(config: DictConfig) -> None:
    warnings.filterwarnings('ignore')
    train(config)

if __name__ == '__main__':
    main()



