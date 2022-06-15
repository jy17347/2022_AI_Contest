"""Predict
"""

import hydra
import warnings
from omegaconf import OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore
from kospeech.evaluator import EvalConfig
from kospeech.data.audio import FilterBankConfig
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.label_loader import load_dataset
from kospeech.data.data_loader import SpectrogramDataset
from kospeech.evaluator.evaluator import Evaluator
from kospeech.utils import check_envirionment, logger
from kospeech.model_builder import load_test_model
from datetime import datetime, timezone, timedelta
import numpy as np
import random
import os
import torch
from kospeech.model_builder import build_model


def inference(config: DictConfig):
    PROJECT_DIR = os.path.dirname(__file__)

    # Serial
    train_serial = config.eval.train_serial
    kst = timezone(timedelta(hours=9))
    predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
    predict_serial = train_serial + '_' + predict_timestamp

    # Seed
    torch.manual_seed(config.eval.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.eval.seed)
    random.seed(config.eval.seed)
    torch.cuda.manual_seed_all(config.eval.seed)
    device = check_envirionment(config.eval.use_cuda, config.eval.gpus)

    """
    01. Set Vocabulary
    """
    vocab = KsponSpeechVocabulary(
        os.path.join(PROJECT_DIR, 'data', 'vocab', 'labels.csv'),
        output_unit=config.eval.output_unit,
    )

    """
    02. Load data
    """
    audio_paths, transcripts = load_dataset(os.path.join(PROJECT_DIR, config.eval.transcripts_path))
    testset = SpectrogramDataset(audio_paths=audio_paths, transcripts=transcripts,
                                 sos_id=vocab.sos_id, eos_id=vocab.eos_id,
                                 dataset_path=config.eval.dataset_path,  config=config, spec_augment=False)

    """
    03. Set model
    """
    config.eval.model_path = os.path.join(PROJECT_DIR, config.eval.model_path)
    model = build_model(config, vocab, device)
    model = load_test_model(model, config.eval)


    """
    04. Set evaluator
    """
    evaluator = Evaluator(
        dataset=testset,
        vocab=vocab,
        batch_size=config.eval.batch_size,
        device=device,
        num_workers=config.eval.num_workers,
        print_every=config.eval.print_every,
        decode=config.eval.decode,
        beam_size=config.eval.k,
    )

    evaluator.evaluate(model, audio_paths, os.path.join(PROJECT_DIR, 'results', 'predict', predict_serial, 'submission.csv'))


cs = ConfigStore.instance()
cs.store(group="eval", name="default", node=EvalConfig, package="eval")
cs.store(group="audio", name="fbank", node=FilterBankConfig, package="audio")


@hydra.main(config_path='config', config_name="eval")
def main(config: DictConfig) -> None:
    warnings.filterwarnings('ignore')
    logger.info(OmegaConf.to_yaml(config))
    inference(config)

if __name__ == '__main__':
    main()
