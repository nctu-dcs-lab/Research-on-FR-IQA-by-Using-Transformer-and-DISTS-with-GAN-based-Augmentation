import argparse
import os
import shutil

from src.config.config import get_cfg_defaults
from src.tool.trainer import TrainerPhase1, TrainerPhase2, TrainerPhase3


def main(cfg):
    if cfg.TRAIN.WEIGHT_DIR and not os.path.isdir(cfg.TRAIN.WEIGHT_DIR):
        os.makedirs(cfg.TRAIN.WEIGHT_DIR)

    if cfg.TRAIN.LOG_DIR and os.path.isdir(cfg.TRAIN.LOG_DIR):
        shutil.rmtree(cfg.TRAIN.LOG_DIR)

    if cfg.TRAIN.PHASE == 1:
        trainer = TrainerPhase1(cfg)
    elif cfg.TRAIN.PHASE == 2:
        trainer = TrainerPhase2(cfg)
    else:
        trainer = TrainerPhase3(cfg)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Configuration YAML file for training')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    try:
        cfg.merge_from_file(args.config)
    except:
        print('Using default configuration file')
        if cfg.TRAIN.PHASE == 2:
            print('Incorrect to train phase2 without loading phase1 weight')

    assert cfg.MODEL.BACKBONE.NAME in ['VGG16', 'InceptionResNetV2']
    assert cfg.MODEL.BACKBONE.FEAT_LEVEL in ['low', 'medium', 'high', 'mixed']
    assert cfg.MODEL.EVALUATOR in ['IQT', 'DISTS', 'Transformer']

    if cfg.MODEL.BACKBONE.NAME == 'VGG16':
        cfg.MODEL.BACKBONE.CHANNELS = (3, 64, 128, 256, 512, 512)
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = (192 * 192, 192 * 192, 96 * 96, 48 * 48, 24 * 24, 12 * 12)
    elif cfg.MODEL.BACKBONE.FEAT_LEVEL == 'low':
        cfg.MODEL.BACKBONE.CHANNELS = tuple(320 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(21 * 21 for _ in range(6))
    elif cfg.MODEL.BACKBONE.FEAT_LEVEL == 'medium':
        cfg.MODEL.BACKBONE.CHANNELS = tuple(1088 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(10 * 10 for _ in range(6))
    elif cfg.MODEL.BACKBONE.FEAT_LEVEL == 'high':
        cfg.MODEL.BACKBONE.CHANNELS = tuple(2080 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(4 * 4 for _ in range(6))
    else:  # mixed
        cfg.MODEL.BACKBONE.CHANNELS = tuple(320 for _ in range(6)) + tuple(1088 for _ in range(6)) + tuple(
            2080 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(21 * 21 for _ in range(6)) + tuple(10 * 10 for _ in range(6)) + tuple(
            4 * 4 for _ in range(6))

    cfg.freeze()

    main(cfg)
