import argparse
import os
import shutil

from src.config.config_phase1 import get_cfg_defaults
from src.tool.trainer import TrainerPhase1


def main(cfg):
    if cfg.TRAIN.WEIGHT_DIR:
        if not os.path.isdir(cfg.TRAIN.WEIGHT_DIR):
            os.makedirs(cfg.TRAIN.WEIGHT_DIR)

    if cfg.TRAIN.LOG_DIR and os.path.isdir(cfg.TRAIN.LOG_DIR):
        shutil.rmtree(cfg.TRAIN.LOG_DIR)

    trainer = TrainerPhase1(cfg)
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Configuration YAML file for train phase1')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    try:
        cfg.merge_from_file(args.config)
    except:
        print('Using default configuration file')

    cfg.freeze()
    main(cfg)
