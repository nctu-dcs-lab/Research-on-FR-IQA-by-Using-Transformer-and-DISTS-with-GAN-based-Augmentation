import argparse
import os

import wandb

from src.config.config import get_cfg_defaults
from src.tool.trainer import TrainerPhase1, TrainerPhase2, TrainerPhase3


def main(cfg):
    if cfg.TRAIN.WEIGHT_DIR and not os.path.isdir(cfg.TRAIN.WEIGHT_DIR):
        os.makedirs(cfg.TRAIN.WEIGHT_DIR)

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
    assert cfg.MODEL.BACKBONE.FEAT_LEVEL in ['low', 'medium', 'high', 'mixed', 'reduced mixed']
    assert cfg.MODEL.EVALUATOR in ['IQT', 'DISTS', 'Transformer']

    # hyperparameter for phase 1

    # hyperparameter_defaults = dict(
    #     batch_size=cfg.DATASETS.BATCH_SIZE,
    #     img_size=cfg.DATASETS.IMG_SIZE[0],
    #     netD_learning_rate=cfg.TRAIN.LEARNING_RATE.NET_D,
    #     netG_learning_rate=cfg.TRAIN.LEARNING_RATE.NET_G,
    #     criterion_weight_errD_real_adv=cfg.TRAIN.CRITERION_WEIGHT.ERRD_REAL_ADV,
    #     criterion_weight_errD_real_clf=cfg.TRAIN.CRITERION_WEIGHT.ERRD_REAL_CLF,
    #     criterion_weight_errD_real_qual=cfg.TRAIN.CRITERION_WEIGHT.ERRD_REAL_QUAL,
    #     criterion_weight_errD_fake_adv=cfg.TRAIN.CRITERION_WEIGHT.ERRD_FAKE_ADV,
    #     criterion_weight_errD_fake_clf=cfg.TRAIN.CRITERION_WEIGHT.ERRD_FAKE_CLF,
    #     criterion_weight_errG_adv=cfg.TRAIN.CRITERION_WEIGHT.ERRG_ADV,
    #     criterion_weight_errG_clf=cfg.TRAIN.CRITERION_WEIGHT.ERRG_CLF,
    #     criterion_weight_errG_qual=cfg.TRAIN.CRITERION_WEIGHT.ERRG_QUAL,
    #     criterion_weight_errG_cont=cfg.TRAIN.CRITERION_WEIGHT.ERRG_CONT,
    #     latent_dim=cfg.MODEL.LATENT_DIM,
    #     fixed_backbone=cfg.MODEL.BACKBONE.FIXED,
    #     transformer_num_layers=cfg.MODEL.TRANSFORMER.TRANSFORMER_LAYERS,
    #     transformer_dim=cfg.MODEL.TRANSFORMER.TRANSFORMER_DIM,
    #     transformer_mha_num_heads=cfg.MODEL.TRANSFORMER.MHA_NUM_HEADS,
    #     transformer_feat_dim=cfg.MODEL.TRANSFORMER.FEAT_DIM,
    #     transformer_head_dim=cfg.MODEL.TRANSFORMER.HEAD_DIM
    # )

    hyperparameter_defaults = dict(
        batch_size=cfg.DATASETS.BATCH_SIZE,
        img_size=cfg.DATASETS.IMG_SIZE[0],
        netD_learning_rate=cfg.TRAIN.LEARNING_RATE.NET_D,
        fixed_backbone=cfg.MODEL.BACKBONE.FIXED,
        transformer_num_layers=cfg.MODEL.TRANSFORMER.TRANSFORMER_LAYERS,
        transformer_dim=cfg.MODEL.TRANSFORMER.TRANSFORMER_DIM,
        transformer_mha_num_heads=cfg.MODEL.TRANSFORMER.MHA_NUM_HEADS,
        transformer_feat_dim=cfg.MODEL.TRANSFORMER.FEAT_DIM,
        transformer_head_dim=cfg.MODEL.TRANSFORMER.HEAD_DIM
    )

    wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    cfg.DATASETS.BATCH_SIZE = config['batch_size']
    cfg.DATASETS.IMG_SIZE = (config['img_size'], config['img_size'])
    cfg.TRAIN.LEARNING_RATE.NET_D = config['netD_learning_rate']
    cfg.MODEL.BACKBONE.FIXED = config['fixed_backbone']
    cfg.MODEL.TRANSFORMER.TRANSFORMER_LAYERS = config['transformer_num_layers']
    cfg.MODEL.TRANSFORMER.TRANSFORMER_DIM = config['transformer_dim']
    cfg.MODEL.TRANSFORMER.MHA_NUM_HEADS = config['transformer_mha_num_heads']
    cfg.MODEL.TRANSFORMER.FEAT_DIM = config['transformer_feat_dim']
    cfg.MODEL.TRANSFORMER.HEAD_DIM = config['transformer_head_dim']

    cfg.freeze()

    main(cfg)
