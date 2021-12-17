import argparse
import math
import os
import shutil
from pathlib import Path

import torch
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
from pytorch_fid.inception import InceptionV3
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from src.config.config_phase1 import get_cfg_defaults
from src.data.dataset import create_dataloaders
from src.modeling.evaluate import evaluate_phase1
from src.modeling.log import write_epoch_log
from src.modeling.module import Generator, MultiTask
from src.tool.train import train_phase1

torch.multiprocessing.set_sharing_strategy('file_system')


def main(cfg):
    if cfg.TRAIN.WEIGHT_DIR:
        if not os.path.isdir(cfg.TRAIN.WEIGHT_DIR):
            os.makedirs(cfg.TRAIN.WEIGHT_DIR)

    if cfg.TRAIN.LOG_DIR and os.path.isdir(cfg.TRAIN.LOG_DIR):
        shutil.rmtree(cfg.TRAIN.LOG_DIR)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, datasets_size = create_dataloaders(
        Path(cfg.DATASETS.ROOT_DIR),
        phase=1,
        batch_size=cfg.DATASETS.BATCH_SIZE,
        num_workers=cfg.DATASETS.NUM_WORKERS
    )

    # Set Up Model
    model = {
        'netG': Generator().to(device),
        'netD': MultiTask(pretrained=True).to(device),
        'inception': InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[cfg.MODEL.INCEPTION_DIMS]]).to(device)
    }

    if cfg.TRAIN.RESUME.NET_G:
        model['netG'].load_state_dict(torch.load(cfg.TRAIN.RESUME.NET_G))
    if cfg.TRAIN.RESUME.NET_D:
        model['netD'].load_state_dict(torch.load(cfg.TRAIN.RESUME.NET_D))

    loss = {
        'bce_loss': nn.BCELoss(),
        'l1_loss': nn.L1Loss(),
        'mse_loss': nn.MSELoss(),
        'ce_loss': nn.CrossEntropyLoss()
    }

    optimizer = {
        'optimizerG': optim.Adam(model['netG'].parameters(), lr=cfg.TRAIN.LEARNING_RATE.NET_G),
        'optimizerD': optim.Adam(model['netD'].parameters(), lr=cfg.TRAIN.LEARNING_RATE.NET_D)
    }

    scheduler = {
        'schedulerG': CosineAnnealingWarmRestarts(optimizer['optimizerG'], T_0=1, T_mult=2),
        'schedulerD': CosineAnnealingWarmRestarts(optimizer['optimizerD'], T_0=1, T_mult=2)
    }
    if cfg.TRAIN.START_EPOCH != 0:
        scheduler['schedulerG'].step(cfg.TRAIN.START_EPOCH)
        scheduler['schedulerD'].step(cfg.TRAIN.START_EPOCH)

    if cfg.TRAIN.LOG_DIR:
        writer: SummaryWriter = SummaryWriter(log_dir=cfg.TRAIN.LOG_DIR)
    else:
        writer: SummaryWriter = SummaryWriter()

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.START_EPOCH + cfg.TRAIN.NUM_EPOCHS):
        print(f'Epoch {epoch + 1}/{cfg.TRAIN.START_EPOCH + cfg.TRAIN.NUM_EPOCHS}')
        print('-' * 10)

        results = {
            'train': train_phase1(
                dataloaders['train'],
                model,
                optimizer,
                loss,
                cfg,
                epoch * math.ceil(datasets_size['train'] / cfg.DATASETS.BATCH_SIZE),
                datasets_size['train'],
                writer,
                device
            ),
            'val': evaluate_phase1(
                dataloaders['val'],
                model,
                loss,
                datasets_size['val'],
                device
            )
        }

        scheduler['schedulerG'].step()
        scheduler['schedulerD'].step()

        write_epoch_log(writer, results, epoch + 1)

        if cfg.TRAIN.WEIGHT_DIR:
            torch.save(model['netG'].state_dict(), os.path.join(cfg.TRAIN.WEIGHT_DIR, f'netG_epoch{epoch + 1}.pth'))
            torch.save(model['netD'].state_dict(), os.path.join(cfg.TRAIN.WEIGHT_DIR, f'netD_epoch{epoch + 1}.pth'))

    writer.close()


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
