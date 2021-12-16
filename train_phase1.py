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

from dataset import create_dataloaders
from evaluate import evaluate_phase1
from log import write_epoch_log
from module import Generator, MultiTask
from train import train_phase1

torch.multiprocessing.set_sharing_strategy('file_system')


def main(data_dir,
         model_dir='',
         log_dir='',
         batch_size=16,
         num_epochs=25,
         num_workers=10,
         lr=None,
         latent_dim=100,
         dims=2048,
         loss_weight=None,
         load_netG_path='',
         load_netD_path='',
         start_epoch=0):
    if lr is None:
        lr = {'netG': 1e-5, 'netD': 1e-5}

    if loss_weight is None:
        loss_weight = {
            'errD_real_adv': 1,
            'errD_real_clf': 1,
            'errD_real_qual': 1,
            'errD_fake_adv': 1,
            'errD_fake_clf': 1,
            'errG_adv': 1,
            'errG_clf': 1,
            'errG_qual': 1,
            'errG_cont': 1
        }

    if model_dir:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

    if log_dir and os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, datasets_size = create_dataloaders(
        data_dir,
        phase='phase1',
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Set Up Model
    model = {
        'netG': Generator().to(device),
        'netD': MultiTask(pretrained=True).to(device),
        'inception': InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(device)
    }

    if load_netG_path:
        model['netG'].load_state_dict(torch.load(load_netG_path))
    if load_netD_path:
        model['netD'].load_state_dict(torch.load(load_netD_path))

    loss = {
        'bce_loss': nn.BCELoss(),
        'l1_loss': nn.L1Loss(),
        'mse_loss': nn.MSELoss(),
        'ce_loss': nn.CrossEntropyLoss()
    }

    optimizer = {
        'optimizerG': optim.Adam(model['netG'].parameters(), lr=lr['netG']),
        'optimizerD': optim.Adam(model['netD'].parameters(), lr=lr['netD'])
    }

    scheduler = {
        'schedulerG': CosineAnnealingWarmRestarts(optimizer['optimizerG'], T_0=1, T_mult=2),
        'schedulerD': CosineAnnealingWarmRestarts(optimizer['optimizerD'], T_0=1, T_mult=2)
    }
    if start_epoch != 0:
        scheduler['schedulerG'].step(start_epoch)
        scheduler['schedulerD'].step(start_epoch)

    if log_dir:
        writer: SummaryWriter = SummaryWriter(log_dir=log_dir)
    else:
        writer: SummaryWriter = SummaryWriter()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f'Epoch {epoch + 1}/{start_epoch + num_epochs}')
        print('-' * 10)

        results = {
            'train': train_phase1(
                dataloaders['train'],
                model,
                optimizer,
                loss,
                loss_weight,
                epoch * math.ceil(datasets_size['train'] / batch_size),
                latent_dim,
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

        if model_dir:
            torch.save(model['netG'].state_dict(), os.path.join(model_dir, f'netG_epoch{epoch + 1}.pth'))
            torch.save(model['netD'].state_dict(), os.path.join(model_dir, f'netD_epoch{epoch + 1}.pth'))

    writer.close()


if __name__ == '__main__':
    num_experiment = 3
    main(
        data_dir=Path('../data/PIPAL(processed)/'),
        log_dir=os.path.expanduser(f'~/nfs/result/acgan-iqt/phase1/experiment{num_experiment}/logs'),
        model_dir=os.path.expanduser(f'~/nfs/result/acgan-iqt/phase1/experiment{num_experiment}/models'),
        num_epochs=100,
        lr={'netG': 5e-5, 'netD': 1e-5},
        loss_weight={
            'errD_real_adv': 1,
            'errD_real_clf': 0.1,
            'errD_real_qual': 1,
            'errD_fake_adv': 1,
            'errD_fake_clf': 0.01,
            'errG_adv': 0.01,
            'errG_clf': 0.05,
            'errG_qual': 1,
            'errG_cont': 0.1
        }
    )
