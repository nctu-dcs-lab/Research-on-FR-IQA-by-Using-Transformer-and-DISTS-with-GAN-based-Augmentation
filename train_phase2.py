import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from dataset import create_dataloaders
from evaluate import evaluate_phase2
from module import Generator, MultiTask
from train import train_phase2


def main(netG_path,
         netD_path,
         data_dir,
         save_model_dir='',
         log_dir='',
         batch_size=16,
         num_epochs=25,
         num_workers=10,
         lr=1e-5,
         latent_dim=100):
    if save_model_dir:
        if not os.path.isdir(save_model_dir):
            os.makedirs(save_model_dir)

    if log_dir:
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, datasets_sizes = create_dataloaders(
        data_dir,
        phase='phase2',
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Set Up Model
    model = {
        'netG': Generator().to(device),
        'netD': MultiTask(pretrained=True).to(device)
    }

    model['netG'].load_state_dict(torch.load(netG_path))
    model['netG'].eval()
    model['netD'].load_state_dict(torch.load(netD_path))

    loss = {
        'mse_loss': nn.MSELoss()
    }

    optimizer = optim.Adam(model['netD'].parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = SummaryWriter()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        results = {
            'train': train_phase2(
                dataloaders['train'],
                model,
                optimizer,
                scheduler,
                loss,
                latent_dim,
                datasets_sizes['train'],
                device
            ),
            'val': evaluate_phase2(
                dataloaders['val'],
                model,
                loss,
                latent_dim,
                datasets_sizes['val'],
                device
            )
        }

        writer.add_scalars(
            'Loss', {
                'train_real': results['train']['real_loss'],
                'val_real': results['val']['real_loss'],
                'train_fake': results['train']['fake_loss'],
                'val_fake': results['val']['fake_loss']
            },
            epoch + 1
        )
        writer.add_scalars('PLCC', {x: results[x]['PLCC'] for x in ['train', 'val']}, epoch + 1)
        writer.add_scalars('SRCC', {x: results[x]['SRCC'] for x in ['train', 'val']}, epoch + 1)
        writer.add_scalars('KRCC', {x: results[x]['KRCC'] for x in ['train', 'val']}, epoch + 1)
        writer.flush()

        if save_model_dir:
            torch.save(model['netD'].state_dict(), os.path.join(save_model_dir, f'netD_epoch{epoch + 1}.pth'))

    writer.close()


if __name__ == '__main__':
    num_experiment = 1
    main(
        netG_path=os.path.expanduser('~/nfs/result/acgan-iqt/phase1/experiment2/models/netG_epoch100.pth'),
        netD_path=os.path.expanduser('~/nfs/result/acgan-iqt/phase1/experiment2/models/netD_epoch100.pth'),
        data_dir=Path('../data/PIPAL(processed)/'),
        log_dir=os.path.expanduser(f'~/nfs/result/acgan-iqt/phase2/experiment{num_experiment}/logs'),
        save_model_dir=os.path.expanduser(f'~/nfs/result/acgan-iqt/phase2/experiment{num_experiment}/models'),
        lr=5e-5,
        num_epochs=100
    )
