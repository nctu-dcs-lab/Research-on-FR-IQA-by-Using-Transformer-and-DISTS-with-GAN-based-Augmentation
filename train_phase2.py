import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import create_dataloaders
from evaluate import calculate_correlation_coefficient
from module import Generator, MultiTask


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

    dataloaders, datasets_sizes = create_dataloaders(data_dir,
                                                     phase='phase2',
                                                     batch_size=batch_size,
                                                     num_workers=num_workers)

    # Set Up Model
    netG = Generator().to(device)
    netD = MultiTask(pretrained=True).to(device)

    netG.load_state_dict(torch.load(netG_path))
    netG.eval()
    netD.load_state_dict(torch.load(netD_path))

    mse_loss = nn.MSELoss()

    optimizer = optim.Adam(netD.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = SummaryWriter()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        epoch_real_loss = {'train': 0, 'val': 0}
        epoch_fake_loss = {'train': 0, 'val': 0}
        epoch_loss = {'train': 0, 'val': 0}

        # Initial value for calculate correlation coefficient
        plcc = {'train': 0, 'val': 0}
        srcc = {'train': 0, 'val': 0}
        krcc = {'train': 0, 'val': 0}

        for phase in ['train', 'val']:
            # Initial value for calculate correlation coefficient
            gt_qual = []
            pred_qual = []

            if phase == 'train':
                netD.train()
            else:
                netD.eval()

            with tqdm(dataloaders[phase]) as tepoch:
                for ref_imgs, dist_imgs, scores, categories, origin_scores in tepoch:
                    ref_imgs = ref_imgs.to(device)
                    dist_imgs = dist_imgs.to(device)
                    scores = scores.to(device).float()
                    categories = categories.to(device)

                    # Format batch
                    batch_size = ref_imgs.size(0)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        """
                        Deal with Real Distorted Images
                        """
                        _, _, pred_scores = netD(ref_imgs, dist_imgs)

                        real_loss = mse_loss(pred_scores, scores)

                        # Record original scores and predict scores
                        gt_qual.append(origin_scores)
                        pred_qual.append(pred_scores.cpu().detach())

                        """
                        Deal with Fake Distorted Images
                        """
                        # Generate batch of latent vectors
                        noise = torch.randn(batch_size, latent_dim, device=device)

                        fake_imgs = netG(ref_imgs,
                                         noise,
                                         scores.view(batch_size, -1),
                                         categories.view(batch_size, -1).float())

                        _, _, pred_scores = netD(ref_imgs, fake_imgs.detach())

                        fake_loss = mse_loss(pred_scores, scores)

                        loss = real_loss + fake_loss

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    epoch_real_loss[phase] += real_loss.item() * batch_size
                    epoch_fake_loss[phase] += fake_loss.item() * batch_size
                    epoch_loss[phase] += loss.item() * batch_size

                    # Show training message
                    tepoch.set_postfix({'Real Loss': real_loss.item(),
                                        'Fake Loss': fake_loss.item(),
                                        'Total Loss': loss.item()})
            if phase == 'train':
                scheduler.step()

            epoch_real_loss[phase] /= datasets_sizes[phase]
            epoch_fake_loss[phase] /= datasets_sizes[phase]
            epoch_loss[phase] /= datasets_sizes[phase]

            """
            Calculate correlation coefficient
            """
            plcc[phase], srcc[phase], krcc[phase] = calculate_correlation_coefficient(gt_qual, pred_qual)

            print(f'{phase} loss: {epoch_loss[phase]}')
        if log_dir:
            writer.add_scalars('Loss', {'train_real': epoch_real_loss['train'],
                                        'val_real': epoch_real_loss['val'],
                                        'train_fake': epoch_fake_loss['train'],
                                        'val_fake': epoch_fake_loss['val']}, epoch + 1)
            writer.add_scalars('PLCC', {x: plcc[x] for x in ['train', 'val']}, epoch + 1)
            writer.add_scalars('SRCC', {x: srcc[x] for x in ['train', 'val']}, epoch + 1)
            writer.add_scalars('KRCC', {x: krcc[x] for x in ['train', 'val']}, epoch + 1)
            writer.flush()

        if save_model_dir:
            torch.save(netD.state_dict(), os.path.join(save_model_dir, f'netD_epoch{epoch + 1}.pth'))

    writer.close()


if __name__ == '__main__':
    main(
        netG_path=os.path.expanduser(''),
        netD_path=os.path.expanduser(''),
        data_dir=Path('../data/PIPAL(processed)/'),
        log_dir='',
        save_model_dir='',
        lr=5e-5,
        num_epochs=100
    )
