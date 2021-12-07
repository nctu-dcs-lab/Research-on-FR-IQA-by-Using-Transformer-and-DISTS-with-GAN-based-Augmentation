import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import create_dataloaders
from evaluate import calculate_correlation_coefficient
from module import Generator, MultiTask


def img_transform(img):
    img = img.numpy().transpose((0, 2, 3, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def get_activations(imgs, model):
    with torch.no_grad():
        pred = model(imgs)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    return pred.squeeze(3).squeeze(2).cpu().numpy()


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

    if load_netG_path == '' and load_netD_path == '':
        if log_dir:
            if os.path.isdir(log_dir):
                shutil.rmtree(log_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, datasets_sizes = create_dataloaders(data_dir, batch_size=batch_size, num_workers=num_workers)

    # Set Up Model
    netG = Generator().to(device)
    if load_netG_path:
        netG.load_state_dict(torch.load(load_netG_path))
    netD = MultiTask(pretrained=True).to(device)
    if load_netD_path:
        netD.load_state_dict(torch.load(load_netD_path))

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception = InceptionV3([block_idx]).to(device)

    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    optimizerG = optim.Adam(netG.parameters(), lr=lr['netG'])
    optimizerD = optim.Adam(netD.parameters(), lr=lr['netD'])

    if log_dir:
        writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

    iteration = 0
    for epoch in range(num_epochs):
        print(f'Epoch {start_epoch + epoch + 1}/{start_epoch + num_epochs}')
        print('-' * 10)

        epoch_err = {
            'train': {
                'real_clf': 0,
                'fake_clf': 0,
                'real_qual': 0,
                'fake_qual': 0,
                'cont': 0
            },
            'val': {
                'real_clf': 0,
                'real_qual': 0
            }
        }

        # Initial value for calculate FID score
        real_pred_arr = np.empty((datasets_sizes['train'], dims))
        fake_pred_arr = np.empty((datasets_sizes['train'], dims))
        start_idx = 0
        fid_value = math.inf

        # Initial value for calculate correlation coefficient
        plcc = {'train': 0, 'val': 0}
        srcc = {'train': 0, 'val': 0}
        krcc = {'train': 0, 'val': 0}

        for phase in ['train', 'val']:
            # Initial value for calculate correlation coefficient
            gt_qual = []
            pred_qual = []

            with tqdm(dataloaders[phase]) as tepoch:
                for ref_imgs, dist_imgs, scores, categories, origin_scores in tepoch:
                    ref_imgs = ref_imgs.to(device)
                    dist_imgs = dist_imgs.to(device)
                    scores = scores.to(device).float()
                    categories = categories.to(device)

                    # Format batch
                    batch_size = ref_imgs.size(0)

                    with torch.set_grad_enabled(phase == 'train'):
                        """
                        Discriminator
                        """
                        optimizerD.zero_grad()

                        """
                        Discriminator with real image
                        """
                        validity = torch.full((batch_size,), 1, dtype=torch.float, device=device)

                        pred_validity, pred_categories, pred_scores = netD(ref_imgs, dist_imgs)

                        D_x = pred_validity.mean().item()

                        errD_real_adv = bce_loss(pred_validity, validity)
                        errD_real_clf = ce_loss(pred_categories, categories)
                        errD_real_qual = mse_loss(pred_scores, scores)

                        errD_real = \
                            loss_weight['errD_real_adv'] * errD_real_adv + \
                            loss_weight['errD_real_clf'] * errD_real_clf + \
                            loss_weight['errD_real_qual'] * errD_real_qual

                        # Record original scores and predict scores
                        gt_qual.append(origin_scores)
                        pred_qual.append(pred_scores.cpu().detach())

                        """
                        Discriminator with fake image
                        """
                        # Generate batch of latent vectors
                        noise = torch.randn(batch_size, latent_dim, device=device)

                        fake_imgs = netG(ref_imgs,
                                         noise,
                                         scores.view(batch_size, -1),
                                         categories.view(batch_size, -1).float())
                        validity = torch.full((batch_size,), 0, dtype=torch.float, device=device)

                        pred_validity, pred_categories, _ = netD(ref_imgs, fake_imgs.detach())

                        D_G_z1 = pred_validity.mean().item()

                        errD_fake_adv = bce_loss(pred_validity, validity)
                        errD_fake_clf = ce_loss(pred_categories, categories)

                        errD_fake = \
                            loss_weight['errD_fake_adv'] * errD_fake_adv + \
                            loss_weight['errD_fake_clf'] * errD_fake_clf

                        errD = errD_real + errD_fake
                        if phase == 'train':
                            errD.backward()
                            optimizerD.step()

                        """
                        Generator
                        """
                        optimizerG.zero_grad()

                        validity = torch.full((batch_size,), 1, dtype=torch.float, device=device)

                        pred_validity, pred_categories, pred_scores = netD(ref_imgs, fake_imgs)

                        D_G_z2 = pred_validity.mean().item()

                        errG_adv = bce_loss(pred_validity, validity)
                        errG_clf = ce_loss(pred_categories, categories)
                        errG_qual = mse_loss(pred_scores, scores)
                        errG_cont = l1_loss(fake_imgs, dist_imgs)

                        errG = \
                            loss_weight['errG_adv'] * errG_adv + \
                            loss_weight['errG_clf'] * errG_clf + \
                            loss_weight['errG_qual'] * errG_qual + \
                            loss_weight['errG_cont'] * errG_cont

                        if phase == 'train':
                            errG.backward()
                            optimizerG.step()

                    if phase == 'train':
                        """
                        Record activations
                        """
                        with torch.no_grad():
                            real_pred = get_activations(dist_imgs, inception)
                            fake_pred = get_activations(fake_imgs.detach(), inception)

                        real_pred_arr[start_idx:start_idx + real_pred.shape[0]] = real_pred
                        fake_pred_arr[start_idx:start_idx + fake_pred.shape[0]] = fake_pred

                        start_idx = start_idx + real_pred.shape[0]

                    """
                    Record epoch loss
                    """
                    epoch_err[phase]['real_clf'] += errD_real_clf.item() * batch_size
                    epoch_err[phase]['real_qual'] += errD_real_qual.item() * batch_size
                    if phase == 'train':
                        epoch_err[phase]['fake_clf'] += errG_clf.item() * batch_size
                        epoch_err[phase]['fake_qual'] += errG_qual.item() * batch_size
                        epoch_err[phase]['cont'] += errG_cont.item() * batch_size

                    # Show training message
                    if phase == 'train':
                        tepoch.set_postfix({'Loss_D': errD.item(),
                                            'Loss_G': errG.item(),
                                            'D(x)': D_x,
                                            'D(G(z))': f'{D_G_z1: .4f} /{D_G_z2: .4f}'})
                        # Write logs by iteration
                        if log_dir and iteration % 100 == 0:
                            writer.add_scalars('Loss_netD',
                                               {'total': errD.item(),
                                                'weighted_real_adv': (
                                                    loss_weight['errD_real_adv'] * errD_real_adv).item(),
                                                'weighted_real_clf': (
                                                    loss_weight['errD_real_clf'] * errD_real_clf).item(),
                                                'weighted_real_qual': (
                                                    loss_weight['errD_real_qual'] * errD_real_qual).item(),
                                                'weighted_fake_adv': (
                                                    loss_weight['errD_fake_adv'] * errD_fake_adv).item(),
                                                'weighted_fake_clf': (
                                                    loss_weight['errD_fake_clf'] * errD_fake_clf).item()},
                                               iteration // 100)
                            writer.add_scalars('Loss_netG',
                                               {'total': errG.item(),
                                                'weighted_adv': (loss_weight['errG_adv'] * errG_adv).item(),
                                                'weighted_clf': (loss_weight['errG_clf'] * errG_clf).item(),
                                                'weighted_qual': (loss_weight['errG_qual'] * errG_qual).item(),
                                                'weighted_cont': (loss_weight['errG_cont'] * errG_cont).item()},
                                               iteration // 100)
                            writer.add_scalars('Loss_adversarial',
                                               {'netD_real': errD_real_adv.item(),
                                                'netD_fake': errD_fake_adv.item(),
                                                'netG_fake': errG_adv.item()},
                                               iteration // 100),
                            writer.add_scalars('Validity',
                                               {'D(x)': D_x, 'D(G(z))1': D_G_z1, 'D(G(z))2': D_G_z2},
                                               iteration // 100)
                            writer.add_images('Real Distorted Image',
                                              img_transform(dist_imgs.cpu().detach()),
                                              iteration // 100,
                                              dataformats='NHWC')
                            writer.add_images('Fake Distorted Image',
                                              img_transform(fake_imgs.cpu().detach()),
                                              iteration // 100,
                                              dataformats='NHWC')
                            writer.flush()
                    iteration += 1

            """
            Calculate FID score
            """
            if phase == 'train':
                real_mu = np.mean(real_pred_arr, axis=0)
                fake_mu = np.mean(fake_pred_arr, axis=0)

                real_sigma = np.cov(real_pred_arr, rowvar=False)
                fake_sigma = np.cov(fake_pred_arr, rowvar=False)

                fid_value = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

            """
            Calculate correlation coefficient
            """
            gt_qual = torch.cat(gt_qual).numpy()
            pred_qual = torch.cat(pred_qual).numpy()
            plcc[phase], srcc[phase], krcc[phase] = calculate_correlation_coefficient(gt_qual, pred_qual)

        # Write logs by epoch
        if log_dir:
            writer.add_scalars('Loss_classification',
                               {'train_real': epoch_err['train']['real_clf'] / datasets_sizes['train'],
                                'train_fake': epoch_err['train']['fake_clf'] / datasets_sizes['train'],
                                'val_real': epoch_err['val']['real_clf'] / datasets_sizes['val']},
                               start_epoch + epoch + 1)
            writer.add_scalars('Loss_quality',
                               {'train_real': epoch_err['train']['real_qual'] / datasets_sizes['train'],
                                'train_fake': epoch_err['train']['fake_qual'] / datasets_sizes['train'],
                                'val_real': epoch_err['val']['real_qual'] / datasets_sizes['val']},
                               start_epoch + epoch + 1)
            writer.add_scalars('PLCC', {x: plcc[x] for x in ['train', 'val']}, start_epoch + epoch + 1)
            writer.add_scalars('SRCC', {x: srcc[x] for x in ['train', 'val']}, start_epoch + epoch + 1)
            writer.add_scalars('KRCC', {x: krcc[x] for x in ['train', 'val']}, start_epoch + epoch + 1)
            writer.add_scalar('Loss_content',
                              epoch_err['train']['cont'] / datasets_sizes['train'],
                              start_epoch + epoch + 1)
            writer.add_scalar('FID', fid_value, start_epoch + epoch + 1)
            writer.flush()

        if model_dir:
            torch.save(netG.state_dict(), os.path.join(model_dir, f'netG_epoch{start_epoch + epoch + 1}.pth'))
            torch.save(netD.state_dict(), os.path.join(model_dir, f'netD_epoch{start_epoch + epoch + 1}.pth'))

    if log_dir:
        writer.close()


if __name__ == '__main__':
    num_experiment = 16

    main(
        data_dir=Path('../data/PIPAL(processed)/'),
        log_dir='',
        model_dir='',
        num_epochs=50,
        lr={'netG': 5e-5, 'netD': 1e-5},
    )
