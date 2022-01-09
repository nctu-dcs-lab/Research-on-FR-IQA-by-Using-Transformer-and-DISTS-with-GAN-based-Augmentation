import math
import os
from pathlib import Path

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch import optim, nn
from torch.nn.functional import adaptive_avg_pool2d
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.modeling.module import Generator, MultiTask
from src.tool.evaluate import calculate_correlation_coefficient
from src.tool.log import write_iteration_log, write_epoch_log


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


class Trainer:
    def __init__(self, cfg):

        self.dataloaders, self.datasets_size = create_dataloaders(
            Path(cfg.DATASETS.ROOT_DIR),
            phase=cfg.TRAIN.PHASE,
            batch_size=cfg.DATASETS.BATCH_SIZE,
            num_workers=cfg.DATASETS.NUM_WORKERS
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.netG = Generator().to(self.device)
        self.netD = MultiTask(cfg).to(self.device)
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[cfg.MODEL.INCEPTION_DIMS]]).to(self.device)

        if cfg.TRAIN.RESUME.NET_G:
            self.netG.load_state_dict(torch.load(cfg.TRAIN.RESUME.NET_G))
        if cfg.TRAIN.RESUME.NET_D:
            self.netD.load_state_dict(torch.load(cfg.TRAIN.RESUME.NET_D))

        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=cfg.TRAIN.LEARNING_RATE.NET_G)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=cfg.TRAIN.LEARNING_RATE.NET_D)

        self.schedulerG = CosineAnnealingWarmRestarts(self.optimizerG, T_0=1, T_mult=2)
        self.schedulerD = CosineAnnealingWarmRestarts(self.optimizerD, T_0=1, T_mult=2)

        if cfg.TRAIN.START_EPOCH != 0:
            self.schedulerG.step(cfg.TRAIN.START_EPOCH)
            self.schedulerG.step(cfg.TRAIN.START_EPOCH)

        if cfg.TRAIN.LOG_DIR:
            self.writer: SummaryWriter = SummaryWriter(log_dir=cfg.TRAIN.LOG_DIR)
        else:
            self.writer: SummaryWriter = SummaryWriter()

        self.start_epoch = cfg.TRAIN.START_EPOCH
        self.num_epoch = cfg.TRAIN.NUM_EPOCHS
        self.criterion_weight = cfg.TRAIN.CRITERION_WEIGHT
        self.iteration = self.start_epoch * math.ceil(self.datasets_size['train'] / cfg.DATASETS.BATCH_SIZE)
        self.weight_dir = cfg.TRAIN.WEIGHT_DIR

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epoch):
            print(f'Epoch {epoch + 1}/{self.start_epoch + self.num_epoch}')
            print('-' * 10)

            results = {
                'train': self.epoch_train(),
                'val': self.epoch_eval()
            }

            self.schedulerG.step()
            self.schedulerD.step()

            self.write_epoch_log(results, epoch + 1)

            if self.weight_dir:
                self.save_weight(epoch + 1)
        self.writer.close()

    def epoch_train(self):
        pass

    def epoch_eval(self):
        pass

    def write_epoch_log(self, results, epoch):
        pass

    def save_weight(self, epoch):
        torch.save(self.netD.state_dict(), os.path.join(self.weight_dir, f'netD_epoch{epoch}.pth'))


class TrainerPhase1(Trainer):
    def __init__(self, cfg):
        super(TrainerPhase1, self).__init__(cfg=cfg)
        self.netG.train()
        self.netD.train()
        self.latent_dim = cfg.MODEL.LATENT_DIM

    def epoch_train(self):
        record = {
            'gt_scores': [],
            'pred_scores': [],
            'inception_real_pred': [],
            'inception_fake_pred': []
        }

        result = {
            'real_clf': 0,
            'real_qual': 0,
            'fake_clf': 0,
            'fake_qual': 0,
            'cont': 0
        }

        with tqdm(self.dataloaders['train']) as tepoch:
            for ref_imgs, dist_imgs, scores, categories, origin_scores in tepoch:
                ref_imgs = ref_imgs.to(self.device)
                dist_imgs = dist_imgs.to(self.device)
                scores = scores.to(self.device).float()
                categories = categories.to(self.device)

                # Format batch
                bs = ref_imgs.size(0)

                """
                Discriminator
                """
                self.optimizerD.zero_grad()

                """
                Discriminator with real image
                """
                validity = torch.full((bs,), 1, dtype=torch.float, device=self.device)

                pred_validity, pred_categories, pred_scores = self.netD(ref_imgs, dist_imgs)

                record['D_x'] = pred_validity.mean().item()

                errD_real_adv = self.bce_loss(pred_validity, validity)
                errD_real_clf = self.ce_loss(pred_categories, categories)
                errD_real_qual = self.mse_loss(pred_scores, scores)

                record['errD_real_adv'] = errD_real_adv.item()
                record['errD_real_clf'] = errD_real_clf.item()
                record['errD_real_qual'] = errD_real_qual.item()

                errD_real = \
                    self.criterion_weight['ERRD_REAL_ADV'] * errD_real_adv + \
                    self.criterion_weight['ERRD_REAL_CLF'] * errD_real_clf + \
                    self.criterion_weight['ERRD_REAL_QUAL'] * errD_real_qual

                # Record original scores and predict scores
                record['gt_scores'].append(origin_scores)
                record['pred_scores'].append(pred_scores.cpu().detach())

                """
                Discriminator with fake image
                """
                # Generate batch of latent vectors
                noise = torch.randn(bs, self.latent_dim, device=self.device)

                fake_imgs = self.netG(ref_imgs,
                                      noise,
                                      scores.view(bs, -1),
                                      categories.view(bs, -1).float())
                validity = torch.full((bs,), 0, dtype=torch.float, device=self.device)

                pred_validity, pred_categories, _ = self.netD(ref_imgs, fake_imgs.detach())

                record['D_G_z1'] = pred_validity.mean().item()

                errD_fake_adv = self.bce_loss(pred_validity, validity)
                errD_fake_clf = self.ce_loss(pred_categories, categories)

                record['errD_fake_adv'] = errD_fake_adv.item()
                record['errD_fake_clf'] = errD_fake_clf.item()

                errD_fake = \
                    self.criterion_weight['ERRD_FAKE_ADV'] * errD_fake_adv + \
                    self.criterion_weight['ERRD_FAKE_CLF'] * errD_fake_clf

                errD = errD_real + errD_fake
                record['errD'] = errD.item()

                errD.backward()
                self.optimizerD.step()

                """
                Generator
                """
                self.optimizerG.zero_grad()

                validity = torch.full((bs,), 1, dtype=torch.float, device=self.device)

                pred_validity, pred_categories, pred_scores = self.netD(ref_imgs, fake_imgs)

                record['D_G_z2'] = pred_validity.mean().item()

                errG_adv = self.bce_loss(pred_validity, validity)
                errG_clf = self.ce_loss(pred_categories, categories)
                errG_qual = self.mse_loss(pred_scores, scores)
                errG_cont = self.l1_loss(fake_imgs, dist_imgs)

                record['errG_adv'] = errG_adv.item()
                record['errG_clf'] = errG_clf.item()
                record['errG_qual'] = errG_qual.item()
                record['errG_cont'] = errG_cont.item()

                errG = \
                    self.criterion_weight['ERRG_ADV'] * errG_adv + \
                    self.criterion_weight['ERRG_CLF'] * errG_clf + \
                    self.criterion_weight['ERRG_QUAL'] * errG_qual + \
                    self.criterion_weight['ERRG_CONT'] * errG_cont

                record['errG'] = errG.item()

                errG.backward()
                self.optimizerG.step()

                record['real_imgs'] = img_transform(dist_imgs.cpu().detach())
                record['fake_imgs'] = img_transform(fake_imgs.cpu().detach())

                """
                Record activations
                """
                with torch.no_grad():
                    record['inception_real_pred'].append(get_activations(dist_imgs, self.inception))
                    record['inception_fake_pred'].append(get_activations(fake_imgs.detach(), self.inception))

                """
                Show logs
                """
                tepoch.set_postfix({
                    'Loss_D': record['errD'],
                    'Loss_G': record['errG'],
                    'D(x)': record['D_x'],
                    'D(G(z))': f'{record["D_G_z1"]: .4f}/{record["D_G_z2"]: .4f}'
                })

                if self.iteration % 100 == 0:
                    write_iteration_log(self.writer, record, self.iteration, self.criterion_weight)

                """
                Record epoch loss
                """
                result['real_clf'] += record['errD_real_clf'] * bs
                result['real_qual'] += record['errD_real_qual'] * bs
                result['fake_clf'] += record['errG_clf'] * bs
                result['fake_qual'] += record['errG_qual'] * bs
                result['cont'] += record['errG_cont'] * bs

                self.iteration += 1

        result['real_clf'] /= self.datasets_size['train']
        result['real_qual'] /= self.datasets_size['train']
        result['fake_clf'] /= self.datasets_size['train']
        result['fake_qual'] /= self.datasets_size['train']
        result['cont'] /= self.datasets_size['train']

        """
        Calculate FID score
        """
        real_mu = np.mean(np.concatenate(record['inception_real_pred']), axis=0)
        fake_mu = np.mean(np.concatenate(record['inception_fake_pred']), axis=0)

        real_sigma = np.cov(np.concatenate(record['inception_real_pred']), rowvar=False)
        fake_sigma = np.cov(np.concatenate(record['inception_fake_pred']), rowvar=False)

        result['FID'] = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

        """
        Calculate correlation coefficient
        """
        result['PLCC'], result['SRCC'], result['KRCC'] = \
            calculate_correlation_coefficient(
                torch.cat(record['gt_scores']).numpy(),
                torch.cat(record['pred_scores']).numpy()
            )

        return result

    def epoch_eval(self):
        record = {
            'gt_scores': [],
            'pred_scores': [],
        }

        result = {
            'real_clf': 0,
            'real_qual': 0,
        }

        self.netG.eval()
        self.netD.eval()

        for ref_imgs, dist_imgs, scores, categories, origin_scores in tqdm(self.dataloaders['val']):
            ref_imgs = ref_imgs.to(self.device)
            dist_imgs = dist_imgs.to(self.device)
            scores = scores.to(self.device).float()
            categories = categories.to(self.device)

            # Format batch
            bs = ref_imgs.size(0)

            with torch.no_grad():
                _, pred_categories, pred_scores = self.netD(ref_imgs, dist_imgs)

                record['errD_real_clf'] = self.ce_loss(pred_categories, categories).item()
                record['errD_real_qual'] = self.mse_loss(pred_scores, scores).item()

            # Record original scores and predict scores
            record['gt_scores'].append(origin_scores)
            record['pred_scores'].append(pred_scores.cpu().detach())

            """
            Record epoch loss
            """
            result['real_clf'] += record['errD_real_clf'] * bs
            result['real_qual'] += record['errD_real_qual'] * bs

        result['real_clf'] /= self.datasets_size['val']
        result['real_qual'] /= self.datasets_size['val']

        """
        Calculate correlation coefficient
        """
        result['PLCC'], result['SRCC'], result['KRCC'] = \
            calculate_correlation_coefficient(
                torch.cat(record['gt_scores']).numpy(),
                torch.cat(record['pred_scores']).numpy()
            )

        return result

    def write_epoch_log(self, results, epoch):
        write_epoch_log(self.writer, results, epoch)

    def save_weight(self, epoch):
        super(TrainerPhase1, self).save_weight(epoch)
        torch.save(self.netG.state_dict(), os.path.join(self.weight_dir, f'netG_epoch{epoch}.pth'))


class TrainerPhase2(Trainer):
    def __init__(self, cfg):
        super(TrainerPhase2, self).__init__(cfg)
        self.latent_dim = cfg.MODEL.LATENT_DIM

    def epoch_train(self):
        record = {
            'gt_scores': [],
            'pred_scores': []
        }

        result = {
            'real_loss': 0,
            'fake_loss': 0
        }

        self.netG.eval()
        self.netD.train()

        with tqdm(self.dataloaders['train']) as tepoch:
            for iteration, (ref_imgs, dist_imgs, scores, categories, origin_scores) in enumerate(tepoch):
                ref_imgs = ref_imgs.to(self.device)
                dist_imgs = dist_imgs.to(self.device)
                scores = scores.to(self.device).float()
                categories = categories.to(self.device)

                # Format batch
                bs = ref_imgs.size(0)

                self.optimizerD.zero_grad()

                """
                Deal with Real Distorted Images
                """
                _, _, pred_scores = self.netD(ref_imgs, dist_imgs)

                real_loss = self.mse_loss(pred_scores, scores)

                # Record original scores and predict scores
                record['gt_scores'].append(origin_scores)
                record['pred_scores'].append(pred_scores.cpu().detach())

                """
                Deal with Fake Distorted Images
                """
                # Generate batch of latent vectors
                noise = torch.randn(bs, self.latent_dim, device=self.device)

                fake_imgs = self.netD(ref_imgs,
                                      noise,
                                      scores.view(bs, -1),
                                      categories.view(bs, -1).float())

                _, _, pred_scores = self.netD(ref_imgs, fake_imgs.detach())

                fake_loss = self.mse_loss(pred_scores, scores)

                total_loss = real_loss + fake_loss
                total_loss.backward()
                self.optimizerD.step()

                result['real_loss'] += real_loss.item() * bs
                result['fake_loss'] += fake_loss.item() * bs

                # Show training message
                tepoch.set_postfix({
                    'Real Loss': real_loss.item(),
                    'Fake Loss': fake_loss.item(),
                    'Total Loss': total_loss.item()
                })

        result['real_loss'] /= self.datasets_size['train']
        result['fake_loss'] /= self.datasets_size['train']

        """
        Calculate correlation coefficient
        """
        result['PLCC'], result['SRCC'], result['KRCC'] = \
            calculate_correlation_coefficient(
                torch.cat(record['gt_scores']).numpy(),
                torch.cat(record['pred_scores']).numpy()
            )

        return result

    def epoch_eval(self):
        record = {
            'gt_scores': [],
            'pred_scores': []
        }

        result = {
            'real_loss': 0,
            'fake_loss': 0,
            'total_loss': 0
        }

        self.netG.eval()
        self.netD.eval()

        with tqdm(self.dataloaders['train']) as tepoch:
            for iteration, (ref_imgs, dist_imgs, scores, categories, origin_scores) in enumerate(tepoch):
                ref_imgs = ref_imgs.to(self.device)
                dist_imgs = dist_imgs.to(self.device)
                scores = scores.to(self.device).float()
                categories = categories.to(self.device)

                # Format batch
                bs, ncrops, c, h, w = ref_imgs.size()

                with torch.no_grad():
                    """
                    Evaluate real distorted images
                    """
                    _, _, pred_scores = self.netD(ref_imgs.view(-1, c, h, w), dist_imgs.view(-1, c, h, w))
                    pred_scores_avg = pred_scores.view(bs, ncrops, -1).mean(1).view(-1)

                    real_loss = self.mse_loss(pred_scores_avg, scores)

                    # Record original scores and predict scores
                    record['gt_scores'].append(origin_scores)
                    record['pred_scores'].append(pred_scores_avg.cpu().detach())

                    """
                    Evaluate fake distorted images
                    """
                    noise = torch.randn(bs, self.latent_dim, device=self.device)

                    fake_imgs = self.netD(
                        ref_imgs.view(-1, c, h, w),
                        noise.repeat_interleave(ncrops, dim=0),
                        scores.repeat_interleave(ncrops).view(bs * ncrops, -1),
                        categories.repeat_interleave(ncrops).view(bs * ncrops, -1).float()
                    )

                    _, _, pred_scores = self.netD(ref_imgs.view(-1, c, h, w), fake_imgs.detach())
                    pred_scores_avg = pred_scores.view(bs, ncrops, -1).mean(1).view(-1)

                    fake_loss = self.mse_loss(pred_scores_avg, scores)

                result['real_loss'] += real_loss.item() * bs
                result['fake_loss'] += fake_loss.item() * bs

                # Show training message
                tepoch.set_postfix({
                    'Real Loss': real_loss.item(),
                    'Fake Loss': fake_loss.item()
                })

        result['real_loss'] /= self.datasets_size['val']
        result['fake_loss'] /= self.datasets_size['val']

        """
        Calculate correlation coefficient
        """
        result['PLCC'], result['SRCC'], result['KRCC'] = \
            calculate_correlation_coefficient(
                torch.cat(record['gt_scores']).numpy(),
                torch.cat(record['pred_scores']).numpy()
            )

        return result

    def write_epoch_log(self, results, epoch):
        self.writer.add_scalars(
            'Loss', {
                'train_real': results['train']['real_loss'],
                'val_real': results['val']['real_loss'],
                'train_fake': results['train']['fake_loss'],
                'val_fake': results['val']['fake_loss']
            },
            epoch + 1
        )
        self.writer.add_scalars('PLCC', {x: results[x]['PLCC'] for x in ['train', 'val']}, epoch)
        self.writer.add_scalars('SRCC', {x: results[x]['SRCC'] for x in ['train', 'val']}, epoch)
        self.writer.add_scalars('KRCC', {x: results[x]['KRCC'] for x in ['train', 'val']}, epoch)
        self.writer.flush()
