import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, kendalltau, pearsonr
from tqdm import tqdm

from dataset import create_dataloaders
from module import MultiTask

warnings.simplefilter('ignore', np.RankWarning)


def calculate_correlation_coefficient(gt_qual, pred_qual):
    z = np.polyfit(pred_qual, gt_qual, 3)
    p = np.poly1d(z)

    return pearsonr(gt_qual, p(pred_qual))[0], \
           np.abs(spearmanr(gt_qual, pred_qual)[0]), \
           np.abs(kendalltau(gt_qual, pred_qual)[0])


def evaluate(dataloader, model, mse_loss, dataset_size, device=torch.device('cpu')):
    gt_qual = []
    pred_qual = []
    epoch_loss = 0

    model.eval()

    with tqdm(dataloader) as tepoch:
        for ref_imgs, dist_imgs, scores, _, origin_scores in tepoch:
            ref_imgs = ref_imgs.to(device)
            dist_imgs = dist_imgs.to(device)
            scores = scores.to(device).float()

            # Format batch
            bs, ncrops, c, h, w = ref_imgs.size()

            with torch.no_grad():
                pred_scores = model(ref_imgs.view(-1, c, h, w), dist_imgs.view(-1, c, h, w))
                pred_scores_avg = pred_scores.view(bs, ncrops, -1).mean(1).view(-1)
                loss = mse_loss(pred_scores_avg, scores)

                # Record original scores and predict scores
                gt_qual.append(origin_scores)
                pred_qual.append(pred_scores_avg.cpu().detach())

                epoch_loss += loss.item() * bs

                # Show evaluate message
                tepoch.set_postfix({'Loss': loss.item()})

    gt_qual = torch.cat(gt_qual).numpy()
    pred_qual = torch.cat(pred_qual).numpy()

    plcc, srcc, krcc = calculate_correlation_coefficient(gt_qual, pred_qual)

    return epoch_loss / dataset_size, plcc, srcc, krcc


def main(data_dir, netD_path, batch_size=16, num_workers=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, datasets_size = create_dataloaders(data_dir, batch_size=batch_size, num_workers=num_workers)

    # Set Up Model
    netD = MultiTask(pretrained=True)
    netD.load_state_dict(torch.load(netD_path))
    netD = netD.to(device)
    netD.eval()

    mse_loss = nn.MSELoss()

    epoch_loss, plcc, srcc, krcc = evaluate(dataloaders['test'], model, mse_loss, datasets_size['test'], device)

    print(f'Mean Square Error: {epoch_loss}')
    print(f'PLCC: {plcc}')
    print(f'SRCC: {srcc}')
    print(f'KRCC: {krcc}')


if __name__ == '__main__':
    main(
        Path('../data/PIPAL(processed)/'),
        netD_path=os.path.expanduser(f'~/nfs/result/iqt/experiment{5}/models/model_epoch{95}.pth')
    )
