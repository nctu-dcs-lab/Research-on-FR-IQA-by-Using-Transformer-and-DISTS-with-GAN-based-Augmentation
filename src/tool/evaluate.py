import warnings

import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau, pearsonr
from tqdm import tqdm

warnings.simplefilter('ignore', np.RankWarning)


def calculate_correlation_coefficient(gt_qual, pred_qual):
    z = np.polyfit(pred_qual, gt_qual, 3)
    p = np.poly1d(z)

    return pearsonr(gt_qual, p(pred_qual))[0], \
           np.abs(spearmanr(gt_qual, pred_qual)[0]), \
           np.abs(kendalltau(gt_qual, pred_qual)[0])


def evaluate(dataloader, netD, device=torch.device('cpu')):
    record = {
        'gt_scores': [],
        'pred_scores': [],
    }
    result = {}

    netD.eval()
    with tqdm(dataloader) as tepoch:
        for iteration, (ref_imgs, dist_imgs, _, _, origin_scores) in enumerate(tepoch):
            ref_imgs = ref_imgs.to(device)
            dist_imgs = dist_imgs.to(device)

            # Format batch
            bs, ncrops, c, h, w = ref_imgs.size()

            with torch.no_grad():
                """
                Evaluate distorted images
                """
                _, _, pred_scores = netD(ref_imgs.view(-1, c, h, w), dist_imgs.view(-1, c, h, w))
                pred_scores_avg = pred_scores.view(bs, ncrops, -1).mean(1).view(-1)

                # Record original scores and predict scores
                record['gt_scores'].append(origin_scores)
                record['pred_scores'].append(pred_scores_avg.cpu().detach())

    """
    Calculate correlation coefficient
    """
    result['PLCC'], result['SRCC'], result['KRCC'] = \
        calculate_correlation_coefficient(
            torch.cat(record['gt_scores']).numpy(),
            torch.cat(record['pred_scores']).numpy()
        )
    return result
