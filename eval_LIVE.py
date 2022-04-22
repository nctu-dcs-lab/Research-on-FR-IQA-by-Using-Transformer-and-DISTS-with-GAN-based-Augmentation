import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.config import get_cfg_defaults
from src.data.dataset import LIVE
from src.modeling.module import MultiTask
from src.tool.evaluate import calculate_correlation_coefficient


def main(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = LIVE(root_dir='../data/LIVE', img_size=cfg.DATASETS.IMG_SIZE)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.DATASETS.BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.DATASETS.NUM_WORKERS)

    netD = MultiTask(cfg).to(device)
    netD.load_state_dict(torch.load(args.netD_path))

    record = {
        'gt_scores': [],
        'pred_scores': [],
    }
    result = {}

    netD.eval()
    with tqdm(dataloader) as tepoch:
        for iteration, (ref_imgs, dist_imgs, dmos) in enumerate(tepoch):
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
                record['gt_scores'].append(dmos)
                record['pred_scores'].append(pred_scores_avg.cpu().detach())

    """
    Calculate correlation coefficient
    """
    result['PLCC'], result['SRCC'], result['KRCC'] = \
        calculate_correlation_coefficient(
            torch.cat(record['gt_scores']).numpy(),
            torch.cat(record['pred_scores']).numpy()
        )

    print(f'LIVE Dataset')
    print(f'PLCC: {result["PLCC"]}')
    print(f'SRCC: {result["SRCC"]}')
    print(f'KRCC: {result["KRCC"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Configuration YAML file for evaluating')
    parser.add_argument('--netD_path', required=True, type=str, help='Load model path')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    try:
        cfg.merge_from_file(args.config)
    except:
        print('Using default configuration file')

    assert cfg.MODEL.BACKBONE.NAME in ['VGG16', 'InceptionResNetV2']
    assert cfg.MODEL.BACKBONE.FEAT_LEVEL in ['low', 'medium', 'high', 'mixed', 'reduced mixed']
    assert cfg.MODEL.EVALUATOR in ['IQT', 'DISTS', 'Transformer']

    cfg.freeze()

    main(args, cfg)
