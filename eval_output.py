import argparse
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from src.config.config import get_cfg_defaults
from src.data.dataset import create_dataloaders
from src.modeling.module import MultiTask


def main(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, datasets_size = create_dataloaders(
        Path(cfg.DATASETS.ROOT_DIR),
        batch_size=cfg.DATASETS.BATCH_SIZE,
        num_workers=cfg.DATASETS.NUM_WORKERS
    )

    netD = MultiTask(cfg).to(device)
    netD.load_state_dict(torch.load(args.netD_path))

    records = {}
    for mode in ['val', 'test']:
        record = {
            'gt_scores': [],
            'pred_scores': [],
        }

        netD.eval()
        with tqdm(dataloaders[mode]) as tepoch:
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

        records[mode] = record

    with open(args.output_file + '.pickle', 'wb') as handle:
        pickle.dump(records, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Configuration YAML file for evaluating')
    parser.add_argument('--netD_path', required=True, type=str, help='Load model path')
    parser.add_argument('--output_file', default='output', type=str, help='Result file name')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    try:
        cfg.merge_from_file(args.config)
    except:
        print('Using default configuration file')

    assert cfg.MODEL.BACKBONE.NAME in ['InceptionResNetV2']
    assert cfg.MODEL.BACKBONE.FEAT_LEVEL in ['low', 'medium', 'high', 'mixed']
    assert cfg.MODEL.EVALUATOR in ['IQT']

    if cfg.MODEL.BACKBONE.FEAT_LEVEL == 'low':
        cfg.MODEL.BACKBONE.CHANNELS = tuple(320 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(21 * 21 for _ in range(6))
    elif cfg.MODEL.BACKBONE.FEAT_LEVEL == 'medium':
        cfg.MODEL.BACKBONE.CHANNELS = tuple(1088 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(10 * 10 for _ in range(6))
    elif cfg.MODEL.BACKBONE.FEAT_LEVEL == 'high':
        cfg.MODEL.BACKBONE.CHANNELS = tuple(2080 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(4 * 4 for _ in range(6))
    else:  # mixed
        cfg.MODEL.BACKBONE.CHANNELS = tuple(320 for _ in range(6)) + tuple(1088 for _ in range(6)) + tuple(
            2080 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(21 * 21 for _ in range(6)) + tuple(10 * 10 for _ in range(6)) + tuple(
            4 * 4 for _ in range(6))

    cfg.freeze()

    main(args, cfg)
