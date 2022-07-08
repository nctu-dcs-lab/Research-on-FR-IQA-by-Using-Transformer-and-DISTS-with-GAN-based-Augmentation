import argparse

import torch
from torch.utils.data import DataLoader

from src.config.config import get_cfg_defaults
from src.data.dataset import create_dataloaders, LIVE, TID2013
from src.modeling.module import MultiTask
from src.tool.evaluate import evaluate


def main(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netD = MultiTask(cfg).to(device)
    netD.load_state_dict(torch.load(args.netD_path))

    if args.dataset == 'PIPAL':
        dataloaders, datasets_size = create_dataloaders(cfg, phase='eval')

        results = {}
        for mode in ['train', 'val', 'test']:
            results[mode] = evaluate(dataloaders[mode], netD, device)
            print(f'{mode}')
            print(f'PLCC: {results[mode]["PLCC"]}')
            print(f'SRCC: {results[mode]["SRCC"]}')
            print(f'KRCC: {results[mode]["KRCC"]}')

    else:
        if args.dataset == 'LIVE':
            dataset = LIVE(root_dir='../data/LIVE', img_size=cfg.DATASETS.IMG_SIZE)

        else:
            dataset = TID2013(root_dir='../data/TID2013', img_size=cfg.DATASETS.IMG_SIZE)

        dataloader = DataLoader(dataset,
                                batch_size=cfg.DATASETS.BATCH_SIZE,
                                shuffle=False,
                                num_workers=cfg.DATASETS.NUM_WORKERS)

        result = evaluate(dataloader, netD, device)
        print(f'PLCC: {result["PLCC"]}')
        print(f'SRCC: {result["SRCC"]}')
        print(f'KRCC: {result["KRCC"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Configuration YAML file for evaluating')
    parser.add_argument('--netD_path', required=True, type=str, help='Load model path')
    parser.add_argument('--dataset',
                        default='PIPAL',
                        choices=['PIPAL', 'LIVE', 'TID2013'],
                        help='Dataset to be evaluated')
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
