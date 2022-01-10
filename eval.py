import argparse
from pathlib import Path

import torch

from src.config.config import get_cfg_defaults
from src.data.dataset import create_dataloaders
from src.modeling.module import MultiTask
from src.tool.evaluate import evaluate


def main(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, datasets_size = create_dataloaders(
        Path(cfg.DATASETS.ROOT_DIR),
        batch_size=cfg.DATASETS.BATCH_SIZE,
        num_workers=cfg.DATASETS.NUM_WORKERS
    )

    netG = MultiTask(cfg).to(device)
    netG.load_state_dict(torch.load(args.netD_path))

    results = {}
    for mode in ['val', 'test']:
        results[mode] = evaluate(dataloaders[mode], netG, device)
        print(f'{mode}')
        print(f'PLCC: {results[mode]["PLCC"]}')
        print(f'SRCC: {results[mode]["SRCC"]}')
        print(f'KRCC: {results[mode]["KRCC"]}')


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

    cfg.freeze()

    main(args, cfg)
