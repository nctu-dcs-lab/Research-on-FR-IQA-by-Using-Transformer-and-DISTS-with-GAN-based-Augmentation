import argparse
import pickle

import torch

from src.config.config import get_cfg_defaults
from src.data.dataset import create_dataloaders
from src.modeling.module import MultiTask
from src.tool.evaluate import evaluate


def main(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, datasets_size = create_dataloaders(cfg)

    netG = MultiTask(cfg).to(device)
    netG.load_state_dict(torch.load(args.netD_path))

    results = {}
    for mode in ['val', 'test']:
        results[mode] = evaluate(dataloaders[mode], netG, device)
        print(f'{mode}')
        print(f'PLCC: {results[mode]["PLCC"]}')
        print(f'SRCC: {results[mode]["SRCC"]}')
        print(f'KRCC: {results[mode]["KRCC"]}')

    with open(args.output_file + '.pickle', 'wb') as handle:
        pickle.dump(results, handle)


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

    assert cfg.MODEL.BACKBONE.NAME in ['VGG16', 'InceptionResNetV2']
    assert cfg.MODEL.BACKBONE.FEAT_LEVEL in ['low', 'medium', 'high', 'mixed', 'reduced mixed']
    assert cfg.MODEL.EVALUATOR in ['IQT', 'DISTS', 'Transformer']

    cfg.freeze()

    main(args, cfg)
