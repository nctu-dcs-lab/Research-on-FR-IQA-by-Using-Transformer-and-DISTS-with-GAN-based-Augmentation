import argparse
from pathlib import Path

import torch

from src.data.dataset import create_dataloaders
from src.modeling.evaluate import evaluate
from src.modeling.module import MultiTask


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, datasets_size = create_dataloaders(
        Path(args.data_dir),
        phase=args.phase,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    model = {
        'netD': MultiTask(pretrained=True).to(device)
    }
    model['netD'].load_state_dict(torch.load(args.netD_path))

    results = {}
    for mode in ['val', 'test']:
        results[mode] = evaluate(dataloaders[mode], model, device)
        print(f'{mode}')
        print(f'PLCC: {results[mode]["PLCC"]}')
        print(f'SRCC: {results[mode]["SRCC"]}')
        print(f'KRCC: {results[mode]["KRCC"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        default='../data/PIPAL(processed)',
                        type=str,
                        help='Root directory for PIPAL dataset')
    parser.add_argument('--netD_path', required=True, type=str, help='Load model path')
    parser.add_argument('--phase', default='phase1', type=str, choices=['phase1', 'phase2'])
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=10, type=int)

    args = parser.parse_args()

    main(args)
