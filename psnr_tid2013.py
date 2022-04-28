import os
from math import log10, sqrt

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.tool.evaluate import calculate_correlation_coefficient


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


root_dir = '../data/TID2013'

df = pd.read_csv(os.path.join(root_dir, 'mos.csv'))

df['ref_img'] = df['ref_img'].apply(lambda x: os.path.join(root_dir, 'reference_images', f'{x}'))
df['dist_img'] = df['dist_img'].apply(lambda x: os.path.join(root_dir, 'distorted_images', f'{x}'))

record_pred_scores = []
record_gt_scores = np.array(df['mos'])

for i, data in tqdm(df.iterrows()):
    ref_img = np.array(Image.open(data['ref_img']).convert('RGB'), dtype='float')
    dist_img = np.array(Image.open(data['dist_img']).convert('RGB'), dtype='float')
    record_pred_scores.append(PSNR(ref_img, dist_img))

record_pred_scores = np.array(record_pred_scores)

plcc, srcc, krcc = calculate_correlation_coefficient(
    record_gt_scores,
    record_pred_scores
)

print(f'TID2013 Dataset')
print(f'PLCC: {plcc}')
print(f'SRCC: {srcc}')
print(f'KRCC: {krcc}')
