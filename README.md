# Thesis Project: Research on Full-reference Image Quality Assessment by Using Transformer and DISTS with GAN-based Augmentation

## Table of Contents
1. [Hardware](#Hardware)
2. [Installation](#Installation)
3. [Dataset Preparation](#Dataset-Preparation)
4. [Prepare Data and Code](#Prepare-Data-and-Code)
5. [Training](#Training)
6. [Evaluation](#Evaluation)

## Hardware

The following specs were used to create the original solution.

- Ubuntu 20.04.3 LTS
- Intel(R) Core(TM) i9-10900 CPU @ 2.80GHz
- NVIDIA GeForce RTX 3090

## Installation

1. Anaconda is required. Modify **environment.yml** to satisfy your machine environment.
   ```yaml
   name: iqa  # change the environment name if you want
   channels:
   ...
   ...
   prefix: /home/rhome/nelson/.conda/envs/iqa  # Must change the prefix to the directory where you want the conda environment to be set.
   ```

2. Create a conda environment with **environment.yml**.
   {envs_name} is the environment name which you should assign.
   The default of the environment name is **iqa**.

   ```shell
   conda env create -f environment.yml
   conda activate <envs_name>
   ```

## Dataset Preparation

Download the dataset from the following website.

* [PIPAL Public Training Set](https://www.jasongt.com/projectpages/pipal.html)

* [LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm)

* [TID2013](https://www.ponomarenko.info/tid2013.htm)

## Prepare Data and Code

After downloading and extracting, the data directory is structured as:

```text
+- data
    +- PIPAL
    +- LIVE
    +- TID2013
+- code
    +- scores_record
    +- src
    ATDIQA.py
    Augmented ATDIQA.py
    environment.yml
    eval.py
    pred.py
    README.md
    train.py
```

Then, get into the 'code' directory.

```shell
cd code
```

## Training

If you need the help of train.py, you can use the following instruction.

```shell
python train.py --help
```

Train with configuration file:

```shell
python train.py --config <config_path>
```

### DISTS-based and IQT-based Methods

There are several default configuration files in src/config/experiments.

| Model      | Configuration files    |
|------------|------------------------|
| DISTS-Tune | DISTS-Tune_config.yaml |
| IQT        | IQT_config.yaml        |
| IQT-C      | IQT-C_config.yaml      |
| IQT-L      | IQT-L_config.yaml      |
| IQT-M      | IQT-M_config.yaml      |
| IQT-H      | IQT-H_config.yaml      |
| IQT-Mixed  | IQT-Mixed_config.yaml  |

These models are mentioned in my thesis.

#### Example

Take IQT-L for example:

```shell
python train.py --config src/config/experiments/IQT-L_config.yaml
```

### Augmented FR-IQA

There are **two** Augmented FR-IQA mentioned in my thesis, Augmented DISTS-Tune and Augmented IQT-Mixed, respectively.
In addition, these methods require **three** phases training pipeline.

You should train phase 1, before training phase 2. 
Likewise, you should train phase 2, before training phase3.

#### Example

Take Augmented DISTS-Tune for example:

1. Train phase 1

   ```shell
   python train.py --config src/config/experiments/Aug_DISTS-Tune_phase1_config.yaml
   ```
   
2. Train phase 2
   ```shell
   python train.py --config src/config/experiments/Aug_DISTS-Tune_phase2_config.yaml
   ```

3. Train phase 3
   ```shell
   python train.py --config src/config/experiments/Aug_DISTS-Tune_phase3_config.yaml
   ```

## Evaluation

If you need the help of eval.py, you can use the following instruction.

```shell
python eval.py --help
```

Evaluate with configuration file and the path of model:

```shell
python eval.py --config <config_path> --netD_path <netD_path> --dataset <dataset_name>
```

* <config_path> is the path to the configuration file of model.
* <netD_path> is the path of the weights of FR-IQA.
* <dataset_name> can be chose from 'PIPAL', 'LIVE' and 'TID2013'.

### Example

Take evaluating IQT-L on LIVE for example.
Assume that the weights of IQT-L are saved at **experiments/IQT-L/models/netD_epoch200.pth**.

```shell
python eval.py --config src/config/experiments/IQT-L_config.yaml --netD_path experiments/IQT-L/models/netD_epoch200.pth --dataset LIVE
```
