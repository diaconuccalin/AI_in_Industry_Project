import torch

from data.csuite.csuite_datasets import *
from evaluation.metrics import eval_all
from utils.params import *


def seed_everything():
    torch.manual_seed(SEED)

    return SEED


def main_sanity_check():
    available_csuite_datasets = [
        lingauss, linexp, nonlingauss, nonlin_simpson, symprod_simpson, large_backdoor, weak_arrows
    ]

    for csuite_dataset in available_csuite_datasets:
        sanity_check(csuite_dataset)

    return None


def evaluation_demo():
    gt = torch.tensor(np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0]
        ]
    ))

    pred = torch.tensor(np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ]
    ))

    print(eval_all(pred, gt))

    return None


def main():
    main_sanity_check()
    evaluation_demo()

    return None


if __name__ == '__main__':
    seed_everything()
    main()
