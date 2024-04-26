from data.csuite.csuite_datasets import *
from evaluation.metrics import eval_all
from models.bnlearn_models import bnlearn_sachs, bnlearn_csuite
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
    print(eval_all(
        large_backdoor(1, False, True, "torch")[0],
        weak_arrows(1, False, True, "torch")[0])
    )

    return None


def main_bnlearn_csuite():
    available_csuite_datasets = [
        lingauss, linexp, nonlingauss, nonlin_simpson, symprod_simpson, large_backdoor, weak_arrows
    ]

    for csuite_dataset in available_csuite_datasets:
        bnlearn_csuite(csuite_dataset=csuite_dataset, estimate_cpds=True)
        print("\n")


def main():
    main_sanity_check()
    evaluation_demo()

    bnlearn_sachs()
    main_bnlearn_csuite()

    return None


if __name__ == '__main__':
    seed_everything()
    main()
