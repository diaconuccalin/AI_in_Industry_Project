from causica.distributions import ContinuousNoiseDist

from data.csuite.csuite_datasets import *
from data.sachs.sachs_datasets import unaltered_dataset
from evaluation.metrics import eval_all
from models.bnlearn_models import bnlearn_sachs, bnlearn_csuite
from models.causica_deci import causica_deci
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
        print(csuite_dataset.__name__)
        bnlearn_csuite(csuite_dataset=csuite_dataset, estimate_cpds=True)
        print("\n")


def main_deci():
    available_csuite_datasets = [
        lingauss, linexp, nonlingauss, nonlin_simpson, symprod_simpson, large_backdoor, weak_arrows
    ]

    train_config = {
        "batch_size": 128,
        "epochs": 2000,
        "init_alpha": 0.0,
        "init_rho": 1.0,
        "gumbel_temp": 0.25,
        "prior_sparsity_lambda": 5.0,
        "embedding_size": 32,
        "out_dim_g": 32,
        "num_layers_g": 2,
        "num_layers_zeta": 2,
        "noise_dist": ContinuousNoiseDist.SPLINE
    }

    for csuite_dataset in available_csuite_datasets:
        df, gt_graph = csuite_dataset(2000, True, True)
        pred_graph = causica_deci(df, train_config)

        print(csuite_dataset.__name__)
        print(eval_all(torch.tensor(pred_graph), gt_graph))
        print()

    df, gt_graph = unaltered_dataset(get_data=True, return_index_name_correlation=False, return_adjacency_graph=True)
    pred_graph = causica_deci(df, train_config)
    print("Sachs dataset:")
    print(eval_all(torch.tensor(pred_graph), gt_graph))


def main():
    main_sanity_check()
    evaluation_demo()

    main_bnlearn_csuite()
    print("Sachs dataset:")
    bnlearn_sachs()

    main_deci()

    return None


if __name__ == '__main__':
    seed_everything()
    main()
