import torch
from causica.datasets.causica_dataset_format import VariablesMetadata, Variable
from causica.datasets.tensordict_utils import tensordict_shapes, tensordict_from_pandas
from causica.datasets.variable_types import VariableTypeEnum
from causica.distributions import GibbsDAGPrior, DistributionModule, AdjacencyDistribution, \
    ENCOAdjacencyDistributionModule, create_noise_modules, JointNoiseModule
from causica.functional_relationships import DECIEmbedFunctionalRelationships
from causica.graph.dag_constraint import calculate_dagness
from causica.sem.sem_distribution import SEMDistributionModule
from causica.training.auglag import AugLagLRConfig, AugLagLR, AugLagLossCalculator
from torch.utils.data import DataLoader

from evaluation.metrics import eval_all


def causica_deci(df, train_config, gt_graph=None):
    # Set training device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # Create dataloader
    data_tensordict = tensordict_from_pandas(df)
    data_tensordict = data_tensordict.apply(lambda t: t.to(dtype=torch.float32, device=device))
    dataloader = DataLoader(
        dataset=data_tensordict,
        collate_fn=lambda x: x,
        batch_size=train_config["batch_size"],
        shuffle=True,
        drop_last=False
    )

    # Create prior distribution over DAGs
    num_nodes = len(df.columns)
    prior = GibbsDAGPrior(num_nodes=num_nodes, sparsity_lambda=train_config["prior_sparsity_lambda"])

    # Create variational posterior distribution over adjacency matrices (to be optimized)
    adjacency_dist: DistributionModule[AdjacencyDistribution] = ENCOAdjacencyDistributionModule(num_nodes)

    # Create GNN to estimate functional relationships
    functional_relationships = DECIEmbedFunctionalRelationships(
        shapes=tensordict_shapes(data_tensordict),
        embedding_size=train_config["embedding_size"],
        out_dim_g=train_config["out_dim_g"],
        num_layers_g=train_config["num_layers_g"],
        num_layers_zeta=train_config["num_layers_zeta"]
    )

    # Create noise distributions for each node using the definitions in csuite's variables_metadata
    variables_shapes = tensordict_shapes(data_tensordict)
    variables_metadata = VariablesMetadata(
        [Variable(
            group_name=col,
            name=col + "_0",
            type=VariableTypeEnum.CONTINUOUS,
            lower=min(df[col]),
            upper=max(df[col]),
            always_observed=True
        ) for col in df.columns]
    )

    types_dict = {var.group_name: var.type for var in variables_metadata.variables}

    noise_submodules = create_noise_modules(variables_shapes, types_dict, train_config["noise_dist"])
    noise_module = JointNoiseModule(noise_submodules)

    # Create SEM module to combine var adj distr, func relationships, and noise distr for each node
    sem_module: SEMDistributionModule = SEMDistributionModule(adjacency_dist, functional_relationships, noise_module)
    sem_module.to(device)

    # Create optimizer. Define a separate lr for each module
    modules = {
        "functional_relationships": sem_module.functional_relationships,
        "vardist": sem_module.adjacency_module,
        "noise_dist": sem_module.noise_module,
    }

    auglag_config = AugLagLRConfig()
    parameter_list = [
        {"params": module.parameters(), "lr": auglag_config.lr_init_dict[name], "name": name}
        for name, module in modules.items()
    ]

    optimizer = torch.optim.Adam(parameter_list)

    # Augmented Lagrangian Scheduler for DECI to optimize towards DAG by slowly increasing rho and alpha
    scheduler = AugLagLR(config=auglag_config)
    auglag_loss = AugLagLossCalculator(
        init_alpha=train_config["init_alpha"],
        init_rho=train_config["init_rho"]
    )

    # Main training loop
    # Steps: sample graph from SEM, compute log probability of batch given graph, create ELBO to be optimized,
    # calculate DAG constraint, combine DAG constraint with ELBO to get loss
    num_samples = len(data_tensordict)
    for epoch in range(train_config["epochs"]):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            sem_distribution = sem_module()
            sem, *_ = sem_distribution.relaxed_sample(
                torch.Size([]),
                temperature=train_config["gumbel_temp"]
            )

            batch_log_prob = sem.log_prob(batch).mean()
            sem_distribution_entropy = sem_distribution.entropy()
            prior_term = prior.log_prob(sem.graph)
            objective = (-sem_distribution_entropy - prior_term) / num_samples - batch_log_prob
            constraint = calculate_dagness(sem.graph)

            loss = auglag_loss(objective, constraint / num_samples)

            loss.backward()
            optimizer.step()

            # Update Auglag parameters
            scheduler.step(
                optimizer=optimizer,
                loss=auglag_loss,
                loss_value=loss,
                lagrangian_penalty=constraint
            )

            loss.to(device=device)

            # Log metrics
            if epoch % 100 == 0 and i == 0:
                print(
                    f"epoch:{epoch} loss:{loss.item():.5g} nll:{-batch_log_prob.detach().cpu().numpy():.5g} "
                    f"dagness:{constraint.item():.5f} num_edges:{(sem.graph > 0.0).sum()} "
                    f"alpha:{auglag_loss.alpha:.5g} rho:{auglag_loss.rho:.5g} "
                    f"step:{scheduler.outer_opt_counter}|{scheduler.step_counter} "
                    f"num_lr_updates:{scheduler.num_lr_updates}"
                )

                if gt_graph is not None:
                    pred_graph = adjacency_dist().mode.cpu().numpy()
                    print(eval_all(torch.tensor(pred_graph), gt_graph))

    # Obtain result
    vardist = adjacency_dist()
    return vardist.mode.cpu().numpy()
