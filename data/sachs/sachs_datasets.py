import numpy as np
import os
import pandas as pd
import torch


def unaltered_dataset(
        get_data=True,
        return_index_name_correlation=False,
        return_adjacency_graph=False,
        adjacency_graph_format="torch",
):
    to_return = tuple()

    if get_data:
        cwd = os.path.abspath(os.getcwd())
        root_path = cwd[:cwd.find("Industry_Project")] + "Industry_Project"
        to_return += (pd.read_excel(os.path.join(root_path, "data/sachs/original_data/1.cd3cd28.xls")),)

    if return_index_name_correlation:
        to_return += ({
            "praf":     0,
            "pmek":     1,
            "plcg":     2,
            "PIP2":     3,
            "PIP3":     4,
            "p44/42":   5,
            "pakts473": 6,
            "PKA":      7,
            "PKC":      8,
            "P38":      9,
            "pjnk":     10
        },)

    if return_adjacency_graph:
        # TODO: Check if dotted connections should be included
        adjacency_graph = [
            # praf  pmek  plcg  PIP2  PIP3  p44/42  pakts473  PKA  PKC  P38  pjnk
            [    0,    1,    0,    0,    0,      0,        0,   0,   0,   0,    0],  # praf
            [    0,    0,    0,    0,    0,      1,        0,   0,   0,   0,    0],  # pmek
            [    0,    0,    0,    1,    1,      0,        0,   0,   0,   0,    0],  # plcg
            [    0,    0,    0,    0,    0,      0,        0,   0,   0,   0,    0],  # PIP2
            [    0,    0,    0,    1,    0,      0,        0,   0,   0,   0,    0],  # PIP3
            [    0,    0,    0,    0,    0,      0,        1,   0,   0,   0,    0],  # p44/42
            [    0,    0,    0,    0,    0,      0,        0,   0,   0,   0,    0],  # pakts473
            [    1,    1,    0,    0,    0,      1,        1,   0,   0,   0,    1],  # PKA
            [    1,    1,    0,    0,    0,      0,        0,   1,   0,   1,    1],  # PKC
            [    0,    0,    0,    0,    0,      0,        0,   0,   0,   0,    0],  # P38
            [    0,    0,    0,    0,    0,      0,        0,   0,   0,   0,    0],  # pjnk
        ]

        if adjacency_graph_format == "list":
            to_return += (adjacency_graph,)
        else:
            adjacency_graph = np.array(adjacency_graph)
            if adjacency_graph_format == "numpy":
                to_return += (adjacency_graph,)
            else:
                to_return += (torch.tensor(adjacency_graph),)

    return to_return
