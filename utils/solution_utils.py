import numpy as np
import pandas as pd
import sklearn.feature_selection


# Adapted from https://www.bnlearn.com/research/sachs05/
# and https://github.com/vspinu/bnlearn/blob/fec3cc0371fabfb6fe7abe0574038522beb2aa13/R/discretize.R
# and https://dspace.mit.edu/handle/1721.1/8699?show=full
def hartemink_discretization(df, breaks=3, initial_breaks=60):
    # Cache useful quantities
    nodes = df.columns

    # Perform initial discretization
    quantile_df = df.copy(deep=True)
    for col in nodes:
        quantile_df[col] = pd.qcut(list(df[col]), initial_breaks, labels=False)

    # Count down through discretization levels
    for n_levels in range(initial_breaks, breaks, -1):
        if n_levels % 10 == 0:
            print(f"Working on the {n_levels}th level of discretization.")
        # Go through each variable
        for node in nodes:
            # Prepare df by isolating current node
            x = quantile_df.loc[:, quantile_df.columns != node]
            y = quantile_df[node]
            mutual_information_values = list()

            # Go through level pairs
            for collapsing in range(1, n_levels):
                local_y = np.copy(y)
                local_y[local_y >= collapsing] -= 1

                # Compute mutual information
                mutual_information_values.append(sum(sklearn.feature_selection.mutual_info_regression(x, local_y)))

            # Collapse according to best result
            quantile_df[node][y >= (np.argmax(mutual_information_values) + 1)] -= 1

    return quantile_df.astype("float64")
