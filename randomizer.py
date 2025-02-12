"""
This file takes in a .parquet file and output a new .paquet file with X% randomly selected rows from the input.
"""

import pandas as pd

percentage = 0.8

# Read and retrieve edge_index
df = pd.read_parquet("./raw/perfect.parquet", engine="auto")
df = df.reset_index(drop=True)

# Randomly sample specified percentage of the data
df_sampled = df.sample(frac=percentage, random_state=42)

# Save the sampled DataFrame to a new parquet file
df_sampled.to_parquet("./raw/rand80.parquet")