
import sys, os
import pandas as pd

import pandas as pd
import numpy as np
from itertools import permutations
from datetime import datetime
from tqdm import tqdm

from constants import DATA_DIR
from utils import filter_by_date, plot_journeys


df = pd.read_csv(os.path.join(DATA_DIR, "output.csv"))
filtered = filter_by_date(df, "2024-01-01", column="leave_at")
print(filtered.head(1000))

plot_journeys(df=filtered)



df = filtered

# Parse datetimes
df["leave_at"] = pd.to_datetime(df["leave_at"], format="%Y-%m-%d %H:%M:%S")
df["arrive_at"] = pd.to_datetime(df["arrive_at"], format="%Y-%m-%d %H:%M:%S")

# Create transition cost matrix (waiting minutes if feasible, else inf)
n = len(df)
transition_minutes = np.full((n, n), np.inf)
for i in tqdm(range(n)):
    for j in range(n):
        if i == j:
            transition_minutes[i, j] = 0.0
        else:
            arrive_i = df.loc[i, "arrive_at"]
            leave_j = df.loc[j, "leave_at"]
            gap = (leave_j - arrive_i).total_seconds() / 60.0
            if gap >= 0:
                transition_minutes[i, j] = gap
            else:
                transition_minutes[i, j] = np.inf  # overlapping → infeasible

# Build combined cost matrix = waiting minutes + cost of journey j
journey_costs = df["cost"].astype(float).to_numpy()
combined_cost = np.full((n, n), np.inf)
for i in range(n):
    for j in range(n):
        if np.isfinite(transition_minutes[i, j]):
            combined_cost[i, j] = transition_minutes[i, j] + journey_costs[j]

transition_df = pd.DataFrame(transition_minutes, index=df["id"], columns=df["id"])
combined_df = pd.DataFrame(combined_cost, index=df["id"], columns=df["id"])

print("=== Transition minutes (arrive_i -> leave_j) ===")
print(transition_df, "\n")

print("=== Combined cost (waiting minutes + journey cost of j) ===")
print(combined_df, "\n")

# Brute-force best ordering (small n)
best_sequence = None
best_total_cost = np.inf

for perm in tqdm(permutations(range(n))):
    feasible = True
    total_cost = journey_costs[perm[0]]
    for idx in range(1, n):
        prev = perm[idx-1]
        cur = perm[idx]
        trans = transition_minutes[prev, cur]
        if not np.isfinite(trans):
            feasible = False
            break
        total_cost += trans + journey_costs[cur]
    if feasible and total_cost < best_total_cost:
        best_total_cost = total_cost
        best_sequence = perm

print("=== Journey DataFrame ===")
print(df[["id","child_name","leave_at","arrive_at","cost"]], "\n")

if best_sequence is None:
    print("No feasible sequence: trips overlap so they cannot be done consecutively by one vehicle.")
else:
    ids = df.loc[list(best_sequence), "id"].tolist()
    names = df.loc[list(best_sequence), "child_name"].tolist()
    print("Best sequence of journeys:")
    print("  IDs:", ids)
    print("  Children:", names)
    print(f"  Total cost = {best_total_cost:.2f}")
