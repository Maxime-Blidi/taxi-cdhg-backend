import pandas as pd
import numpy as np
from datetime import datetime
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from utils import filter_by_date
import os, sys
from constants import DATA_DIR
from tqdm import tqdm

# === Load data ===
df = pd.read_csv(os.path.join(DATA_DIR, "output.csv"))
df = filter_by_date(df, "2024-01-01", column="leave_at")
df["leave_at"] = pd.to_datetime(df["leave_at"])
df["arrive_at"] = pd.to_datetime(df["arrive_at"])
n = len(df)

# === Settings ===
num_vehicles = 3
depot = 0  # index for depot (garage)

# === Build nodes ===
# Node 0 = depot
# For each journey i:
#   pickup = 2*i+1
#   dropoff = 2*i+2
num_nodes = 1 + 2 * n

# === Cost matrix ===
# Here we fake travel cost as waiting time difference between times.
# Replace with a real distance/time matrix if available.
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
cost_matrix = np.full((n, n), np.inf)
for i in range(n):
    for j in range(n):
        if np.isfinite(transition_minutes[i, j]):
            cost_matrix[i, j] = transition_minutes[i, j] + journey_costs[j]

print(cost_matrix)
# === Time windows ===
time_windows = [(0, 24 * 60)]  # depot can operate all day
for i in range(n):
    leave = int(df.loc[i, "leave_at"].timestamp() / 60)
    arrive = int(df.loc[i, "arrive_at"].timestamp() / 60)
    time_windows.append((leave, 24 * 60))   # pickup: after leave_at
    time_windows.append((0, arrive))        # dropoff: before arrive_at

# === Capacities ===
# Each pickup adds +1 passenger, each dropoff -1
demands = [0]  # depot
for _ in range(n):
    demands.append(1)   # pickup
    demands.append(-1)  # dropoff
vehicle_capacity = 4  # max passengers per taxi

# === OR-Tools model ===
manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
routing = pywrapcp.RoutingModel(manager)

def cost_callback(from_index, to_index):
    return int(cost_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

transit_callback_index = routing.RegisterTransitCallback(cost_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Time dimension
routing.AddDimension(
    transit_callback_index,
    30,       # waiting slack
    24 * 60,  # horizon (1 day, in minutes)
    False,
    "Time",
)
time_dimension = routing.GetDimensionOrDie("Time")

for node, (start, end) in enumerate(time_windows):
    index = manager.NodeToIndex(node)
    time_dimension.CumulVar(index).SetRange(start, end)

# Capacity dimension
def demand_callback(from_index):
    return demands[manager.IndexToNode(from_index)]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,
    [vehicle_capacity] * num_vehicles,
    True,
    "Capacity",
)

# Pickup & delivery constraints
for i in range(n):
    pickup = manager.NodeToIndex(2 * i + 1)
    dropoff = manager.NodeToIndex(2 * i + 2)
    routing.AddPickupAndDelivery(pickup, dropoff)
    routing.solver().Add(routing.VehicleVar(pickup) == routing.VehicleVar(dropoff))
    routing.solver().Add(time_dimension.CumulVar(pickup) <= time_dimension.CumulVar(dropoff))

# === Search ===
search_params = pywrapcp.DefaultRoutingSearchParameters()
search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
search_params.time_limit.seconds = 10

solution = routing.SolveWithParameters(search_params)

# === Output ===
if solution:
    for v in range(num_vehicles):
        index = routing.Start(v)
        plan = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            tmin = solution.Min(time_var)
            plan.append((node, tmin))
            index = solution.Value(routing.NextVar(index))
        print(f"Taxi {v}: {plan}")
else:
    print("No solution found.")
