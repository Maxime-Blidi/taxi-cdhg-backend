
import pandas as pd
import os, sys
import time, requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import numpy as np
from math import radians, sin, cos, asin, sqrt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import threading
from tqdm import tqdm
import json

from constants import DATA_DIR, OPENROUTESERVICE_API_KEY, CONFIG
from utils import filter_by_date

def _guess_column(cands, cols):
    cols_lower = {c.lower(): c for c in cols}
    for pat in cands:
        for k,v in cols_lower.items():
            if pat in k:
                return v
    return None

drivers_path = os.path.join(DATA_DIR, "Chauffeurs-ASE.xlsx")
trajets_path = os.path.join(DATA_DIR, "Trajets Enfants.xlsx")

drivers = pd.read_excel(drivers_path) if drivers_path.lower().endswith(("xlsx","xls")) else pd.read_csv(drivers_path)
trajets  = pd.read_excel(trajets_path) if trajets_path.lower().endswith(("xlsx","xls")) else pd.read_csv(trajets_path)
trajets = trajets[trajets['date'] == CONFIG["date"]].copy()

# Drivers
garage_col = _guess_column(["garage","lieu","base","depot"], drivers.columns) or "Lieu garage Véhicule"
cap_col    = _guess_column(["capacite","places","passagers","seat"], drivers.columns) or "capacite passagers Véhicule"
cost_col   = _guess_column(["cout","kilomet","€/km","euro","euro/km"], drivers.columns) or "Cout kilométrique (€)"

# Trajets
home_col   = _guess_column(["domicile","adresse","home","residen","foyer"], trajets.columns) or "lieu_depart" # Corrected default
school_col = _guess_column(["ecole","école","etablissement","school","etabl","arrivee"], trajets.columns) or "lieu_arrivee" # Corrected default and added 'arrivee'
dist_col   = _guess_column(["distance","km"], trajets.columns)  # optional
dur_col    = _guess_column(["durée","duree","temps","minutes"], trajets.columns)  # optional
bell_col   = _guess_column(["cloche","bell","horaire","heure","debut","entrée","entree","depart"], trajets.columns)  # optional, added 'depart'

print("\nDetected mapping:")
print({"garage":garage_col, "capacity":cap_col, "€/km":cost_col, "home":home_col, "school":school_col,
    "dist?":dist_col, "dur?":dur_col, "bell?":bell_col})

# Normalize types
drivers[garage_col] = drivers[garage_col].astype(str)
trajets[home_col]   = trajets[home_col].astype(str)
trajets[school_col] = trajets[school_col].astype(str)


def map_data(drivers: pd.DataFrame, trajets: pd.DataFrame):
        
    # Drivers
    garage_col = _guess_column(["garage","lieu","base","depot"], drivers.columns) or "Lieu garage Véhicule"

    # Trajets
    home_col   = _guess_column(["domicile","adresse","home","residen","foyer"], trajets.columns) or "lieu_depart" # Corrected default
    school_col = _guess_column(["ecole","école","etablissement","school","etabl","arrivee"], trajets.columns) or "lieu_arrivee" # Corrected default and added 'arrivee'

    addr_set = set(drivers[garage_col].astype(str).dropna().str.strip().tolist())
    addr_set |= set(trajets[home_col].astype(str).dropna().str.strip().tolist())
    addr_set |= set(trajets[school_col].astype(str).dropna().str.strip().tolist())
    addr_list = sorted(a for a in addr_set if a and a.lower()!="nan")

    def ors_geocode(query, api_key):
        try:
            url = "https://api.openrouteservice.org/geocode/search"
            params = {"api_key": api_key, "text": query, "size": 1}
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                feats = data.get("features", [])
                if feats:
                    coords = feats[0]["geometry"]["coordinates"]  # [lon, lat]
                    return float(coords[1]), float(coords[0])
        except Exception:
            pass
        return None

    geo_cache = {}
    nomi = Nominatim(user_agent="ase_ors_full")
    nomi_geocode = RateLimiter(nomi.geocode, min_delay_seconds=CONFIG["geocode_rate_delay_s"], swallow_exceptions=True)

    for i, a in enumerate(addr_list):
        a_q = a + ", Haute-Garonne, France"
        latlon = ors_geocode(a_q, OPENROUTESERVICE_API_KEY)
        if latlon is None:
            loc = nomi_geocode(a_q) or nomi_geocode(a)
            if loc: latlon = (loc.latitude, loc.longitude)
        if latlon is None:
            latlon = (43.6045 + np.random.randn()*0.01, 1.4440 + np.random.randn()*0.01)  # Toulouse fallback
        geo_cache[a] = latlon
        if (i+1) % 50 == 0:
            print(f"Geocoded {i+1}/{len(addr_list)}")

    drivers["garage_lat"] = drivers[garage_col].map(lambda a: geo_cache.get(str(a).strip(), (None,None))[0])
    drivers["garage_lon"] = drivers[garage_col].map(lambda a: geo_cache.get(str(a).strip(), (None,None))[1])
    trajets["home_lat"]   = trajets[home_col].map(lambda a: geo_cache.get(str(a).strip(), (None,None))[0])
    trajets["home_lon"]   = trajets[home_col].map(lambda a: geo_cache.get(str(a).strip(), (None,None))[1])
    trajets["school_lat"] = trajets[school_col].map(lambda a: geo_cache.get(str(a).strip(), (None,None))[0])
    trajets["school_lon"] = trajets[school_col].map(lambda a: geo_cache.get(str(a).strip(), (None,None))[1])

    print("Geocoding complete.")

    return drivers, trajets


def compute_costs(drivers: pd.DataFrame, trajets: pd.DataFrame):

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0088
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        c = 2*asin(sqrt(a))
        return R*c

    def try_calibrate_speed(df, dist_col, dur_col, cap):
        spd = CONFIG["avg_speed_kmh"]
        if (dist_col in df.columns) and (dur_col in df.columns):
            samp = df[[dist_col, dur_col]].copy()
            samp = samp.apply(pd.to_numeric, errors="coerce").dropna()
            samp = samp[(samp[dist_col] > 0) & (samp[dur_col] > 0)]
            if len(samp) >= 10:
                speeds = (samp[dist_col] / (samp[dur_col]/60.0)).clip(upper=cap)
                spd = float(np.median(speeds))
        return min(spd, cap)

    dist_col   = _guess_column(["distance","km"], trajets.columns)  # optional
    dur_col    = _guess_column(["durée","duree","temps","minutes"], trajets.columns)  # optional
    bell_col   = _guess_column(["cloche","bell","horaire","heure","debut","entrée","entree","depart"], trajets.columns)  # optional, added 'depart'

    CONFIG["avg_speed_kmh"] = try_calibrate_speed(trajets, dist_col, dur_col, CONFIG["cap_speed_kmh"])
    print(f"Using avg speed (km/h): {CONFIG['avg_speed_kmh']:.1f} (≤ {CONFIG['cap_speed_kmh']})")

    # For now, use all data but we'll filter by date in the optimization cell
    unique_children = trajets.groupby('enfant_id').first().reset_index()
    print(f"Total routes: {len(trajets)}")
    print(f"Unique children: {len(unique_children)}")

    # Nodes - use unique children instead of all routes
    depots = [{"type":"depot","driver_i":i,"lat":r["garage_lat"],"lon":r["garage_lon"]} for i,r in drivers.iterrows()]
    pickups = [{"type":"pickup","row_i":i,"lat":r["home_lat"],"lon":r["home_lon"]} for i,r in unique_children.iterrows()]
    drops   = [{"type":"drop","row_i":i,"lat":r["school_lat"],"lon":r["school_lon"]} for i,r in unique_children.iterrows()]
    nodes = depots + pickups + drops
    N = len(nodes)

    # Distance/time matrices
    dist_km = np.zeros((N,N), dtype=float)
    time_min = np.zeros((N,N), dtype=int)
    speed = max(5.0, float(CONFIG["avg_speed_kmh"]))
    for i in range(N):
        for j in range(N):
            if i==j: continue
            d = haversine_km(nodes[i]["lat"], nodes[i]["lon"], nodes[j]["lat"], nodes[j]["lon"])
            dist_km[i,j] = d
            time_min[i,j] = int(round(60 * d / max(1e-6, speed)))

    print("Matrices:", dist_km.shape, time_min.shape)

    return dist_km, time_min


def solve(drivers: pd.DataFrame, trajets: pd.DataFrame, solve_date: str = "01/01/2024"):

    cap_col    = _guess_column(["capacite","places","passagers","seat"], drivers.columns) or "capacite passagers Véhicule"
    cost_col   = _guess_column(["cout","kilomet","€/km","euro","euro/km"], drivers.columns) or "Cout kilométrique (€)"

    num_vehicles = len(drivers)
    vehicle_caps = pd.to_numeric(drivers[cap_col], errors="coerce").fillna(8).astype(int).clip(lower=1).tolist()  # Default to 8 if missing
    vehicle_cost = pd.to_numeric(drivers[cost_col], errors="coerce").fillna(1.0).astype(float).clip(lower=0.1).tolist()
    starts = list(range(num_vehicles))
    ends   = list(range(num_vehicles))

    def solve_for_date():
        """Solve the routing problem for a specific date"""

        # Filter data for this specific date
        daily_trajets = trajets[trajets['date'] == solve_date].copy()
        daily_unique_children = daily_trajets.groupby('enfant_id').first().reset_index()

        print(f"Routes for {solve_date}: {len(daily_trajets)}")
        print(f"Unique children for {solve_date}: {len(daily_unique_children)}")

        if len(daily_unique_children) == 0:
            print(f"⚠ No children for date {solve_date}")
            return None, None, None, None, None, None

        # Create nodes for this specific date
        daily_depots = [{"type":"depot","driver_i":i,"lat":r["garage_lat"],"lon":r["garage_lon"]} for i,r in drivers.iterrows()]
        daily_pickups = [{"type":"pickup","row_i":i,"lat":r["home_lat"],"lon":r["home_lon"]} for i,r in daily_unique_children.iterrows()]
        daily_drops = [{"type":"drop","row_i":i,"lat":r["school_lat"],"lon":r["school_lon"]} for i,r in daily_unique_children.iterrows()]
        daily_nodes = daily_depots + daily_pickups + daily_drops

        # Create demands for this date
        daily_demands = np.zeros(len(daily_nodes), dtype=int)
        daily_kid_to_nodes = {}

        for idx in daily_unique_children.index.tolist():
            p = num_vehicles + (idx - daily_unique_children.index.min())
            d = num_vehicles + len(daily_unique_children) + (idx - daily_unique_children.index.min())
            daily_kid_to_nodes[idx] = (p,d)
            daily_demands[p] = 1
            daily_demands[d] = -1

        # Create manager and routing for this date
        manager = pywrapcp.RoutingIndexManager(len(daily_nodes), num_vehicles, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        # Time dimension - create a simple distance matrix for this day
        daily_time_matrix = np.zeros((len(daily_nodes), len(daily_nodes)), dtype=int)
        for i in range(len(daily_nodes)):
            for j in range(len(daily_nodes)):
                if i == j:
                    daily_time_matrix[i, j] = 0
                else:
                    # Use haversine distance for daily nodes
                    lat1, lon1 = daily_nodes[i]['lat'], daily_nodes[i]['lon']
                    lat2, lon2 = daily_nodes[j]['lat'], daily_nodes[j]['lon']
                    if lat1 is not None and lon1 is not None and lat2 is not None and lon2 is not None:
                        from math import radians, sin, cos, asin, sqrt
                        R = 6371.0088
                        dlat = radians(lat2 - lat1)
                        dlon = radians(lon2 - lon1)
                        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
                        c = 2*asin(sqrt(a))
                        distance_km = R * c
                        # Convert to minutes using average speed
                        time_minutes = max(1, int(60 * distance_km / max(1, CONFIG["avg_speed_kmh"])))
                        daily_time_matrix[i, j] = time_minutes
                    else:
                        daily_time_matrix[i, j] = 1  # Default 1 minute

        def daily_transit_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return daily_time_matrix[from_node, to_node]

        transit_time_index = routing.RegisterTransitCallback(daily_transit_callback)
        routing.AddDimension(transit_time_index, 0, 24*60, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        # Cost function - create daily distance matrix for costs
        daily_dist_matrix = np.zeros((len(daily_nodes), len(daily_nodes)), dtype=float)
        for i in range(len(daily_nodes)):
            for j in range(len(daily_nodes)):
                if i == j:
                    daily_dist_matrix[i, j] = 0
                else:
                    # Use haversine distance for daily nodes
                    lat1, lon1 = daily_nodes[i]['lat'], daily_nodes[i]['lon']
                    lat2, lon2 = daily_nodes[j]['lat'], daily_nodes[j]['lon']
                    if lat1 is not None and lon1 is not None and lat2 is not None and lon2 is not None:
                        from math import radians, sin, cos, asin, sqrt
                        R = 6371.0088
                        dlat = radians(lat2 - lat1)
                        dlon = radians(lon2 - lon1)
                        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
                        c = 2*asin(sqrt(a))
                        daily_dist_matrix[i, j] = R * c
                    else:
                        daily_dist_matrix[i, j] = 0.1  # Default 0.1 km

        cost_scaler = 1000
        def make_cost_cb(v):
            coef = float(vehicle_cost[v])
            def cb(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return int(round(daily_dist_matrix[from_node, to_node] * coef * cost_scaler))
            return cb
        for v in range(num_vehicles):
            routing.SetArcCostEvaluatorOfVehicle(routing.RegisterTransitCallback(make_cost_cb(v)), v)

        # Capacity constraints
        demand_index = routing.RegisterUnaryTransitCallback(
            lambda index: int(daily_demands[manager.IndexToNode(index)])
        )
        routing.AddDimensionWithVehicleCapacity(demand_index, 0, list(map(int, vehicle_caps)), True, "Capacity")

        # Flexible time windows
        for node in range(len(daily_nodes)):
            idx = manager.NodeToIndex(node)
            if node < num_vehicles:  # Depot nodes
                time_dim.CumulVar(idx).SetRange(0, 24*60)
            else:  # Pickup/drop nodes
                time_dim.CumulVar(idx).SetRange(0, 24*60)

        # Pickup and delivery constraints
        for kid_i, (p, d) in daily_kid_to_nodes.items():
            p_idx = manager.NodeToIndex(p)
            d_idx = manager.NodeToIndex(d)
            routing.AddPickupAndDelivery(p_idx, d_idx)
            routing.solver().Add(routing.VehicleVar(p_idx) == routing.VehicleVar(d_idx))
            routing.solver().Add(time_dim.CumulVar(p_idx) <= time_dim.CumulVar(d_idx))

        def solve_with_progress(routing, params, time_limit_sec):

            best_solution = {"value": None, "solution": None}
            def solver_thread():
                solution = routing.SolveWithParameters(params)
                if solution:
                    best_solution["solution"] = solution
                    # best_solution["value"] = routing.GetObjectiveValue()

            thread = threading.Thread(target=solver_thread)
            thread.start()

            with tqdm(total=time_limit_sec, desc="Solving", unit="s") as pbar:
                for _ in range(time_limit_sec):
                    if not thread.is_alive():
                        break
                    pbar.update(1)
                    time.sleep(1)

            thread.join()
            return best_solution["solution"]

        # Usage
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromSeconds(CONFIG["time_limit_sec"])
        params.log_search = False

        solution = solve_with_progress(routing, params, time_limit_sec=CONFIG["time_limit_sec"])

        if solution is not None:
            print(f"✅ Solution found for {solve_date}!")
            return manager, routing, time_dim, solution, daily_nodes, daily_unique_children
        else:
            print(f"⚠ No solution found for {solve_date}")
            return None, None, None, None, daily_nodes, daily_unique_children

    manager, routing, time_dim, solution, nodes, unique_children = solve_for_date()

    if solution is None:
        return None, None, None, None

    print(f"✅ Solution found for day: {solve_date}")
    print(f"📊 Processing {len(unique_children)} children with {num_vehicles} vehicles.")
    print(f"📊 Daily capacity: {sum(vehicle_caps)} seats for {len(unique_children)} children")
    print(f"📊 Capacity utilization: {len(unique_children)/sum(vehicle_caps)*100:.1f}%")

    return manager, routing, time_dim, solution


def optimize(drivers: pd.DataFrame, 
             trajets: pd.DataFrame, 
             manager, 
             routing, 
             time_dim, 
             solution):

    if manager is None or routing is None or time_dim is None or solution is None:
        return pd.DataFrame()

    def mm_to_hhmm(m):
        m = int(m); return f"{m//60:02d}:{m%60:02d}"

    rows = []
    routes_struct = []
    total_cost_eur = 0.0
    vehicle_stats = {}
    num_vehicles = len(drivers)

    # Get the minimum index from unique_children for proper indexing
    unique_children = trajets.groupby('enfant_id').first().reset_index()
    min_idx = unique_children.index.min()

    vehicle_caps = pd.to_numeric(drivers[cap_col], errors="coerce").fillna(8).astype(int).clip(lower=1).tolist()  # Default to 8 if missing
    vehicle_cost = pd.to_numeric(drivers[cost_col], errors="coerce").fillna(1.0).astype(float).clip(lower=0.1).tolist()

    # Build comprehensive route data
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route_stops = []
        route_cost = 0.0
        order = 0
        children_served = 0

        # Check if vehicle has any stops
        if routing.IsEnd(idx):
            print(f"⚠️ Vehicle {v} has no assigned stops")
            vehicle_stats[v] = {"stops": 0, "children": 0, "cost": 0.0, "active": False}
            continue

        depots = [{"type":"depot","driver_i":i,"lat":r["garage_lat"],"lon":r["garage_lon"]} for i,r in drivers.iterrows()]
        pickups = [{"type":"pickup","row_i":i,"lat":r["home_lat"],"lon":r["home_lon"]} for i,r in unique_children.iterrows()]
        drops   = [{"type":"drop","row_i":i,"lat":r["school_lat"],"lon":r["school_lon"]} for i,r in unique_children.iterrows()]
        nodes = depots + pickups + drops

        # Extract route stops
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            arr = solution.Value(time_dim.CumulVar(idx))

            # Determine stop type and details
            stop_info = {
                "vehicle": v,
                "order": order,
                "node": node,
                "type": nodes[node]["type"],
                "arrive_min": arr,
                "arrive_hhmm": mm_to_hhmm(arr),
                "lat": nodes[node]["lat"],
                "lon": nodes[node]["lon"],
            }

            # Add child information for pickup/drop nodes
            if nodes[node]["type"] == "pickup":
                # Calculate child index based on node position and unique_children indexing
                child_pos = node - num_vehicles
                if child_pos < len(unique_children):
                    child_idx = unique_children.index[child_pos]
                    stop_info["child_id"] = child_idx
                    stop_info["address"] = unique_children.at[child_pos, home_col] if child_pos < len(unique_children) else "Unknown"
                    children_served += 1
            elif nodes[node]["type"] == "drop":
                # Calculate child index for drop nodes
                child_pos = node - num_vehicles - len(unique_children)
                if child_pos < len(unique_children):
                    child_idx = unique_children.index[child_pos]
                    stop_info["child_id"] = child_idx
                    stop_info["address"] = unique_children.at[child_pos, school_col] if child_pos < len(unique_children) else "Unknown"
            elif nodes[node]["type"] == "depot":
                stop_info["address"] =  "Unknown" #drivers.at[0, garage_col] if v < len(drivers) else "Unknown"

            rows.append(stop_info)
            route_stops.append((nodes[node]["lat"], nodes[node]["lon"]))

            # Calculate cost to next stop
            nxt = solution.Value(routing.NextVar(idx))
            if not routing.IsEnd(nxt):
                i = node; j = manager.IndexToNode(nxt)
                # Create a simple distance matrix for cost calculation
                if i < len(nodes) and j < len(nodes):
                    from math import radians, sin, cos, asin, sqrt
                    R = 6371.0088
                    lat1, lon1 = nodes[i]["lat"], nodes[i]["lon"]
                    lat2, lon2 = nodes[j]["lat"], nodes[j]["lon"]
                    if all(coord is not None for coord in [lat1, lon1, lat2, lon2]):
                        dlat = radians(lat2 - lat1)
                        dlon = radians(lon2 - lon1)
                        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
                        c = 2*asin(sqrt(a))
                        distance_km = R * c
                        route_cost += distance_km * float(vehicle_cost[v])

            idx = nxt; order += 1

        # End node (return to depot)
        node = manager.IndexToNode(idx)
        arr = solution.Value(time_dim.CumulVar(idx))
        rows.append({
            "vehicle": v, "order": order, "node": node, "type": nodes[node]["type"],
            "arrive_min": arr, "arrive_hhmm": mm_to_hhmm(arr),
            "lat": nodes[node]["lat"], "lon": nodes[node]["lon"],
            "address": "Unknown" #drivers.at[v, garage_col] if v < len(drivers) else "Unknown"
        })
        route_stops.append((nodes[node]["lat"], nodes[node]["lon"]))

        total_cost_eur += route_cost
        routes_struct.append({
            "vehicle": v,
            "seq": route_stops,
            "cost_eur": route_cost,
            "stops": len(route_stops),
            "children": children_served
        })

        vehicle_stats[v] = {
            "stops": len(route_stops),
            "children": children_served,
            "cost": route_cost,
            "active": len(route_stops) > 1
        }

    # Create comprehensive routes DataFrame
    routes_df = pd.DataFrame(rows).sort_values(["vehicle","order"]).reset_index(drop=True)

    # Check for unassigned children
    assigned_children = set()
    for _, row in routes_df.iterrows():
        if 'child_id' in row and pd.notna(row['child_id']):
            assigned_children.add(int(row['child_id']))

    all_children = set(unique_children.index.tolist())
    unassigned = all_children - assigned_children

    if unassigned:
        print(f"\n⚠️ UNASSIGNED CHILDREN ({len(unassigned)}):")
        for child_id in sorted(unassigned):
            if child_id in unique_children.index:
                child_data = unique_children.loc[child_id]
                print(f"  - Child {child_id}: {child_data[home_col]} → {child_data[school_col]}")
    
    return routes_df


import pandas as pd
from typing import List, Dict


def dispatch_children_to_drivers(driver_dfs: List[pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    """
    Dispatch instructions across multiple drivers so each child_id
    is assigned to exactly one driver. Each driver ends up with a dataframe
    containing all instructions for their assigned children.

    Args:
        driver_dfs (list of pd.DataFrame): One dataframe per driver.
            Each dataframe contains the same instructions in different order.

    Returns:
        dict: {driver_index: driver_df}
    """
    # Assume all drivers have the same set of child_ids
    all_child_ids = sorted(
        pd.to_numeric(driver_dfs[0]["child_id"], errors="coerce").dropna().unique()
    )
    
    num_drivers = len(driver_dfs)
    dispatch: Dict[int, pd.DataFrame] = {}

    # Round-robin assignment of child_ids to drivers
    for driver_index, driver_name in enumerate(drivers["prenom"].tolist()):
        assigned_ids = [
            child_id for i, child_id in enumerate(all_child_ids) if i % num_drivers == driver_index
        ]
        
        df = driver_dfs[driver_index]
        driver_df = (
            df[df["child_id"].isin(assigned_ids)]
            .sort_values("order")
            .reset_index(drop=True)
        )

        dispatch[driver_name] = driver_df

    return dispatch


def discriminate_drivers(trajets: pd.DataFrame, drivers: pd.DataFrame):
    
    routes_df = []
    for available_driver in tqdm(drivers["prenom"].tolist()):
        print("CURRENT DRIVER", available_driver)
        driver: pd.DataFrame = drivers[drivers["prenom"]==available_driver]
        route_df = optimize(driver, trajets, *solve(drivers=driver, trajets=trajets, solve_date=CONFIG["date"]))
        route_df.to_csv(f"{available_driver}.csv", index=False)
        routes_df.append(route_df)
        
    for name, df in dispatch_children_to_drivers(driver_dfs=routes_df).items():
        print(name)
        df.head()
        df.to_csv(f"{name}.csv", index=False)


if __name__ == "__main__" and "solve" in sys.argv:
    map_data(drivers=drivers, trajets=trajets)
    # distance_matrix, time_matrix = compute_costs(drivers=drivers, trajets=trajets)
    # print(distance_matrix)
    # print(time_matrix)
    discriminate_drivers(trajets=trajets, drivers=drivers)

    # routes_df = optimize(drivers, trajets, *solve(drivers=drivers, trajets=trajets, solve_date=CONFIG["date"]))
    


if __name__ == "__main__" and "show" in sys.argv:

    import folium

    m = folium.Map(location=[43.6045, 1.4440], zoom_start=11)

    # Create layer groups for each vehicle
    vehicle_layers = {}
    active_routes = 0
    total_stops = 0

    # Base map
    m = folium.Map(location=[43.6045, 1.4440], zoom_start=11)

    # Colors and dash patterns for vehicles
    vehicle_colors = ["#FF0000", "#0066FF", "#00CC00", "#FF6600", "#9900CC", "#FF0066"]
    dash_patterns = [None, "5,5", "10,5", "5,5,2,5", "10,5,2,5", "2,5"]

    # Group by vehicle
    for idx, driver_name in enumerate(drivers["prenom"].tolist()):

        routes_df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"{driver_name}.csv"))

        # for vehicle_id, vehicle_df in routes_df.groupby('vehicle'):

        vehicle_group = folium.FeatureGroup(name=f"Driver {driver_name}")
        color = vehicle_colors[idx % len(vehicle_colors)]
        dash = dash_patterns[idx % len(dash_patterns)]

        # Sort by order
        vehicle_df = routes_df.sort_values('order')

        # Draw lines connecting stops
        coords = list(zip(vehicle_df['lat'], vehicle_df['lon']))
        folium.PolyLine(
            coords,
            color=color,
            weight=6,
            opacity=0.8,
            dash_array=dash,
            tooltip=f"Driver {driver_name}"
        ).add_to(vehicle_group)
        
        for idx, row in vehicle_df.iterrows():
            if row['type'] == 'depot':
                if row['order'] == vehicle_df['order'].min():
                    popup = f"<b>Driver {driver_name} - START DEPOT</b><br>{row['address']}<br>Time: {row['arrive_hhmm']}"
                else:
                    popup = f"<b>Driver {driver_name} - END DEPOT</b><br>{row['address']}<br>Time: {row['arrive_hhmm']}"
            elif row['type'] == 'pickup':
                popup = f"<b>PICKUP</b><br>Child: {row['child_id']}<br>Time: {row['arrive_hhmm']}<br>{row['address']}"
            elif row['type'] == 'drop':
                popup = f"<b>DROP</b><br>Child: {row['child_id']}<br>Time: {row['arrive_hhmm']}<br>{row['address']}"
            else:
                popup = f"<b>Stop</b><br>Time: {row['arrive_hhmm']}"

            # Use DivIcon to show the order number inside the marker
            icon = folium.DivIcon(
                html=f"""
                <div style="
                    background-color:{color};
                    color:white;
                    font-weight:bold;
                    border-radius:50%;
                    width:30px;
                    height:30px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    border: 2px solid white;">
                    {row['order']}
                </div>
                """
            )

            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=popup,
                icon=icon
            ).add_to(vehicle_group)

        vehicle_group.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Save map
    m.show_in_browser()
    m.save("vehicle_routes.html")
    print("Map saved as vehicle_routes.html")