
import sys, os
import pandas as pd


from constants import LOGS_DIR, CONFIG_DIR, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER, POSTGRES_SCHEMA, POSTGRES_DATABASE, \
                    POSTGRES_MASTER_USER, POSTGRES_MASTER_PASSWORD, DATA_DIR, \
                    AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET, AWS_REGION_NAME


print(os.path.join(DATA_DIR, "Chauffeurs-ASE.csv"))
drivers = pd.read_csv(
    filepath_or_buffer=os.path.join(DATA_DIR, "Chauffeurs-ASE.csv"),
    delimiter=";",
    # dtype=object,
    encoding="latin1"
)
print(drivers.columns)

journeys = pd.read_csv(
    filepath_or_buffer=os.path.join(DATA_DIR, "Trajets Enfants.csv"),
    delimiter=";",
    # dtype=object,
    encoding="latin1"  
)
print(journeys.columns)

children = pd.read_csv(
    filepath_or_buffer=os.path.join(DATA_DIR, "Liste_Enfants.csv"),
    delimiter=";",
    # dtype=object,
    encoding="latin1"  
)
print(children.columns)


merged = (
    journeys
    .merge(children, left_on="enfant_id", right_on="id", suffixes=("", "_child"))
    .rename(columns={
        "id_x": "id",
        "enfant_id": "child_id",
        "prenom": "child_name",
        "type_demande": "demand_type",
        "lieu_depart": "start_location",
        "lieu_arrivee": "arrival_location",
        "heure_depart": "leave_at",
        "distance_km": "journey_distance",
        "duree_estimee": "journey_length",
        "cout_taxi_estime": "cost"
    })
)

print(merged.head())

# Ensure leave_at is a datetime
merged["leave_at"] = pd.to_datetime(
    merged["date"].astype(str) + " " + merged["leave_at"].astype(str),
    errors="coerce"
)

# Convert journey_length to numeric minutes (assume minutes if stored as string/number)
merged["journey_length"] = pd.to_numeric(merged["journey_length"], errors="coerce")

# Compute arrive_at = leave_at + journey_length minutes
merged["arrive_at"] = merged["leave_at"] + pd.to_timedelta(merged["journey_length"], unit="m")

# Reorder columns
merged = merged[
    ["id", "child_id", "child_name", "demand_type",
     "start_location", "arrival_location", "leave_at", "arrive_at",
     "journey_distance", "journey_length", "cost"]
]

merged["leave_at"] = pd.to_datetime(merged["leave_at"], format="%Y-%m-%d %H:%M:%S")
merged["arrive_at"] = pd.to_datetime(merged["arrive_at"], format="%Y-%m-%d %H:%M:%S")

print(merged.head())

merged.to_csv(os.path.join(DATA_DIR, "output.csv"), index=False)
