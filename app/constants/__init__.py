import os, sys
import logging
from dotenv import load_dotenv
import json
import boto3
from boto3.session import Session

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"), override=True)

logger = logging.Logger(name="Constants")

# CONSTANTS
MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(MAIN_DIR)
TMP_DIR = os.path.join(ROOT_DIR, "tmp")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
BIN_DIR = os.path.join(ROOT_DIR, "bin")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# SECRETS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_ACCESS_KEY_SECRET = os.getenv("AWS_ACCESS_KEY_SECRET", "")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "")
BUCKET_NAME = os.getenv("BUCKET_NAME", "")
SECRET_NAME = os.getenv("SECRET_NAME", "")
AWS_PROFILE=os.getenv("AWS_PROFILE", "")

OPENROUTESERVICE_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImUyZDkwYWZkOWExOTRiOWE5NzE4ZDkxOTI3ZWRjOTQ0IiwiaCI6Im11cm11cjY0In0="

CONFIG = {
    # File name tokens used to auto-detect the right files in the ZIP
    "drivers_patterns": ["Chauffeurs","ASE","Chauffeurs-ASE","chauffeur"],
    "trajets_patterns": ["Trajets","enfants","trajet","Trajets Enfants","enfant"],

    "date": "01/01/2024",

    # Solver & model
    "default_bell_time": "08:30",     # Used if per-row bell time is missing
    "time_limit_sec": 10,            # OR-Tools search time (hard-constraint attempt)
    "fallback_time_limit_sec": 180,   # time for optional-drop mode (if needed)
    "first_solution": "PARALLEL_CHEAPEST_INSERTION",
    "metaheuristic": "GUIDED_LOCAL_SEARCH",

    # Optional-drop settings (to guarantee a solution)
    "drop_pair_penalty_eur": 10000.0, # penalty if a pickup–drop pair is dropped (very large)

    # Travel-time model
    "avg_speed_kmh": 20.0,            # Baseline before calibration
    "cap_speed_kmh": 70.0,            # Hard cap ≤ 65 km/h
    "geocode_rate_delay_s": 0.25,     # Gentle throttle for fallback geocoding

    # Output
    "map_html": "routes.html",
    "routes_csv": "routes.csv",
}

