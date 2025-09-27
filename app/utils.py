import pandas as pd
from datetime import datetime

import pandas as pd
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def filter_by_date(df: pd.DataFrame, date_str: str, column: str = "leave_at", format="%d/%m/%Y") -> pd.DataFrame:
    """
    Filters the dataframe for rows where the given datetime column
    matches the specified date (format: 'dd/mm/yyyy').

    Parameters:
        df (pd.DataFrame): Your dataframe
        date_str (str): Date string in 'dd/mm/yyyy' format
        column (str): Column to filter on ('leave_at' or 'arrive_at')

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # Parse the target date (without time)
    target_date = datetime.strptime(date_str, format).date()

    # Ensure the datetime column is parsed correctly
    df[column] = pd.to_datetime(df[column], format=format)

    # Filter rows matching the date part
    return df[df[column].dt.date == target_date]



def plot_journeys(df: pd.DataFrame):
    """
    Plot journeys showing leave_at and arrive_at across the day.
    Each journey is a horizontal bar.
    """
    # Ensure datetime parsing
    df["leave_at"] = pd.to_datetime(df["leave_at"], format="%Y-%m-%d %H:%M:%S")
    df["arrive_at"] = pd.to_datetime(df["arrive_at"], format="%Y-%m-%d %H:%M:%S")

    # Sort by leave_at for a cleaner plot
    df = df.sort_values("leave_at").reset_index(drop=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each journey as a horizontal bar
    for i, row in df.iterrows():
        ax.barh(
            y=i,
            width=(row["arrive_at"] - row["leave_at"]).seconds / 3600,  # duration in hours
            left=row["leave_at"].hour + row["leave_at"].minute / 60,    # start time in hours
            height=0.4,
            label=row["child_name"]
        )
        # Add labels for clarity
        ax.text(
            row["leave_at"].hour + row["leave_at"].minute / 60,
            i,
            row["child_name"],
            va="center",
            ha="right",
            fontsize=8
        )

    # Format axes
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["id"])
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Journey ID")
    ax.set_title("Journeys Timeline (Leave → Arrive)")

    # X axis covers 24 hours
    ax.set_xlim(0, 24)

    plt.tight_layout()
    plt.show()
