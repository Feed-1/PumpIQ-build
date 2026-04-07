"""
combine_machine_data.py

Combines three machine runtime tables:
  - currently_running: machines actively in service
  - replaced_batch_1 : first batch of retired/replaced machines
  - replaced_batch_2 : second batch of retired/replaced machines

A 'status' column is added to each table before merging:
  "Running"  – machine is currently in service
  "Failed"   – machine was replaced due to a failure
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Table 1 – Currently running machines
# ---------------------------------------------------------------------------
currently_running = pd.DataFrame(
    {
        "make": [
            "Grundfos", "Xylem", "Flowserve", "KSB", "Sulzer",
            "Goulds",  "Wilo",  "Armstrong",
        ],
        "model": [
            "CM5-4",    "LR 4150",  "Mark 3",  "Etanorm 40-250",
            "CPE 65-1", "3196 MTX", "VeroLine", "Design Envelope 4380",
        ],
        "process": [
            "Water Supply",  "HVAC Circulation", "Chemical Transfer",
            "Boiler Feed",   "Cooling Tower",    "Crude Oil Transfer",
            "Pressure Boost","District Heating",
        ],
        "run_hours": [4200, 8750, 3100, 11200, 6540, 9800, 2300, 7600],
        "failure_reason": ["N/A"] * 8,
    }
)

# ---------------------------------------------------------------------------
# Table 2 – Replaced machines (batch 1)
# ---------------------------------------------------------------------------
replaced_batch_1 = pd.DataFrame(
    {
        "make": [
            "Grundfos", "Flowserve", "Gorman-Rupp", "ITT",
            "Ebara",    "Pentair",
        ],
        "model": [
            "NB 50-160/160", "Durco Mark II", "T6A60S", "PumpSmart PS220",
            "3M Series",     "Aurora 341",
        ],
        "process": [
            "Wastewater",       "Solvent Recirculation", "Slurry Transfer",
            "Condensate Return", "Sea Water Cooling",    "Fire Suppression",
        ],
        "run_hours": [15300, 22100, 9870, 18600, 12450, 7800],
        "failure_reason": [
            "Seal Failure",
            "Impeller Erosion",
            "Bearing Overheating",
            "Cavitation Damage",
            "Corrosion",
            "Motor Burnout",
        ],
    }
)

# ---------------------------------------------------------------------------
# Table 3 – Replaced machines (batch 2)
# ---------------------------------------------------------------------------
replaced_batch_2 = pd.DataFrame(
    {
        "make": [
            "Sulzer",    "KSB",       "Wilo",     "Xylem",
            "Goulds",    "Weir Group","Netzsch",
        ],
        "model": [
            "Ahlstar A22-1", "Omega 100-315", "MHI 805 EM", "e-SV 32-10",
            "AF Series",     "GEHO TZPM",     "NEMO Series",
        ],
        "process": [
            "Pulp & Paper",    "Mining Dewatering", "Pressure Boosting",
            "Reverse Osmosis", "Petroleum Refinery","Tailings Transfer",
            "Food Processing",
        ],
        "run_hours": [31000, 28500, 17200, 24700, 19350, 41200, 8960],
        "failure_reason": [
            "Shaft Fracture",
            "Abrasive Wear",
            "Voltage Surge",
            "Membrane Fouling",
            "Coke Buildup",
            "Liner Cracking",
            "Contamination",
        ],
    }
)

# ---------------------------------------------------------------------------
# Add status column
# ---------------------------------------------------------------------------
currently_running["status"] = "Running"
replaced_batch_1["status"]  = "Failed"
replaced_batch_2["status"]  = "Failed"

# ---------------------------------------------------------------------------
# Combine all three tables
# ---------------------------------------------------------------------------
combined = (
    pd.concat(
        [currently_running, replaced_batch_1, replaced_batch_2],
        ignore_index=True,
    )
    .rename(columns={"run_hours": "run_hours"})   # explicit no-op keeps intent clear
)

# Reorder columns for readability
combined = combined[["make", "model", "process", "run_hours", "failure_reason", "status"]]

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
pd.set_option("display.max_rows",    100)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width",       120)
pd.set_option("display.max_colwidth", 30)

print("=" * 90)
print("COMBINED MACHINE RUNTIME DATA")
print("=" * 90)
print(combined.to_string(index=True))
print()

print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Total machines    : {len(combined)}")
print(f"Currently running : {(combined['status'] == 'Running').sum()}")
print(f"Failed / replaced : {(combined['status'] == 'Failed').sum()}")
print()

print("Run hours by status:")
print(combined.groupby("status")["run_hours"].agg(["sum", "mean", "min", "max"]).rename(
    columns={"sum": "total_hrs", "mean": "avg_hrs", "min": "min_hrs", "max": "max_hrs"}
))
print()

print("Top failure reasons:")
failed = combined[combined["status"] == "Failed"]
print(failed["failure_reason"].value_counts().to_string())

# Optionally save to CSV
combined.to_csv("machine_data_combined.csv", index=False)
print("\nSaved → machine_data_combined.csv")
