# -*- coding: utf-8 -*-
"""
Created on 2023-07-14

@author: ivespe

Intro script for Exercise 2 ("Load analysis to evaluate the need for flexibility") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""

# %% Dependencies

import pandapower as pp
import pandapower.plotting as pp_plotting
import pandas as pd
import os
import load_scenarios as ls
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np



# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)
path_data_set         = r"C:\Users\nora0\OneDrive - NTNU\5. KLASSE\GridOptimation\CINELDI_MV_reference_system-flexibility_course_NTNU_public\Data_sets" + "\\"

filename_load_data_fullpath = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')

# Subset of load buses to consider in the grid area, considering the area at the end of the main radial in the grid
bus_i_subset = [90, 91, 92, 96]

# Assumed power flow limit in MW that limit the load demand in the grid area (through line 85-86)
P_lim = 0.637 

# Maximum load demand of new load being added to the system
P_max_new = 0.4

# Which time series from the load data set that should represent the new load
i_time_series_new_load = 90


# %% Read pandapower network

net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)

# %% Extract hourly load time series for a full year for all the load points in the CINELDI reference system
# (this code is made available for solving task 3)

load_profiles = lp.load_profiles(filename_load_data_fullpath)

# Get all the days of the year
repr_days = list(range(1,366))

# Get normalized load profiles for representative days mapped to buses of the CINELDI reference grid;
# the column index is the bus number (1-indexed) and the row index is the hour of the year (0-indexed)
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,repr_days)

# Retrieve normalized load time series for new load to be added to the area
new_load_profiles = load_profiles.get_profile_days(repr_days)
new_load_time_series = new_load_profiles[i_time_series_new_load]*P_max_new

# Calculate load time series in units MW (or, equivalently, MWh/h) by scaling the normalized load time series by the
# maximum load value for each of the load points in the grid data set (in units MW); the column index is the bus number
# (1-indexed) and the row index is the hour of the year (0-indexed)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])
# %%



# OPPGAVE 1
# %% Voltage profile along the main feeder (bus 0 -> bus 96)

import pandapower.topology as ppt
import networkx as nx
import re

pp.runpp(net, algorithm="nr")

# Bygg graf
G = ppt.create_nxgraph(net, respect_switches=True, include_trafos=True, multi=False)

# Hjelper: slå opp bus enten på indeks eller navn (f.eks. "Bus 96", "96")
def resolve_bus(net, wanted):
    # 1) direkte indeks
    if wanted in net.bus.index:
        return int(wanted)
    # 2) på navn
    names = net.bus["name"].astype(str).fillna("")
    # eksakte treff: "96" eller "Bus 96"
    mask = names.eq(str(wanted)) | names.eq(f"Bus {wanted}") | names.eq(f"bus {wanted}")
    if not mask.any():
        # fallback: navn som slutter med tallet, f.eks. "CINELDI Bus 96"
        mask = names.str.match(rf".*\b{re.escape(str(wanted))}\b$")
    if mask.any():
        return int(net.bus.index[mask][0])
    raise KeyError(f"Fant ikke bus '{wanted}' verken som indeks eller navn.")

# Finn start fra ext_grid (slakk)
start_bus = int(net.ext_grid.bus.iloc[0])

# Prøv å finne “96” etter navn/indeks; hvis ikke, velg fjerneste av interesse-bussene
candidates = [90, 91, 92, 96]
resolved = []
for b in candidates:
    try:
        resolved.append(resolve_bus(net, b))
    except KeyError:
        pass

if not resolved:
    # Ingen av kandidatene finnes ved navn -> velg den fjerneste noden fra start
    # (nyttig hvis nummereringen er helt annerledes)
    lengths = nx.single_source_shortest_path_length(G, start_bus)
    end_bus = max(lengths, key=lengths.get)
else:
    # Velg den av kandidatene som er lengst unna start_bus
    lengths = nx.single_source_shortest_path_length(G, start_bus)
    end_bus = max(resolved, key=lambda x: lengths.get(x, -1))

# Beregn korteste vei
path_buses = nx.shortest_path(G, source=start_bus, target=end_bus)

# Spenningsprofil langs path
vm = net.res_bus.vm_pu
vm_path = vm.loc[path_buses].values

plt.figure()
plt.plot(range(len(path_buses)), vm_path)
plt.xticks(range(len(path_buses)), path_buses, rotation=90)
plt.ylabel('Spenning [p.u.]')
plt.xlabel(f'Bus langs {start_bus}→{end_bus}')
plt.title('Spenningsprofil langs hovedradialen')
plt.grid(True)
plt.tight_layout()
plt.show()

# Laveste spenning i studieområdet (bruk kandidatene som faktisk finnes)
area_buses = resolved if resolved else list(vm.index)  # fallback: hele nettet
vm_area = vm.loc[area_buses]
vmin = vm_area.min()
bus_min = vm_area.idxmin()
print(f"Minste spenning i området {area_buses}: {vmin:.4f} p.u. (på bus {bus_min})")


# OPPGAVE 2
# %% OPPGAVE 2: How voltage decreases as area load increases (matplotlib)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Helper to resolve bus by label (e.g., 90, 91, 92, 96) ----
def resolve_bus_by_label(net, label):
    names = net.bus["name"].astype(str).fillna("")
    # exact matches like "90" or "Bus 90" or names ending with " 90"
    mask = (
        names.eq(str(label)) |
        names.eq(f"Bus {label}") |
        names.str.endswith(f" {label}")
    )
    if mask.any():
        return int(net.bus.index[mask][0])
    # if the label is actually an index that exists
    if label in net.bus.index:
        return int(label)
    raise KeyError(f"Bus '{label}' not found by name or index.")

# ---- Area definition ----
area_labels = [90, 91, 92, 96]
area_buses = [resolve_bus_by_label(net, b) for b in area_labels]

# ---- 1) Base-load table (MW) for the four buses and their sum ----
base_p_all = net.load["p_mw"].copy()
mask_area = net.load["bus"].isin(area_buses)

# handle multiple load rows per bus: sum per bus
base_per_bus = (
    net.load.loc[mask_area, ["bus", "p_mw"]]
      .groupby("bus", as_index=True)["p_mw"]
      .sum()
      .reindex(area_buses, fill_value=0.0)
)

# map back to human-readable labels in requested order
label_map = {b_idx: lbl for b_idx, lbl in zip(area_buses, area_labels)}
table = pd.DataFrame({
    "Bus": [label_map[i] for i in base_per_bus.index],
    "P_base [MW]": base_per_bus.values
})
table.loc[len(table)] = ["Total", table["P_base [MW]"].sum()]

print("\n=== Area base-load table (MW) ===")
print(table.to_string(index=False))

# ---- 2) Sweep proportional scaling on area loads from 1.0 to 2.0 ----
scales = np.linspace(1.0, 2.0, 21)  # 1.00, 1.05, ..., 2.00
rows = []
for s in scales:
    # reset all loads to baseline
    net.load["p_mw"] = base_p_all
    # scale only the loads connected to the area buses
    net.load.loc[mask_area, "p_mw"] = base_p_all.loc[mask_area] * s

    try:
        # numba=False for compatibility; enforce_q_lims to avoid reactive limit issues
        pp.runpp(net, algorithm="nr", numba=False, enforce_q_lims="auto")
        agg_load = float(net.load.loc[mask_area, "p_mw"].sum())
        vmin_area = float(net.res_bus.loc[area_buses, "vm_pu"].min())
        rows.append((s, agg_load, vmin_area))
    except Exception as e:
        print(f"Power flow failed at scale {s:.2f}: {e}")
        rows.append((s, np.nan, np.nan))

# restore baseline loads
net.load["p_mw"] = base_p_all

df = pd.DataFrame(rows, columns=["scale", "agg_load_mw", "vmin_pu"]).dropna()
if df.empty:
    raise RuntimeError("No valid power-flow results produced; check data / solver settings.")

# ---- 3) Interpolate aggregated load at which v_min = 0.95 p.u. ----
df_sorted = df.sort_values("agg_load_mw").reset_index(drop=True)
# Ensure interpolation monotonic direction (vmin decreases with load)
try:
    agg_at_095 = float(np.interp(
        0.95,
        df_sorted["vmin_pu"][::-1].values,      # y-values (descending)
        df_sorted["agg_load_mw"][::-1].values   # x-values aligned
    ))
except Exception:
    agg_at_095 = np.nan

base_agg = float(table.loc[table["Bus"] == "Total", "P_base [MW]"].values[0])
headroom = agg_at_095 - base_agg if np.isfinite(agg_at_095) else np.nan

print(f"\nAggregated area load (base case): {base_agg:.3f} MW")
if np.isfinite(agg_at_095):
    print(f"Aggregated load where v_min ≈ 0.95 p.u.: {agg_at_095:.3f} MW "
          f"(headroom from base ≈ {headroom:.3f} MW)")
else:
    print("Could not determine crossing at 0.95 p.u. – extend sweep or check monotonicity.")

# ---- 4) Plot: lowest voltage vs aggregated area load ----
plt.figure()
plt.plot(df_sorted["agg_load_mw"], df_sorted["vmin_pu"], "-", linewidth=2)
plt.axhline(0.95, linestyle="--", color="red", label="0.95 p.u. limit")
if np.isfinite(agg_at_095):
    plt.axvline(agg_at_095, linestyle=":", color="gray")
    plt.annotate(f"≈ {agg_at_095:.3f} MW @ 0.95 p.u.",
                 xy=(agg_at_095, 0.95), xytext=(10, 15),
                 textcoords="offset points", arrowprops=dict(arrowstyle="->"))

plt.xlabel("Aggregated area load [MW]")
plt.ylabel("Lowest voltage in area [p.u.]")
plt.title("Lowest voltage vs aggregated area load")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



