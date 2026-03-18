#!/usr/bin/env python3
"""Fully vectorized formula discovery using numpy."""
import json, numpy as np, sys, os, time

DATA = '/home/ubuntu/box-box-box/data/historical_races/races_00000-00999.json'

def precompute(races):
    """
    For each race × driver, compute feature vector:
    [laps_S, laps_M, laps_H, agesum_S, agesum_M, agesum_H, n_pits, base_time, temp]
    Also store actual finishing order (as index array).
    """
    records = []
    for race in races:
        rc = race['race_config']
        base = rc['base_lap_time']
        pit_t = rc['pit_lane_time']
        temp = rc['track_temp']
        total_laps = rc['total_laps']
        actual = race['finishing_positions']  # ordered list

        # Compute features for each driver in order of actual finishing position
        # So times[0] should be smallest
        driver_order = {d: i for i, d in enumerate(actual)}
        n_drivers = len(actual)

        lc = np.zeros((n_drivers, 3))   # lap count per compound
        as_ = np.zeros((n_drivers, 3))  # age sum per compound
        aq = np.zeros((n_drivers, 3))   # age^2 sum per compound
        pits = np.zeros(n_drivers)      # pit count
        base_t = np.zeros(n_drivers)    # fixed time component

        t2i = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

        for s in race['strategies'].values():
            did = s['driver_id']
            idx = driver_order[did]
            pit_map = {p['lap']: p['to_tire'] for p in s.get('pit_stops', [])}
            tire = s['starting_tire']
            age = 0
            n_p = 0

            for lap in range(1, total_laps + 1):
                age += 1
                ci = t2i[tire]
                lc[idx, ci] += 1
                as_[idx, ci] += age
                aq[idx, ci] += age * age
                if lap in pit_map:
                    tire = pit_map[lap]
                    age = 0
                    n_p += 1

            pits[idx] = n_p
            base_t[idx] = base * total_laps + n_p * pit_t

        records.append({
            'lc': lc, 'as': as_, 'aq': aq, 'pits': pits,
            'base_t': base_t,
            'temp': temp,
            'n': n_drivers,
        })

    return records

def score_formula(records, params, formula_type):
    """
    Score a set of parameters across all pre-computed races.

    Formula types:
    - 'lin_temp':  off[c] + deg[c] * agesum * temp
    - 'linear':    off[c] + deg[c] * agesum
    - 'quad':      off[c] + deg[c] * age^2_sum
    - 'tempnorm':  off[c] + deg[c] * agesum * temp/30
    - 'addtemp':   off[c] + deg[c] * agesum + alpha * temp_total
    - 'sep':       off[c] + deg[c] * agesum + temp_deg[c] * agesum * temp
    """
    if formula_type == 'lin_temp':
        off = params[:3]      # [S, M, H] offsets
        deg = params[3:6]     # [S, M, H] degradation
    elif formula_type == 'linear':
        off = params[:3]
        deg = params[3:6]
    elif formula_type == 'quad':
        off = params[:3]
        deg = params[3:6]
    elif formula_type == 'tempnorm':
        off = params[:3]
        deg = params[3:6]
    elif formula_type == 'sep':  # compound-specific temp_deg
        off = params[:3]
        deg = params[3:6]
        tdeg = params[6:9]

    correct = 0
    for r in records:
        lc, as_, aq = r['lc'], r['as'], r['aq']
        temp = r['temp']
        t = r['base_t'].copy()

        # Offset contribution
        t += lc @ off

        # Degradation contribution
        if formula_type == 'lin_temp':
            t += (as_ @ deg) * temp
        elif formula_type == 'linear':
            t += as_ @ deg
        elif formula_type == 'quad':
            t += aq @ deg
        elif formula_type == 'tempnorm':
            t += (as_ @ deg) * (temp / 30.0)
        elif formula_type == 'sep':
            t += as_ @ deg
            t += (as_ @ tdeg) * temp

        # Check if times are in ascending order (already ordered by actual finishing)
        if np.all(t[:-1] <= t[1:]):
            correct += 1
        # Note: strict ascending needed (no ties), but use <=  and count exact matches
        # Actually since drivers are in actual finishing order, if t is ascending, prediction matches
        # But need STRICT: t[0] < t[1] < ... < t[N-1]
        # If any are equal, sorting might give wrong result
        # Use argsort check
        pred_order = np.argsort(t)
        if np.all(pred_order == np.arange(r['n'])):
            correct += 1  # double counted, fix below
        # Simpler: just argsort and compare to expected (0,1,2,...,19)
    # Redo cleanly
    correct = 0
    for r in records:
        lc, as_, aq = r['lc'], r['as'], r['aq']
        temp = r['temp']
        t = r['base_t'].copy()
        t += lc @ off

        if formula_type == 'lin_temp':
            t += (as_ @ deg) * temp
        elif formula_type == 'linear':
            t += as_ @ deg
        elif formula_type == 'quad':
            t += aq @ deg
        elif formula_type == 'tempnorm':
            t += (as_ @ deg) * (temp / 30.0)
        elif formula_type == 'sep':
            t += as_ @ deg
            t += (as_ @ tdeg) * temp

        pred = np.argsort(t)
        if np.all(pred == np.arange(r['n'])):
            correct += 1

    return correct / len(records)

def grid_search_vectorized(records, formula_type, param_grids):
    """
    Vectorized grid search. param_grids = list of arrays for each parameter.
    """
    from itertools import product as iprod
    best_acc, best_params = 0, None

    # Build all combinations
    all_combos = list(iprod(*param_grids))
    print(f"  Testing {len(all_combos)} combinations for {formula_type}...")

    t0 = time.time()
    for combo in all_combos:
        params = np.array(combo)
        acc = score_formula(records, params, formula_type)
        if acc > best_acc:
            best_acc = acc
            best_params = combo
            if acc > 0.2:
                print(f"    New best: {acc*100:.1f}% | {combo}")

    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s. Best: {best_acc*100:.1f}% {best_params}")
    return best_acc, best_params

print("Loading and precomputing features...")
with open(DATA) as f:
    races = json.load(f)[:300]

records = precompute(races[:100])
print(f"Precomputed {len(records)} races")

# Print race 0 summary
r = records[0]
print(f"\nRace 0: temp={races[0]['race_config']['track_temp']}")
print(f"  Driver 1 (winner): lc={r['lc'][0]}, as={r['as'][0]}, pits={r['pits'][0]}")
print(f"  Driver 20 (last):  lc={r['lc'][-1]}, as={r['as'][-1]}, pits={r['pits'][-1]}")

# ============================================================
# Quick test specific formulas
# ============================================================
print("\n=== Quick formula tests ===")

# linear_temp formula - test a range of clean values
so_vals = np.array([-0.5, -1.0, -1.5, -2.0])
mo_vals = np.array([0.0])
ho_vals = np.array([0.5, 1.0, 1.5, 2.0])
sd_vals = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
md_vals = np.array([0.0005, 0.001, 0.0015, 0.002])
hd_vals = np.array([0.0002, 0.0003, 0.0005, 0.0007, 0.001])

print("\nFormula: off[c] + deg[c]*agesum*temp (linear_temp)")
bA, pA = grid_search_vectorized(records, 'lin_temp',
    [so_vals, mo_vals, ho_vals, sd_vals, md_vals, hd_vals])

print("\nFormula: off[c] + deg[c]*agesum (linear, no temp)")
sd2 = np.array([0.02, 0.05, 0.08, 0.1, 0.15])
md2 = np.array([0.01, 0.02, 0.03, 0.05])
hd2 = np.array([0.005, 0.01, 0.015, 0.02])
bB, pB = grid_search_vectorized(records, 'linear',
    [so_vals, mo_vals, ho_vals, sd2, md2, hd2])

print("\nFormula: off[c] + deg[c]*agesum*temp/30 (temp_norm)")
bD, pD = grid_search_vectorized(records, 'tempnorm',
    [so_vals, mo_vals, ho_vals, sd2, md2, hd2])

print("\n=== RESULTS SUMMARY ===")
print(f"lin_temp:  {bA*100:.1f}% {pA}")
print(f"linear:    {bB*100:.1f}% {pB}")
print(f"temp_norm: {bD*100:.1f}% {pD}")

# Best formula - do finer search
best_type = max([('lin_temp',bA,pA),('linear',bB,pB),('temp_norm',bD,pD)], key=lambda x:x[1])
print(f"\nBest formula type: {best_type[0]} ({best_type[1]*100:.1f}%)")

# Fine search around best params
if best_type[0] == 'lin_temp':
    bp = best_type[2]
    print(f"\nFine search around {bp}...")
    eps = 0.0002
    so_f = np.arange(bp[0]-0.3, bp[0]+0.31, 0.1)
    mo_f = np.array([0.0])
    ho_f = np.arange(bp[2]-0.3, bp[2]+0.31, 0.1)
    sd_f = np.arange(max(0.0001, bp[3]-0.001), bp[3]+0.0011, 0.0002)
    md_f = np.arange(max(0.0001, bp[4]-0.0005), bp[4]+0.00051, 0.0001)
    hd_f = np.arange(max(0.0001, bp[5]-0.0002), bp[5]+0.00021, 0.00005)
    bF, pF = grid_search_vectorized(records, 'lin_temp', [so_f, mo_f, ho_f, sd_f, md_f, hd_f])
    print(f"Fine result: {bF*100:.1f}% {pF}")

    # Validate on 300 races
    recs300 = precompute(races[:300])
    if pF:
        acc300 = score_formula(recs300, np.array(pF), 'lin_temp')
        print(f"300-race accuracy: {acc300*100:.1f}%")
