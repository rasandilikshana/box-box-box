#!/usr/bin/env python3
"""
Fast Formula Discovery - Uses analytical approach to find exact parameters.

Key insight: total_time difference between two drivers is LINEAR in parameters.
We can use scipy to minimize ranking errors.

Formula being tested:
  lap_time = base_lap_time + compound_offset[c] + compound_deg[c] * tire_age * track_temp
"""

import json, os, glob, numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import product

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'historical_races')

def compute_stint_features(strategy, race_config):
    """
    Compute per-driver features needed for fast time calculation.
    Returns a dict with per-compound stint summaries.
    """
    total_laps = race_config['total_laps']
    pit_map = {stop['lap']: stop['to_tire'] for stop in strategy.get('pit_stops', [])}
    pit_laps = set(pit_map.keys())

    current_tire = strategy['starting_tire']
    tire_age = 0

    # Accumulate: for each lap, record (compound, tire_age)
    # We need: sum of tire_age per compound, count of laps per compound
    compound_lap_count = {'SOFT': 0, 'MEDIUM': 0, 'HARD': 0}
    compound_age_sum = {'SOFT': 0, 'MEDIUM': 0, 'HARD': 0}
    n_pits = 0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        compound_lap_count[current_tire] += 1
        compound_age_sum[current_tire] += tire_age

        if lap in pit_laps:
            current_tire = pit_map[lap]
            tire_age = 0
            n_pits += 1

    return compound_lap_count, compound_age_sum, n_pits


def compute_total_time_fast(features, race_config, params):
    """
    Fast time computation using pre-computed features.
    params = [soft_off, med_off, hard_off, soft_deg, med_deg, hard_deg, temp_coeff]
    """
    base = race_config['base_lap_time']
    pit = race_config['pit_lane_time']
    temp = race_config['track_temp']
    total_laps = race_config['total_laps']

    soft_off, med_off, hard_off, soft_deg, med_deg, hard_deg, temp_c = params

    offsets = {'SOFT': soft_off, 'MEDIUM': med_off, 'HARD': hard_off}
    degs = {'SOFT': soft_deg, 'MEDIUM': med_deg, 'HARD': hard_deg}

    compound_lap_count, compound_age_sum, n_pits = features

    time = base * total_laps  # fixed component

    for c in ['SOFT', 'MEDIUM', 'HARD']:
        laps = compound_lap_count[c]
        age_sum = compound_age_sum[c]
        time += offsets[c] * laps
        time += degs[c] * age_sum * temp_c * temp

    time += pit * n_pits
    return time


def precompute_race_features(races):
    """Pre-compute all driver features for speed."""
    race_data = []
    for race in races:
        rc = race['race_config']
        drivers = []
        for pos_key, strategy in race['strategies'].items():
            feat = compute_stint_features(strategy, rc)
            drivers.append((strategy['driver_id'], feat))
        actual = race['finishing_positions']
        race_data.append((rc, drivers, actual))
    return race_data


def loss_and_accuracy(params, race_data):
    """Compute loss (wrong pairs) and accuracy."""
    total_wrong_pairs = 0
    correct_races = 0

    for rc, drivers, actual in race_data:
        times = {}
        for did, feat in drivers:
            times[did] = compute_total_time_fast(feat, rc, params)

        predicted = sorted(times.keys(), key=lambda d: times[d])

        if predicted == actual:
            correct_races += 1

        # Count wrong consecutive pairs
        for i in range(len(actual) - 1):
            if times[actual[i]] >= times[actual[i+1]]:
                total_wrong_pairs += 1

    accuracy = correct_races / len(race_data)
    return total_wrong_pairs, accuracy


def main():
    print("Loading races...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))

    # Load first 1000 races
    with open(files[0]) as f:
        races_all = json.load(f)

    races_train = races_all[:500]
    races_val = races_all[500:]

    print(f"Train: {len(races_train)}, Val: {len(races_val)}")
    print("Pre-computing features...")

    train_data = precompute_race_features(races_train)
    val_data = precompute_race_features(races_val)

    def loss_fn(params):
        wrong, _ = loss_and_accuracy(params, train_data)
        return float(wrong)

    # Test initial guess
    x0 = [-0.5, 0.0, 0.5, 0.1, 0.05, 0.02, 0.001]
    wrong, acc = loss_and_accuracy(x0, train_data)
    print(f"\nInitial guess: wrong_pairs={wrong}, accuracy={acc*100:.1f}%")

    # ============================================================
    # Phase 1: Try many clean parameter combinations (grid search)
    # ============================================================
    print("\n--- Phase 1: Grid Search over Clean Parameters ---")

    # Focus on likely clean values for competition
    soft_offs = [-0.3, -0.5, -0.8, -1.0, -1.2, -1.5, -2.0]
    hard_offs = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    soft_degs = [0.02, 0.03, 0.05, 0.06, 0.08, 0.1]
    med_degs  = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    hard_degs = [0.005, 0.008, 0.01, 0.015, 0.02]
    temp_cs   = [0.0005, 0.001, 0.0015, 0.002, 0.003]

    best_wrong = float('inf')
    best_params = x0

    # Coarse grid first (test 100 races for speed)
    mini_data = train_data[:100]

    from itertools import product as iproduct
    count = 0
    for s_off, h_off, s_deg, m_deg, h_deg, t_c in iproduct(soft_offs, hard_offs, soft_degs, med_degs, hard_degs, temp_cs):
        p = [s_off, 0.0, h_off, s_deg, m_deg, h_deg, t_c]
        wrong, acc = loss_and_accuracy(p, mini_data)
        if wrong < best_wrong:
            best_wrong = wrong
            best_params = p
            print(f"  New best: wrong={wrong}, acc={acc*100:.1f}% | params={p}")
        count += 1

    print(f"\nGrid searched {count} combinations")
    print(f"Best: wrong={best_wrong}, params={best_params}")

    # Validate best grid params on full train
    wrong, acc = loss_and_accuracy(best_params, train_data)
    print(f"Full train: wrong={wrong}, accuracy={acc*100:.1f}%")

    # ============================================================
    # Phase 2: Differential Evolution optimization
    # ============================================================
    print("\n--- Phase 2: Differential Evolution ---")

    bounds = [
        (-3.0, 0.0),    # soft_offset
        (-0.1, 0.1),    # medium_offset (near 0)
        (0.0, 3.0),     # hard_offset
        (0.01, 0.5),    # soft_deg
        (0.005, 0.2),   # medium_deg
        (0.001, 0.1),   # hard_deg
        (0.0001, 0.02), # temp_coeff
    ]

    result = differential_evolution(
        loss_fn,
        bounds,
        maxiter=200,
        popsize=20,
        seed=42,
        disp=True,
        tol=0.0001,
        init='latinhypercube',
    )

    opt_params = list(result.x)
    wrong, acc = loss_and_accuracy(opt_params, train_data)
    print(f"\nOptimized: loss={result.fun}, train accuracy={acc*100:.1f}%")
    print(f"Params: {opt_params}")

    wrong_val, acc_val = loss_and_accuracy(opt_params, val_data)
    print(f"Val accuracy: {acc_val*100:.1f}%")

    # ============================================================
    # Phase 3: Round to clean values and test
    # ============================================================
    print("\n--- Phase 3: Rounding to Clean Values ---")

    # Try rounding at different precisions
    for precision in [0.01, 0.005, 0.001]:
        rounded = [round(p / precision) * precision for p in opt_params]
        wrong_r, acc_r = loss_and_accuracy(rounded, train_data)
        print(f"  Rounded to {precision}: wrong={wrong_r}, acc={acc_r*100:.1f}%  {rounded}")

    # ============================================================
    # Phase 4: Test best params on ALL files
    # ============================================================
    print("\n--- Phase 4: Test on More Data ---")
    use_params = opt_params if loss_and_accuracy(opt_params, train_data)[0] < best_wrong else best_params

    for i, fpath in enumerate(files[:5]):
        with open(fpath) as f:
            races_f = json.load(f)
        rd = precompute_race_features(races_f)
        wrong_f, acc_f = loss_and_accuracy(use_params, rd)
        print(f"  File {i}: {os.path.basename(fpath)}: accuracy={acc_f*100:.1f}% (wrong_pairs={wrong_f})")

    print("\n=== FINAL DISCOVERED PARAMETERS ===")
    param_names = ['soft_offset', 'medium_offset', 'hard_offset', 'soft_deg', 'medium_deg', 'hard_deg', 'temp_coeff']
    for name, val in zip(param_names, use_params):
        print(f"  {name}: {val:.6f}")

    # Save discovered params
    out = {'params': dict(zip(param_names, use_params))}
    with open(os.path.join(os.path.dirname(__file__), 'discovered_params.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nParams saved to analysis/discovered_params.json")


if __name__ == '__main__':
    main()
