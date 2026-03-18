#!/usr/bin/env python3
"""
Analytic Formula Discovery via Linear Programming

The lap time formula is:
  lap_time = base + offset[c] + deg[c] * tire_age * temp_coeff * track_temp

Total time for driver i in race r:
  T_i = base*laps + sum_c(offset[c]*lap_count_i[c]) + sum_c(deg[c]*temp_coeff*temp*age_sum_i[c]) + pits_i * pit_time

Define params θ = [off_S, off_M, off_H, eff_deg_S, eff_deg_M, eff_deg_H]
where eff_deg[c] = deg[c] * temp_coeff (combined into single param)

For driver i before driver j: T_i < T_j
=> θ · f(i,j) < (pits_j - pits_i) * pit_time
where f(i,j)[k] = feature_k_i - feature_k_j

This is a linear feasibility problem! With thousands of constraints, the solution
is uniquely determined.

Strategy:
1. Build constraint matrix from all adjacent pairs in first 200 races
2. Solve LP (minimize 0 subject to Ax < b)
3. Extract exact parameters
4. Round to clean values
5. Validate on all races
"""

import json, os, glob, numpy as np
from scipy.optimize import linprog
from scipy.linalg import lstsq

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'historical_races')

def compute_driver_features(strategy, race_config):
    """Returns (lap_count per compound, age_sum per compound, n_pits)"""
    total_laps = race_config['total_laps']
    pit_map = {stop['lap']: stop['to_tire'] for stop in strategy.get('pit_stops', [])}

    current_tire = strategy['starting_tire']
    tire_age = 0

    lap_count = {'SOFT': 0, 'MEDIUM': 0, 'HARD': 0}
    age_sum   = {'SOFT': 0, 'MEDIUM': 0, 'HARD': 0}
    n_pits = 0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        lap_count[current_tire] += 1
        age_sum[current_tire] += tire_age

        if lap in pit_map:
            current_tire = pit_map[lap]
            tire_age = 0
            n_pits += 1

    return lap_count, age_sum, n_pits


def make_feature_vector(feat_i, feat_j, track_temp):
    """
    Feature vector for constraint: driver_i finishes before driver_j
    θ = [off_S, off_M, off_H, eff_deg_S, eff_deg_M, eff_deg_H]
    """
    lc_i, as_i, _ = feat_i
    lc_j, as_j, _ = feat_j

    # offset contribution: (lap_count_i - lap_count_j) * offset[c]
    f = [
        lc_i['SOFT']   - lc_j['SOFT'],
        lc_i['MEDIUM'] - lc_j['MEDIUM'],
        lc_i['HARD']   - lc_j['HARD'],
        (as_i['SOFT']   - as_j['SOFT'])   * track_temp,
        (as_i['MEDIUM'] - as_j['MEDIUM']) * track_temp,
        (as_i['HARD']   - as_j['HARD'])   * track_temp,
    ]
    return f


def build_constraint_matrix(races, max_races=None):
    """
    Build A_ub, b_ub for linprog.
    Constraint for each adjacent pair: θ · f(i,j) < rhs
    i.e., θ · f(i,j) - epsilon <= rhs  → A_ub * θ <= b_ub with b_ub = rhs - epsilon
    """
    A_rows = []
    b_vals = []
    pair_data = []

    if max_races:
        races = races[:max_races]

    for race in races:
        rc = race['race_config']
        track_temp = rc['track_temp']
        pit_time = rc['pit_lane_time']

        # Build feature map
        feat = {}
        for pos_key, strategy in race['strategies'].items():
            did = strategy['driver_id']
            feat[did] = compute_driver_features(strategy, rc)

        actual = race['finishing_positions']  # ordered 1st to 20th

        # For each adjacent pair: actual[i] finishes before actual[i+1]
        for i in range(len(actual) - 1):
            d_faster = actual[i]
            d_slower = actual[i + 1]

            fi = feat[d_faster]
            fj = feat[d_slower]

            fvec = make_feature_vector(fi, fj, track_temp)

            # Known pit penalty contribution
            pit_faster = fi[2]
            pit_slower = fj[2]
            pit_rhs = (pit_slower - pit_faster) * pit_time  # subtract pit diff from both sides

            # Constraint: fvec · θ < pit_rhs
            # For LP (Ax <= b): fvec · θ <= pit_rhs - epsilon
            eps = 1e-4
            A_rows.append(fvec)
            b_vals.append(pit_rhs - eps)
            pair_data.append((d_faster, d_slower, race['race_id']))

    A = np.array(A_rows, dtype=float)
    b = np.array(b_vals, dtype=float)
    return A, b, pair_data


def solve_lp(A, b):
    """Find feasible θ satisfying Ax <= b (minimize nothing)."""
    n_params = A.shape[1]

    # Objective: minimize 0 (feasibility problem)
    # Add bound constraints for physical reasonableness
    bounds = [
        (-5.0, 0.0),   # off_SOFT (negative = faster)
        (-1.0, 1.0),   # off_MEDIUM (reference, ~0)
        (0.0, 5.0),    # off_HARD (positive = slower)
        (0.0, 1.0),    # eff_deg_SOFT (positive degradation)
        (0.0, 0.5),    # eff_deg_MEDIUM
        (0.0, 0.2),    # eff_deg_HARD
    ]

    # Try to find interior point by minimizing sum of slack variables
    # Transform: Ax + s = b, s >= 0; maximize sum(s) to maximize feasibility margin
    # Or simpler: use phase 1 LP

    # Use minimize with L1 objective to find parameters that minimize violations
    from scipy.optimize import linprog

    # Phase 1: find feasible solution
    result = linprog(
        c=np.zeros(n_params),  # minimize 0 (feasibility)
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        method='highs',
        options={'presolve': True, 'disp': False}
    )

    if result.status == 0:
        print(f"LP feasible! Status: {result.message}")
        return result.x
    else:
        print(f"LP infeasible or error: {result.message}")
        print("Trying with slack variables...")
        return None


def solve_with_slack(A, b):
    """
    Minimize sum of violations: min sum(s) s.t. Ax - s <= b, s >= 0
    This finds minimum-violation solution.
    """
    n_params = A.shape[1]
    n_constraints = A.shape[0]

    # Variables: [θ (n_params), s (n_constraints)]
    # Objective: minimize sum(s) = [0,...,0, 1,...,1]
    c = np.concatenate([np.zeros(n_params), np.ones(n_constraints)])

    # Constraints: Aθ - s <= b
    A_ub = np.hstack([A, -np.eye(n_constraints)])
    b_ub = b

    bounds_theta = [
        (-5.0, 0.0),
        (-1.0, 1.0),
        (0.0, 5.0),
        (0.0, 1.0),
        (0.0, 0.5),
        (0.0, 0.2),
    ]
    bounds_s = [(0, None)] * n_constraints

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds_theta + bounds_s,
        method='highs',
        options={'presolve': True, 'disp': False}
    )

    if result.status == 0:
        theta = result.x[:n_params]
        slack = result.x[n_params:]
        total_violation = slack.sum()
        print(f"Solved with slack. Total violation: {total_violation:.4f}")
        print(f"Violated constraints: {(slack > 1e-6).sum()} / {n_constraints}")
        return theta
    else:
        print(f"Slack LP failed: {result.message}")
        return None


def score_params(theta, races, use_temp_coeff=False):
    """
    Evaluate accuracy. theta = [off_S, off_M, off_H, eff_deg_S, eff_deg_M, eff_deg_H]
    eff_deg = deg * temp_coeff (already combined)
    """
    correct = 0
    for race in races:
        rc = race['race_config']
        temp = rc['track_temp']
        pit_time = rc['pit_lane_time']
        total_laps = rc['total_laps']
        base = rc['base_lap_time']

        offsets = {'SOFT': theta[0], 'MEDIUM': theta[1], 'HARD': theta[2]}
        eff_degs = {'SOFT': theta[3], 'MEDIUM': theta[4], 'HARD': theta[5]}

        times = {}
        for pos_key, strategy in race['strategies'].items():
            did = strategy['driver_id']
            lc, as_, n_pits = compute_driver_features(strategy, rc)

            t = base * total_laps
            for c in ['SOFT', 'MEDIUM', 'HARD']:
                t += offsets[c] * lc[c]
                t += eff_degs[c] * as_[c] * temp
            t += n_pits * pit_time
            times[did] = t

        predicted = sorted(times.keys(), key=lambda d: times[d])
        if predicted == race['finishing_positions']:
            correct += 1

    return correct / len(races)


def main():
    print("Loading races...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))

    with open(files[0]) as f:
        races_all = json.load(f)

    print(f"Loaded {len(races_all)} races")

    # ============================================================
    # Build constraint matrix from first 300 races
    # ============================================================
    print("\n--- Building LP Constraint Matrix ---")
    train_races = races_all[:300]
    A, b, pairs = build_constraint_matrix(train_races)
    print(f"Constraint matrix: {A.shape[0]} constraints, {A.shape[1]} params")
    print(f"b range: [{b.min():.2f}, {b.max():.2f}]")

    # ============================================================
    # Solve LP
    # ============================================================
    print("\n--- Solving LP ---")
    theta = solve_lp(A, b)

    if theta is None:
        print("Direct LP failed, trying slack approach...")
        theta = solve_with_slack(A, b)

    if theta is None:
        print("Both LP approaches failed!")
        return

    print(f"\nLP solution: {theta}")

    # Score on training data
    acc = score_params(theta, train_races)
    print(f"Training accuracy: {acc*100:.1f}%")

    # Score on all 1000 races
    acc_all = score_params(theta, races_all)
    print(f"Full file accuracy: {acc_all*100:.1f}%")

    # ============================================================
    # Try to identify clean values
    # ============================================================
    print("\n--- Trying to find clean parameter values ---")
    print(f"Raw solution: {theta}")

    # Try rounding to various precisions
    for prec in [0.5, 0.25, 0.1, 0.05, 0.025, 0.01]:
        rounded = np.round(theta / prec) * prec
        acc_r = score_params(rounded, races_all)
        print(f"  Round to {prec}: {rounded.tolist()} → acc={acc_r*100:.1f}%")

    # ============================================================
    # Score across multiple files
    # ============================================================
    print("\n--- Multi-file Validation ---")
    for i, fpath in enumerate(files[:5]):
        with open(fpath) as f:
            races_f = json.load(f)
        acc_f = score_params(theta, races_f)
        print(f"  File {i} ({os.path.basename(fpath)}): {acc_f*100:.1f}%")

    # ============================================================
    # Print final parameters
    # ============================================================
    print("\n=== FINAL PARAMETERS ===")
    labels = ['off_SOFT', 'off_MEDIUM', 'off_HARD', 'eff_deg_SOFT', 'eff_deg_MEDIUM', 'eff_deg_HARD']
    for l, v in zip(labels, theta):
        print(f"  {l}: {v:.8f}")

    # Separate temp_coeff if possible
    # Since eff_deg[c] = deg[c] * temp_coeff, and we want clean values,
    # find the GCD of the eff_degs to extract temp_coeff
    eff_degs = theta[3:6]
    print(f"\n  Effective deg rates (deg * temp_coeff): {eff_degs}")
    # Try common temp_coeff values
    for tc in [0.001, 0.002, 0.003, 0.005, 0.01, 0.0033, 0.0025]:
        deg_rates = eff_degs / tc
        print(f"  If temp_coeff={tc}: deg_rates = {deg_rates}")

    # Save result
    out = {
        'formula': 'lap_time = base + off[c] + eff_deg[c] * tire_age * track_temp',
        'params': {l: float(v) for l, v in zip(labels, theta)},
        'note': 'eff_deg = deg * temp_coeff combined'
    }
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), 'lp_params.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to analysis/lp_params.json")


if __name__ == '__main__':
    main()
