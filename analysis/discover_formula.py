#!/usr/bin/env python3
"""
Formula Discovery Script for F1 Race Simulator

Strategy:
1. Load historical races
2. Define parametric lap time formula
3. Use scipy.optimize to find parameters that correctly rank all drivers
4. Validate discovered formula against all races
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import minimize, differential_evolution
import glob

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'historical_races')

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

def compute_driver_total_time(strategy, race_config, params):
    """
    Compute total race time for a single driver.

    params dict keys:
      soft_offset, medium_offset, hard_offset  (base speed per compound)
      soft_deg, medium_deg, hard_deg           (degradation rate per lap per compound)
      temp_coeff                                (temperature multiplier on degradation)
    """
    total_laps = race_config['total_laps']
    base_lap_time = race_config['base_lap_time']
    pit_lane_time = race_config['pit_lane_time']
    track_temp = race_config['track_temp']

    compound_offset = {
        'SOFT':   params['soft_offset'],
        'MEDIUM': params['medium_offset'],
        'HARD':   params['hard_offset'],
    }
    compound_deg = {
        'SOFT':   params['soft_deg'],
        'MEDIUM': params['medium_deg'],
        'HARD':   params['hard_deg'],
    }
    temp_coeff = params['temp_coeff']

    # Build pit stop map: lap -> new tire
    pit_map = {stop['lap']: stop['to_tire'] for stop in strategy.get('pit_stops', [])}
    pit_laps = set(pit_map.keys())

    current_tire = strategy['starting_tire']
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1  # increments BEFORE lap time calc (age=1 on lap 1)

        offset = compound_offset[current_tire]
        deg_rate = compound_deg[current_tire]

        # Degradation formula: deg_rate * tire_age * temp_coeff * track_temp
        lap_time = base_lap_time + offset + deg_rate * tire_age * temp_coeff * track_temp
        total_time += lap_time

        # Pit stop at end of this lap
        if lap in pit_laps:
            total_time += pit_lane_time
            current_tire = pit_map[lap]
            tire_age = 0

    return total_time


def compute_race_order(race, params):
    """Compute predicted finishing order for a race."""
    race_config = race['race_config']
    strategies = race['strategies']

    times = []
    for pos_key, strategy in strategies.items():
        total_time = compute_driver_total_time(strategy, race_config, params)
        times.append((strategy['driver_id'], total_time))

    times.sort(key=lambda x: x[1])
    return [driver_id for driver_id, _ in times]


def score_params(params_list, races):
    """Score params: fraction of races where predicted order matches actual."""
    params = {
        'soft_offset':   params_list[0],
        'medium_offset': params_list[1],
        'hard_offset':   params_list[2],
        'soft_deg':      params_list[3],
        'medium_deg':    params_list[4],
        'hard_deg':      params_list[5],
        'temp_coeff':    params_list[6],
    }

    correct = 0
    for race in races:
        predicted = compute_race_order(race, params)
        actual = race['finishing_positions']
        if predicted == actual:
            correct += 1
    return correct / len(races)


def loss_func(params_list, races):
    """Loss: number of incorrectly ordered pairs across all races (lower = better)."""
    params = {
        'soft_offset':   params_list[0],
        'medium_offset': params_list[1],
        'hard_offset':   params_list[2],
        'soft_deg':      params_list[3],
        'medium_deg':    params_list[4],
        'hard_deg':      params_list[5],
        'temp_coeff':    params_list[6],
    }

    wrong_pairs = 0
    for race in races:
        race_config = race['race_config']
        strategies = race['strategies']
        actual = race['finishing_positions']

        # Compute times
        times = {}
        for pos_key, strategy in strategies.items():
            did = strategy['driver_id']
            times[did] = compute_driver_total_time(strategy, race_config, params)

        # Check each consecutive pair in the actual finishing order
        for i in range(len(actual) - 1):
            d_fast = actual[i]
            d_slow = actual[i + 1]
            if times[d_fast] >= times[d_slow]:
                wrong_pairs += 1

    return wrong_pairs


def load_races(n_files=1, max_per_file=1000):
    """Load races from historical data."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))[:n_files]
    all_races = []
    for f in files:
        with open(f) as fp:
            races = json.load(fp)
        all_races.extend(races[:max_per_file])
    print(f"Loaded {len(all_races)} races from {len(files)} file(s)")
    return all_races


def main():
    print("=" * 60)
    print("F1 Formula Discovery - Phase 1: Try Simple Linear Model")
    print("Formula: lap_time = base + compound_offset[c] + deg_rate[c] * tire_age * temp_coeff * track_temp")
    print("=" * 60)

    # Load small subset first
    races = load_races(n_files=1, max_per_file=200)

    # Initial guess (reasonable F1 values)
    # MEDIUM as reference (offset=0), SOFT faster (negative), HARD slower (positive)
    # Degradation: SOFT degrades more, HARD less
    x0 = [
        -0.5,   # soft_offset
         0.0,   # medium_offset (reference)
         0.5,   # hard_offset
         0.05,  # soft_deg
         0.02,  # medium_deg
         0.01,  # hard_deg
         0.001, # temp_coeff
    ]

    # Quick test with initial guess
    acc = score_params(x0, races)
    print(f"\nInitial guess accuracy: {acc*100:.1f}%")
    print(f"Initial loss: {loss_func(x0, races)}")

    # Grid search over common clean values used in such competitions
    print("\n--- Grid Search for Clean Parameters ---")
    best_acc = 0
    best_params = x0

    # Try common round-number parameter sets
    soft_offsets = [-0.3, -0.5, -1.0, -1.5, -2.0]
    hard_offsets = [0.3, 0.5, 1.0, 1.5, 2.0]
    soft_degs = [0.02, 0.05, 0.1, 0.15, 0.2]
    med_degs  = [0.01, 0.02, 0.03, 0.05, 0.07]
    hard_degs = [0.005, 0.01, 0.015, 0.02, 0.03]
    temp_coeffs = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]

    # Quick coarse grid
    for s_off in soft_offsets:
        for h_off in hard_offsets:
            for s_deg in [0.05, 0.1, 0.15]:
                for m_deg in [0.02, 0.03]:
                    for h_deg in [0.01, 0.015]:
                        for t_c in [0.001, 0.002, 0.003]:
                            p = [s_off, 0.0, h_off, s_deg, m_deg, h_deg, t_c]
                            acc = score_params(p, races[:50])
                            if acc > best_acc:
                                best_acc = acc
                                best_params = p
                                print(f"  New best: {acc*100:.1f}% | params={p}")

    print(f"\nBest after grid search: {best_acc*100:.1f}%")
    print(f"Best params: {best_params}")

    # Refine with scipy optimize
    print("\n--- Scipy Differential Evolution Optimization ---")
    bounds = [
        (-3.0, 0.0),   # soft_offset
        (-0.5, 0.5),   # medium_offset
        (0.0, 3.0),    # hard_offset
        (0.01, 0.5),   # soft_deg
        (0.005, 0.2),  # medium_deg
        (0.001, 0.1),  # hard_deg
        (0.0001, 0.02),# temp_coeff
    ]

    result = differential_evolution(
        loss_func,
        bounds,
        args=(races[:200],),
        maxiter=100,
        popsize=15,
        seed=42,
        disp=True,
        tol=0.001,
    )

    print(f"\nOptimization result: {result.x}")
    print(f"Loss: {result.fun}")

    opt_params = list(result.x)
    acc = score_params(opt_params, races)
    print(f"Accuracy on 200 races: {acc*100:.1f}%")

    # Test on more races
    all_races = load_races(n_files=3, max_per_file=1000)
    acc_all = score_params(opt_params, all_races)
    print(f"Accuracy on {len(all_races)} races: {acc_all*100:.1f}%")

    print("\n--- Discovered Parameters ---")
    param_names = ['soft_offset', 'medium_offset', 'hard_offset', 'soft_deg', 'medium_deg', 'hard_deg', 'temp_coeff']
    for name, val in zip(param_names, opt_params):
        print(f"  {name}: {val:.6f}")

    return opt_params


if __name__ == '__main__':
    main()
